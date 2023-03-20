from dataclasses import asdict
from threading import Event, Lock, Semaphore
from typing import Dict, List, Tuple, Any
from functools import partial, update_wrapper
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from queue import Queue
import argparse
import logging
import hashlib
import json
import sys
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import proxystore as ps
from proxystore.store import register_store
from proxystore.store.file import FileStore
from proxystore.store.globus import GlobusEndpoints, GlobusStore
from proxystore.store.redis import RedisStore
from proxystore.store.multi import MultiStore
from proxystore.store.multi import Policy
from proxystore.store.dim.zmq import ZeroMQStore
from proxystore.store.endpoint import EndpointStore
from proxystore.store.utils import get_key
from rdkit import Chem
from tqdm import tqdm
from colmena.models import Result
from colmena.queue.redis import RedisQueues
from colmena.thinker import BaseThinker, result_processor, event_responder, task_submitter
from colmena.thinker.resources import ResourceCounter
from qcelemental.models import OptimizationResult, AtomicResult

from moldesign.score.nfp import evaluate_mpnn, retrain_mpnn, NFPMessage, custom_objects
from moldesign.store.models import MoleculeData
from moldesign.store.recipes import apply_recipes
from moldesign.utils import get_platform_info

from config import theta_debug_and_chameleon as make_config


def run_simulation(smiles: str, n_nodes: int, spec: str = 'small_basis') -> Tuple[List[OptimizationResult], List[AtomicResult]]:
    """Run the ionization potential computation

    Args:
        smiles: SMILES string to evaluate
        n_nodes: Number of nodes to use
        spec: Name of the quantum chemistry specification
    Returns:
        Relax records for the neutral and ionized geometry
    """
    from moldesign.simulate.functions import generate_inchi_and_xyz, relax_structure
    from moldesign.simulate.specs import get_qcinput_specification

    # Make the initial geometry
    inchi, xyz = generate_inchi_and_xyz(smiles)

    # Make compute spec
    compute_config = {'nnodes': n_nodes, 'cores_per_rank': 2, 'ncores': 64}

    # Get the specification and make it more resilient
    spec, code = get_qcinput_specification(spec)
    if code == "nwchem":
        spec.keywords["dft__iterations"] = 150
        spec.keywords["geometry__noautoz"] = True

    # Compute the neutral geometry and hessian
    neutral_relax = relax_structure(xyz, spec, compute_config=compute_config, charge=0, code=code)

    # Compute the relaxed geometry
    oxidized_relax = relax_structure(neutral_relax.final_molecule.to_string('xyz'), spec, compute_config=compute_config, charge=1, code=code)
    return [neutral_relax, oxidized_relax], []


def _get_proxy_stats(obj: Any, result: Result):
    """Update a Result object with the proxy stats of its result

    Should be run after using the result and just before saving the output

    Args:
        obj: Pointer to value of the result before resolution
        result: Result object to be updated with proxy stats and Globus transfer ID, if available
    """

    if isinstance(obj, ps.proxy.Proxy):
        store = ps.store.get_store(obj)

        # Store the resolution stats
        if store.has_stats:
            stats = store.stats(obj)
            stats = dict((k, asdict(v)) for k, v in stats.items())
            result.task_info['proxy_stats'] = stats


class Thinker(BaseThinker):
    """ML-enhanced optimization loop for molecular design"""

    def __init__(self,
                 queues: RedisQueues,
                 database: List[MoleculeData],
                 search_space: Path,
                 n_to_evaluate: int,
                 n_complete_before_retrain: int,
                 mpnns: List[tf.keras.Model],
                 inference_chunk_size: int,
                 num_qc_workers: int,
                 qc_specification: str,
                 output_dir: str,
                 beta: float,
                 pause_during_update: bool,
                 ps_names: Dict[str, str]):
        """
        Args:
            queues: Queues used to communicate with the method server
            database: Link to the MongoDB instance used to store results
            search_space: Path to a search space of molecules to evaluate
            n_complete_before_retrain: Number of simulations to complete before retraining
            mpnns: List of MPNNs to use for selecting samples
            output_dir: Directory in which to write output results
            pause_during_update: Whether to stop submitting tasks while task list is updating
            ps_names: mapping of topic to proxystore backend to use (or None if not using ProxyStore)
        """
        super().__init__(queues, ResourceCounter(num_qc_workers, ['simulation']), daemon=True)

        # Configuration for the run
        self.inference_chunk_size = inference_chunk_size
        self.n_complete_before_retrain = n_complete_before_retrain
        self.n_evaluated = 0
        self.mpnns = mpnns.copy()
        self.output_dir = Path(output_dir)
        self.beta = beta
        self.ps_names = ps_names
        self.pause_during_update = pause_during_update

        # Get the name of the property given the specification
        if qc_specification == 'xtb':
            self.property_name = 'xtb-vacuum'
        elif qc_specification == 'small_basis':
            self.property_name = 'smb-vacuum-no-zpe'

        # Get the initial database
        self.database = database
        self.logger.info(f'Populated an initial database of {len(self.database)} entries')
        self.train_data_proxy_key = None  # Will be used later

        # Get the target database size
        self.n_to_evaluate = n_to_evaluate

        # List the molecules that have already been searched
        self.already_searched = set([d.identifier['inchi'] for d in self.database])

        # Prepare search space
        self.mols = pd.read_csv(search_space)
        self.mols['dict'] = self.mols['dict'].apply(json.loads)
        self.inference_chunks = np.array_split(self.mols, len(self.mols) // self.inference_chunk_size + 1)
        self.logger.info(f'Split {len(self.mols)} molecules into {len(self.inference_chunks)} chunks for inference')

        # Batch add inference chunks to ProxyStore, as they will be used frequently
        inference_msgs = [chunk['dict'].tolist() for chunk in self.inference_chunks]
        if self.ps_names['infer'] is not None:
            keys = [f'search-{mid}' for mid in range(len(inference_msgs))]
            self.inference_proxies = ps.store.get_store(self.ps_names['infer']).proxy_batch(inference_msgs, subset_tags=['infer'])
        else:
            self.inference_proxies = inference_msgs  # No proxies, just send the message as-is

        # Inter-thread communication stuff
        self.start_inference = Event()  # Mark that inference should start
        self.start_training = Event()  # Mark that retraining should start
        self.task_queue_ready = Event()  # Mark that the task queue is ready
        self.update_in_progress = Event()  # Mark that we are currently re-training the model
        self.update_complete = Event()  # Mark that the update event has finished
        self.task_queue = []  # Holds a list of tasks to be simulated
        self.task_queue_lock = Lock()  # Ensures only one thread edits task queue at a time
        self.ready_models = Queue()
        self.num_training_complete = 0  # Tracks when we are done with training all models
        self.inference_batch = 0
        self.inference_limiter = Semaphore(20)  # Maximum number of inference tasks to send at once (only used without proxystore)

        # Start with training
        self.start_training.set()

        # Allocate all nodes that are under controlled use to simulation
        self.rec.reallocate(None, 'simulation', 'all')

    @task_submitter(task_type='simulation')
    def submit_qc(self):
        # Wait for the first set of tasks to be available
        self.task_queue_ready.wait()

        # If desired, wait until model update is done
        if self.pause_during_update:
            if self.update_in_progress.is_set():
                self.logger.info(f'Waiting until task queue is updated.')
            self.update_complete.wait()

        # Submit the next task
        with self.task_queue_lock:
            inchi, info = self.task_queue.pop(0)
            try:
                mol = Chem.MolFromInchi(inchi)
                smiles = Chem.MolToSmiles(mol)
            except RuntimeError:
                self.logger.error(f'Parse failed for {inchi}')
                raise

            self.logger.info(f'Submitted {smiles} to simulate with NWChem. Run score: {info["ucb"]}')
            self.already_searched.add(inchi)
            self.queues.send_inputs(smiles, task_info=info,
                                    method='run_simulation', keep_inputs=True,
                                    topic='simulate')

    @result_processor(topic='simulate')
    def process_outputs(self, result: Result):
        # Get basic task information
        smiles, = result.args

        # Release nodes for use by other processes
        self.rec.release("simulation", 1)

        # If successful, add to the database
        proxy = result.value
        if result.success:
            # Mark that we've had another complete result
            self.n_evaluated += 1
            self.logger.info(f'Success! Finished screening {self.n_evaluated}/{self.n_to_evaluate} molecules')

            # Determine whether to start re-training
            if self.n_evaluated % self.n_complete_before_retrain == 0:
                if self.update_in_progress.is_set():
                    self.logger.info(f'Waiting until previous training run completes.')
                else:
                    self.logger.info(f'Starting retraining.')
                    self.start_training.set()
            self.logger.info(f'{self.n_complete_before_retrain - self.n_evaluated % self.n_complete_before_retrain} results needed until we re-train again')

            # Store the data in a molecule data object
            self.logger.debug(f'The proxy has been resolved {result.value}')
            data = MoleculeData.from_identifier(smiles=smiles)
            opt_records, hess_records = result.value
            for r in opt_records:
                data.add_geometry(r)
            for r in hess_records:
                data.add_single_point(r)
            data.update_thermochem()
            apply_recipes(data)

            # Add the IPs to the result object
            result.task_info["ip"] = data.oxidation_potential.copy()

            # Add to database
            with open(self.output_dir.joinpath('moldata-records.json'), 'a') as fp:
                print(json.dumps([datetime.now().timestamp(), data.json()]), file=fp)
            self.database.append(data)

            # Write to disk
            with open(self.output_dir.joinpath('qcfractal-records.json'), 'a') as fp:
                for r in opt_records + hess_records:
                    print(r.json(), file=fp)
            self.logger.info(f'Added complete calculation for {smiles} to database.')

            # Mark that we've completed one
            if self.n_evaluated >= self.n_to_evaluate:
                self.logger.info(f'No more molecules left to screen')
                self.done.set()
        else:
            self.logger.info(f'Computations failed for {smiles}. Check JSON file for stacktrace')

        # Write out the result to disk
        _get_proxy_stats(proxy, result)
        with open(self.output_dir.joinpath('simulation-results.json'), 'a') as fp:
            print(result.json(exclude={'value', 'proxystore_kwargs'}), file=fp)
        self.logger.info(f'Processed simulation task.')

    @event_responder(event_name='start_training')
    def train_models(self):
        """Train machine learning models"""
        self.logger.info('Started retraining')

        # Set that a retraining event is in progress
        self.update_complete.clear()
        self.update_in_progress.set()
        self.num_training_complete = 0

        # Make the database
        train_data = dict(
            (d.identifier['smiles'], d.oxidation_potential[self.property_name])
            for d in self.database
            if self.property_name in d.oxidation_potential
        )

        # Proxy it
        ps_name = self.ps_names['train']
        if ps_name is not None:
            train_data_proxy = ps.store.get_store(ps_name).proxy(train_data, subset_tags=['train'])
            self.train_data_proxy_key = get_key(train_data_proxy)
        else:
            train_data_proxy = train_data  # No proxy

        for mid, model in enumerate(self.mpnns):
            self.queues.send_inputs(model.get_config(),
                                    train_data_proxy,
                                    method='retrain_mpnn',
                                    topic='train',
                                    task_info={'model_id': mid},
                                    keep_inputs=False,
                                    input_kwargs={'random_state': mid})
            self.logger.info(f'Submitted model {mid} to train with {len(train_data)} entries')

    @result_processor(topic='train')
    def update_weights(self, result: Result):
        """Process the results of the saved model"""

        # Make sure the run completed
        model_id = result.task_info['model_id']
        proxy = result.value
        if not result.success:
            self.logger.warning(f'Training failed for {model_id}')
        else:
            # Update weights
            weights, history = result.value
            self.mpnns[model_id].set_weights(weights)

            # Print out some status info
            self.logger.info(f'Model {model_id} finished training.')
            with open(self.output_dir.joinpath('training-history.json'), 'a') as fp:
                print(repr(history), file=fp)

        # Send the model to inference
        self.start_inference.set()
        self.ready_models.put(self.mpnns[model_id])

        # Save results to disk
        _get_proxy_stats(proxy, result)
        with open(self.output_dir.joinpath('training-results.json'), 'a') as fp:
            print(result.json(exclude={'inputs', 'value', 'proxystore_kwargs'}), file=fp)

        # Mark that a model has finished training and trigger inference if all done
        self.num_training_complete += 1
        self.logger.info(f'Processed training task. {len(self.mpnns) - self.num_training_complete} models left to go')

        # If all done, evict the training set
        ps_name = self.ps_names['train']
        if ps_name is not None and len(self.mpnns) == self.num_training_complete:
            ps.store.get_store(ps_name).evict(self.train_data_proxy_key)
            self.logger.info('All training completed. Evicted training set')

    @event_responder(event_name='start_inference')
    def launch_inference(self):
        """Submit inference tasks for the yet-unlabelled samples"""

        self.logger.info('Beginning to submit inference tasks')
        # Get the name of the proxy store
        ps_name = self.ps_names['infer']

        # Submit the chunks to the workflow engine
        for mid in range(len(self.mpnns)):

            # Get a model that is ready for inference
            model = self.ready_models.get()

            # Convert it to a pickle-able message
            model_msg = NFPMessage(model)

            # Proxy it once, to be used by all inference tasks
            if ps_name is not None:
                model_msg_proxy = ps.store.get_store(ps_name).proxy(model_msg, subset_tags=['infer'])
            else:
                model_msg_proxy = model_msg  # No proxy

            # Run inference with all segments available
            for cid, (chunk, chunk_msg) in enumerate(zip(self.inference_chunks, self.inference_proxies)):
                # Hack: Submitting tasks too quickly during inference leads to buffer problems w/o proxystore
                if ps_name is None:
                    self.inference_limiter.acquire()

                self.queues.send_inputs(model_msg_proxy, chunk_msg,
                                        topic='infer', method='evaluate_mpnn',
                                        keep_inputs=False,
                                        task_info={'chunk_id': cid, 'chunk_size': len(chunk), 'model_id': mid})
        self.logger.info('Finished submitting molecules for inference')

    @event_responder(event_name='start_inference')
    def selector(self):
        """Re-prioritize the machine learning tasks"""

        #  Make arrays that will hold the output results from each run
        y_pred = [np.zeros((len(x), len(self.mpnns)), dtype=np.float32) for x in self.inference_chunks]

        # Collect the inference runs
        n_tasks = len(self.inference_chunks) * len(self.mpnns)
        for i in range(n_tasks):
            # Wait for a result
            result = self.queues.get_result(topic='infer')
            self.logger.info(f'Received inference task {i + 1}/{n_tasks}')
            proxy = result.value

            # Hack: Control the rate at which we submit tasks w/o ProxyStore 
            if self.ps_names['infer'] is None:
                self.inference_limiter.release()

            # Raise an error if this task failed
            if not result.success:
                raise ValueError(f'Inference failed: {result.failure_info.exception}. Check the logs for further details')

            # Store the outputs
            chunk_id = result.task_info.get('chunk_id')
            model_id = result.task_info.get('model_id')
            y_pred[chunk_id][:, model_id] = np.squeeze(result.value)

            # Save the inference information to disk
            _get_proxy_stats(proxy, result)
            with open(self.output_dir.joinpath('inference-results.json'), 'a') as fp:
                print(result.json(exclude={'value', 'proxystore_kwargs'}), file=fp)

            self.logger.info(f'Processed inference task {i + 1}/{n_tasks}')

        # Compute the mean and std for each prediction
        y_pred = np.concatenate(y_pred, axis=0)
        self._select_molecules(y_pred)

        # Mark that inference is complete
        self.inference_batch += 1

        # Mark that the task list has been updated
        self.update_in_progress.clear()
        self.update_complete.set()
        self.task_queue_ready.set()

    def _select_molecules(self, y_pred):
        """Select a list of molecules given the predictions from each model

        Adds them to the task queue

        Args:
            y_pred: List of predictions for each molecule in self.search_space
        """
        # Compute the average and std of predictions
        y_mean = y_pred.mean(axis=1)
        y_std = y_pred.std(axis=1)

        # Rank compounds according to the upper confidence bound
        molecules = self.mols['inchi'].values
        ucb = y_mean + self.beta * y_std
        sort_ids = np.argsort(-ucb)
        best_list = list(zip(molecules[sort_ids].tolist(),
                             y_mean[sort_ids], y_std[sort_ids], ucb[sort_ids]))

        # Push the list to the task queue
        with self.task_queue_lock:
            self.task_queue = []
            for mol, mean, std, ucb in best_list:
                # Add it to list if not in database or not already in queue
                if mol not in self.already_searched and mol not in self.task_queue:
                    # Note: converting to float b/c np.float32 is not JSON serializable
                    self.task_queue.append((mol, {'mean': float(mean), 'std': float(std), 'ucb': float(ucb),
                                                  'batch': self.inference_batch}))

                # If we reach the target number of molecules, stop adding more
                if len(self.task_queue) >= self.n_to_evaluate:
                    break
        self.logger.info('Updated task list')


if __name__ == '__main__':
    # User inputs
    parser = argparse.ArgumentParser()

    # Network configuration details
    group = parser.add_argument_group(title='Network Configuration', description='How to connect to the Redis task queues and task servers, etc')
    group.add_argument("--redishost", default="127.0.0.1", help="Address at which the redis server can be reached")
    group.add_argument("--redisport", default="6379", help="Port on which redis is available")

    # Computational infrastructure information
    group = parser.add_argument_group(title='Compute Infrastructure', description='Information about how to run the tasks')
    group.add_argument("--ml-endpoint", help='FuncX endpoint ID for model training and interface')
    group.add_argument("--qc-endpoint", help='FuncX endpoint ID for quantum chemistry')
    group.add_argument("--nodes-per-task", default=1, help='Number of nodes per quantum chemistry task. Only needed for NWChem', type=int)
    group.add_argument("--num-qc-workers", required=True, type=int, help="Total number of quantum chemistry workers.")
    group.add_argument("--molecules-per-ml-task", default=10000, type=int, help="Number of molecules per inference chunk")

    # Problem configuration
    group = parser.add_argument_group(title='Problem Definition', description='Defining the search space, models and optimizers-related settings')
    group.add_argument('--qc-specification', default='xtb', choices=['xtb'], help='Which level of quantum chemistry to run')
    group.add_argument('--mpnn-model-path', help='Path to the MPNN h5 files', required=True)
    group.add_argument('--model-count', default=8, help='Number of models to use in the ensemble', type=int)
    group.add_argument('--training-set', help='Path to the molecules used to train the initial models', required=True)
    group.add_argument('--search-space', help='Path to molecules to be screened', required=True)
    group.add_argument("--search-size", default=500, type=int, help="Number of new molecules to evaluate during this search")
    group.add_argument('--retrain-frequency', default=1, type=int, help="Number of completed computations that will trigger a retraining")
    group.add_argument("--beta", default=1, help="Degree of exploration for active learning. This is the beta from the UCB acquistion function", type=float)
    group.add_argument("--pause-during-update", action='store_true', help='Whether to stop running simulations while updating task list')

    # Parameters related to model retraining
    group = parser.add_argument_group(title='Model Training', description='Settings related to model retraining')
    group.add_argument("--learning-rate", default=1e-3, help="Learning rate for re-training the models", type=float)
    group.add_argument('--num-epochs', default=512, type=int, help='Maximum number of epochs for the model training')

    # Parameters related to ProxyStore
    group = parser.add_argument_group(title='ProxyStore', description='Settings related to ProxyStore')
    group.add_argument('--no-proxystore', action='store_true', help='Turn off ProxyStore')
    group.add_argument('--simulate-ps-backend', default=None, choices=[None, 'redis', 'file', 'globus', 'multi'], help='ProxyStore backend to use with "simulate" topic')
    group.add_argument('--infer-ps-backend', default=None, choices=[None, 'redis', 'file', 'globus', 'multi'], help='ProxyStore backend to use with "infer" topic')
    group.add_argument('--train-ps-backend', default=None, choices=[None, 'redis', 'file', 'globus', 'multi'], help='ProxyStore backend to use with "train" topic')
    group.add_argument('--ps-threshold', default=10000, type=int, help='Min size in bytes for transferring objects via ProxyStore')
    group.add_argument('--ps-file-dir', default=None, help='Filesystem directory to use with the ProxyStore file backend')
    group.add_argument('--ps-globus-config', default=None, help='Globus Endpoint config file to use with the ProxyStore Globus backend')
    group.add_argument('--ps-multi-config', default=None, help='MultiStore config file to use with the ProxyStore MultiStore backend')

    # Configuration for optional Parsl backend
    group = parser.add_argument_group(title='Parsl', description='Settings related to Parsl')
    group.add_argument('--use-parsl', action='store_true', help='Use Parsl instead of FuncX')

    # Parse the arguments
    args = parser.parse_args()
    run_params = args.__dict__

    # Load in the models, initial dataset, agent and search space
    models = [
        tf.keras.models.load_model(args.mpnn_model_path, custom_objects=custom_objects)
        for path in tqdm(range(args.model_count), desc='Loading models')
    ]

    # Read in the training set
    with open(args.training_set) as fp:
        database = [MoleculeData.parse_raw(line) for line in fp]

    # Create an output directory with the time and run parameters
    start_time = datetime.utcnow()
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]
    out_dir = os.path.join('runs',
                           f'{args.qc_specification}-N{args.num_qc_workers}-n{args.nodes_per_task}-{params_hash}-{start_time.strftime("%d%b%y-%H%M%S")}')
    os.makedirs(out_dir, exist_ok=False)

    # Save the run parameters to disk
    with open(os.path.join(out_dir, 'run_params.json'), 'w') as fp:
        json.dump(run_params, fp, indent=2)
    with open(os.path.join(out_dir, 'environment.json'), 'w') as fp:
        json.dump(dict(os.environ), fp, indent=2)

    # Save the platform information to disk
    host_info = get_platform_info()
    with open(os.path.join(out_dir, 'host_info.json'), 'w') as fp:
        json.dump(host_info, fp, indent=2)

    # Set up the logging
    handlers = [logging.FileHandler(os.path.join(out_dir, 'runtime.log')),
                logging.StreamHandler(sys.stdout)]


    class ParslFilter(logging.Filter):
        """Filter out Parsl debug logs"""

        def filter(self, record):
            return not (record.levelno == logging.DEBUG and '/parsl/' in record.pathname)


    for h in handlers:
        h.addFilter(ParslFilter())

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO, handlers=handlers)
    logging.info(f'Run directory: {out_dir}')

    # Make the PS files directory inside run directory
    ps_file_dir = os.path.abspath(os.path.join(out_dir, args.ps_file_dir))
    os.makedirs(ps_file_dir, exist_ok=False)
    logging.info(f'Scratch directory for ProxyStore files: {ps_file_dir}')

    # Init ProxyStore backends
    ps_backends = {args.simulate_ps_backend, args.infer_ps_backend, args.train_ps_backend}
    if 'redis' in ps_backends:
        store = RedisStore(name='redis', hostname=args.redishost, port=args.redisport, stats=True)
        register_store(store)
    if 'file' in ps_backends:
        if args.ps_file_dir is None:
            raise ValueError('Must specify --ps-file-dir to use the filesystem ProxyStore backend')
        register_store(FileStore(name='file', store_dir=ps_file_dir, stats=True))
    if 'globus' in ps_backends:
        if args.ps_globus_config is None:
            raise ValueError('Must specify --ps-globus-config to use the Globus ProxyStore backend')
        endpoints = GlobusEndpoints.from_json(args.ps_globus_config)
        register_store(GlobusStore(name='globus', endpoints=endpoints, stats=True, timeout=600))
    if 'multi' in ps_backends:
        if args.ps_multi_config is None:
            raise ValueError('Must specify --ps-multi-config to use the Multi-store ProxyStore backend')
        with open(args.ps_multi_config, 'r') as f:
            endpoints = json.load(f)

        stores = {}
        for uuid, params in endpoints.items():
            if params['store'] == 'zmq':
                s = ZeroMQStore(params['name'], interface=params['interface'], port=params['port'])
                stores[s] = Policy(subset_tags=params['policy-tags'])
            elif params['store'] == 'endpoint':
                params['other_endpoints'].extend(list(endpoints.keys()))
                s = EndpointStore(params['name'], endpoints=params['other_endpoints'])
                stores[s] = Policy(subset_tags=params['policy-tags'])
            elif params['store'] == 'globus':
                endpoints = GlobusEndpoints.from_json(params['config'])
                s = GlobusStore(name='globus', endpoints=endpoints, stats=True, timeout=600)
                stores[s] = Policy(subset_tags=params['policy-tags'])
            else:
                raise NotImplementedError(f'Store {params["store"]} has not yet been enabled for MultiStore.')

        store = MultiStore("multistore", stores=stores)
        register_store(store)

    if args.no_proxystore:
        ps_names = defaultdict(lambda: None)  # No proxystore for no one
    else:
        ps_names = {'simulate': 'multistore', 'infer': 'multistore', 'train': 'multistore'} #{'simulate': args.simulate_ps_backend, 'infer': args.infer_ps_backend, 'train': args.train_ps_backend}

    # Connect to the redis server
    queues = RedisQueues(hostname=args.redishost,
                         port=args.redisport,
                         prefix=start_time.strftime("%d%b%y-%H%M%S"),
                         topics=['simulate', 'infer', 'train'],
                         serialization_method='pickle',
                         keep_inputs=True,
                         proxystore_name=None if args.no_proxystore else ps_names,
                         proxystore_threshold=None if args.no_proxystore else args.ps_threshold)

    # Apply wrappers to functions to affix static settings
    #  Update wrapper changes the __name__ field, which is used by the Method Server
    my_evaluate_mpnn = partial(evaluate_mpnn, batch_size=128)
    my_evaluate_mpnn = update_wrapper(my_evaluate_mpnn, evaluate_mpnn)

    my_retrain_mpnn = partial(retrain_mpnn, num_epochs=args.num_epochs, learning_rate=args.learning_rate, timeout=2700)
    my_retrain_mpnn = update_wrapper(my_retrain_mpnn, retrain_mpnn)

    my_run_simulation = partial(run_simulation, n_nodes=args.nodes_per_task, spec=args.qc_specification)
    my_run_simulation = update_wrapper(my_run_simulation, run_simulation)

    # Create the task servers
    if args.use_parsl:
        from colmena.task_server import ParslTaskServer

        # Create the resource configuration
        config = make_config(out_dir)

        # Assign tasks to the appropriate executor
        methods = [(f, {'executors': ['gpu']}) for f in [my_evaluate_mpnn, my_retrain_mpnn]]
        methods.append((my_run_simulation, {'executors': ['cpu']}))

        # Create the server
        doer = ParslTaskServer(methods, queues, config)
    else:
        from colmena.task_server.funcx import FuncXTaskServer
        from funcx import FuncXClient

        fx_client = FuncXClient()
        task_map = dict((f, args.ml_endpoint) for f in [my_evaluate_mpnn, my_retrain_mpnn])
        task_map[my_run_simulation] = args.qc_endpoint
        doer = FuncXTaskServer(task_map, fx_client, queues)

    # Configure the "thinker" application
    thinker = Thinker(
        queues=queues,
        database=database,
        search_space=args.search_space,
        n_to_evaluate=args.search_size,
        n_complete_before_retrain=args.retrain_frequency,
        mpnns=models,
        inference_chunk_size=args.molecules_per_ml_task,
        num_qc_workers=args.num_qc_workers,
        qc_specification=args.qc_specification,
        output_dir=out_dir,
        beta=args.beta,
        pause_during_update=args.pause_during_update,
        ps_names=ps_names
    )
    logging.info('Created the method server and task generator')

    try:
        # Launch the servers
        #  The method server is a Thread, so that it can access the Parsl DFK
        #  The task generator is a Thread, so that all debugging methods get cast to screen
        doer.start()
        thinker.start()
        logging.info(f'Running on {os.getpid()}')
        logging.info('Launched the servers')

        # Wait for the task generator to complete
        thinker.join()
        logging.info('Task generator has completed')
    finally:
        queues.send_kill_signal()

    # Wait for the method server to complete
    doer.join()

    # Cleanup ProxyStore backends (i.e., delete objects on the filesystem
    # for file/globus backends)
    for ps_backend in ps_backends:
        if ps_backend is not None:
            ps.store.get_store(ps_backend).close()
            logging.info(f'Cleaned up proxystore with {ps_backend} backend')
