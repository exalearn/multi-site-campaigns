from functools import partial, update_wrapper
from threading import Event, Lock
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from random import shuffle, sample
from pathlib import Path
from typing import Dict, Any
from dataclasses import asdict
import hashlib
import logging
import argparse
import shutil
import json
import sys

import ase
from ase.db import connect
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from colmena.models import Result
from colmena.queue import PipeQueues, ColmenaQueues
from colmena.thinker import BaseThinker, event_responder, result_processor, ResourceCounter, task_submitter
from colmena.task_server.funcx import FuncXTaskServer
from funcx import FuncXClient
import proxystore as ps
import numpy as np
import torch

from fff.learning.gc.ase import SchnetCalculator
from fff.learning.gc.functions import GCSchNetForcefield
from fff.learning.gc.models import SchNet, load_pretrained_model
from fff.learning.util.messages import TorchMessage
from fff.sampling.mctbp import MCTBP
from fff.sampling.md import MolecularDynamics
from fff.sampling.mhm import MHMSampler
from fff.simulation import run_calculator
from fff.simulation.utils import read_from_string, write_to_string

logger = logging.getLogger('main')


@dataclass
class Trajectory:
    """Tracks the state of searching along individual trajectories

    We mark the starting point, the last point produced from sampling,
    and the last point we produced that has been validated
    """
    id: int  # ID number of the
    starting: ase.Atoms  # Starting point of the trajectory
    current_timestep = 0  # How many timesteps have been used so far
    last_validated: ase.Atoms = None  # Last validated point on the trajectory
    current: ase.Atoms = None  # Last point produced along the trajectory
    last_run_length: int = 0  # How long between current and last_validated
    name: str = None  # Name of the trajectory

    def __post_init__(self):
        self.last_validated = self.current = self.starting

    def update_current_structure(self, strc: ase.Atoms, run_length: int):
        """Update the structure that has yet to be updated

        Args:
            strc: Structure produced by sampling
            run_length: How many timesteps were performed in sampling run
        """
        self.current = strc.copy()
        self.last_run_length = run_length

    def set_validation(self, success: bool):
        """Set whether the trajectory was successfully validated

        Args:
            success: Whether the validation was successful
        """
        if success:
            self.last_validated = self.current  # Move the last validated forward
            self.current_timestep += self.last_run_length


@dataclass
class SimulationTask:
    atoms: ase.Atoms  # Structure to be run
    traj_id: int  # Which trajectory this came from
    ml_eng: float  # Energy predicted from machine learning model
    ml_std: float | None = None  # Uncertainty of the model


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
            result.task_info['result_proxy_stats'] = stats

        # Store the Xfer ID
        if isinstance(store, ps.store.globus.GlobusStore):
            key = ps.proxy.get_key(obj)
            result.task_info['result_transfer_id'] = key.split(":")[0]


class Thinker(BaseThinker):
    """Class that schedules work on the HPC"""

    def __init__(
            self,
            queues: ColmenaQueues,
            out_dir: Path,
            db_path: Path,
            search_path: Path,
            model: SchNet,
            energy_tolerance: float,
            max_force: float,
            min_run_length: int,
            max_run_length: int,
            samples_per_run: int,
            infer_chunk_size: int,
            infer_pool_size: int,
            n_to_run: int,
            n_models: int,
            n_qc_workers: int,
            retrain_freq: int,
            ps_names: Dict[str, str],
    ):
        """
        Args:
            queues: Queues to send and receive work
            out_dir: Directory in which to write output files
            db_path: Path to the training data which has been collected so far
            search_path: Path to a databases of geometries to use for searching
            model: Initial model being trained. All further models will be trained using these starting weights
            infer_chunk_size: Number of structures to evaluate per each inference task
            infer_pool_size: Number of inference chunks to perform before selecting new tasks
            n_to_run: Number of simulations to run
            n_models: Number of models to train in the ensemble
            energy_tolerance: How large of an energy difference to accept when auditing
            max_force: Structures with forces larger than this threshold will be excluded from training sets
            min_run_length: Minimum length of sampling runs
            max_run_length: Minimum length of sampling runs
            samples_per_run: Number of samples to produce during a sampling run
            retrain_freq: How often to trigger retraining
            ps_names: Mapping of task type to ProxyStore object associated with it
        """
        # Make the resource tracker
        #  For now, we only control resources over how many samplers are run at a time
        rec = ResourceCounter(n_qc_workers, ['sample', 'simulate'])
        super().__init__(queues, resource_counter=rec)

        # Save key configuration
        self.n_qc_workers = n_qc_workers
        self.db_path = db_path
        self.starting_model = model
        self.ps_names = ps_names
        self.n_models = n_models
        self.out_dir = out_dir
        self.energy_tolerance = energy_tolerance
        self.num_to_run = n_to_run
        self.infer_chunk_size = infer_chunk_size
        self.infer_pool_size = infer_pool_size
        self.retrain_freq = retrain_freq
        self.min_run_length = min_run_length
        self.max_run_length = max_run_length
        self.samples_per_run = samples_per_run
        self.max_force = max_force

        # Load in the search space
        with connect(search_path) as db:
            self.search_space = [Trajectory(i, x.toatoms(), name=x.get('filename', f'traj-{i}')) for i, x in enumerate(db.select(''))]
            shuffle(self.search_space)
            self.search_space = deque(self.search_space)
        self.logger.info(f'Loaded a search space of {len(self.search_space)} geometries at {search_path}')

        # State that evolves as we run
        self.training_round = 0
        self.inference_round = 0
        self.num_complete = 0
        self.run_length = min_run_length
        self.audit_results: deque[float] = deque(maxlen=50)  # Audit Error / Run Length

        # Storage for inference tasks and results
        self.inference_pool: list[ase.Atoms] = []  # List of objects ready for inference
        self.inference_results: dict[int, tuple[list[ase.Atoms], list[Result | None]]] = {}  # Results from inf batches
        self.inference_complete: list[tuple[list[ase.Atoms], list[Result]]] = []  # Complete, successful inf batches

        # Create a proxy for the starting model. It never changes
        #  We store it as a TorchMessage object which can be deserialized more easily
        if 'train' in self.ps_names:
            train_store = ps.store.get_store(self.ps_names['train'])
            self.starting_model_proxy = train_store.proxy(TorchMessage(self.starting_model))
        else:
            self.starting_model_proxy = TorchMessage(self.starting_model)

        # Create a proxy for the "active" model that we'll use to generate trajectories
        if 'sample' in self.ps_names:
            sample_store = ps.store.get_store(self.ps_names['sample'])
            self.active_model_proxy = sample_store.proxy(SchnetCalculator(self.starting_model))
        else:
            self.active_model_proxy = SchnetCalculator(self.starting_model)

        # Proxies for the inference models
        self.inference_proxies: list[ps.proxy.Proxy | None] = [None] * self.n_models  # None until first trained

        # Coordination between threads
        #  Communication from the training tasks
        self.training_incomplete = 0  # Number of training tasks that are incomplete
        self.start_training = Event()  # Starts training on the latest data
        self.training_complete = Event()
        self.active_updated = False  # Whether the active model has already been updated for this batch
        self.sampling_ready = Event()  # Starts sampling only after the first models are read
        self.inference_ready = Event()  # Starts inference only after all models are ready

        #  Coordination between sampling and simulation
        self.has_tasks = Event()  # Whether there are tasks ready to submit
        self.task_queue_audit: list[SimulationTask] = []  # Tasks for checking trajectories
        self.task_queue_active: deque[SimulationTask] = deque(maxlen=self.num_to_run)  # Tasks for best training data
        self.task_queue_lock = Lock()  # Locks both of the task queues
        self.reallocating = Event()  # Marks that we are re-allocating resources
        self.to_audit: dict[int, Trajectory] = {}  # List of trajectories being audited

        # Initialize the system settings
        self.start_training.set()  # Start by training the model
        self.rec.reallocate(None, 'sample', n_qc_workers)  # Start with all devoted to sampling

    @event_responder(event_name='start_training')
    def train_models(self):
        """Submit the models to be retrained"""
        self.logger.info('TIMING - Start train_models')
        self.training_complete.clear()
        self.training_round += 1
        self.logger.info(f'Started training batch {self.training_round}')
        self.active_updated = False

        # Load in the training dataset
        with connect(self.db_path) as db:
            logger.info(f'Connected to a database with {len(db)} entries at {self.db_path}')
            all_examples = np.array([x.toatoms() for x in db.select("")], dtype=object)
        self.logger.info(f'Loaded {len(all_examples)} training examples')

        # Remove the unrealistic structures
        if self.max_force is not None:
            all_examples = [a for a in all_examples if np.abs(a.get_forces()).max() < self.max_force]
            self.logger.info(f'Reduced the number of training examples to {len(all_examples)} with forces less than {self.max_force:.2f} eV/A.')

        # Sample the training sets and proxy them
        train_sets = []
        valid_sets = []
        n_train = int(len(all_examples) * 0.9)
        for _ in range(self.n_models):
            shuffle(all_examples)
            train_sets.append(all_examples[:n_train])
            valid_sets.append(all_examples[n_train:])

        if 'train' in self.ps_names:
            store = ps.store.get_store(self.ps_names['train'])  # TODO (wardlt): Store stores not names?
            train_sets = store.proxy_batch(train_sets)
            valid_sets = store.proxy_batch(valid_sets)

        # Send off the models to be trained
        for i, train_set in enumerate(train_sets):
            # Send a training job
            self.queues.send_inputs(
                self.starting_model_proxy,
                train_set,
                valid_sets[i],
                method='train',
                topic='train',
                task_info={'model_id': i, 'training_round': self.training_round, 'train_size': len(all_examples)}
            )
            self.training_incomplete += 1
        self.logger.info('TIMING - Finish train_models')

    @result_processor(topic='train')
    def store_models(self, result: Result):
        """Store a model once it finishes updating"""
        self.logger.info('TIMING - Start store_models')
        model_id = result.task_info["model_id"]
        self.logger.info(f'Received result from model {model_id}. Success: {result.success}')

        # Save the model to disk
        proxy = result.value
        if result.success:
            # Unpack the result and training history
            model_msg, train_log = result.value

            # Store the result to disk
            model_dir = self.out_dir / 'models'
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / f'model-{model_id}-round-{self.training_round}'
            with open(model_path, 'wb') as fp:
                torch.save(model_msg.get_model(), fp)
            self.logger.info(f'Saved model to: {model_path}')

            # Save the training data
            with open(self.out_dir / 'training-history.json', 'a') as fp:
                print(json.dumps(train_log.to_dict(orient='list')), file=fp)

            # Update the "active" model
            if not self.active_updated:
                # Update the active model
                if 'sample' in self.ps_names:
                    sample_store = ps.store.get_store(self.ps_names['sample'])
                    sample_store.evict(ps.proxy.get_key(self.active_model_proxy))
                    self.active_model_proxy = sample_store.proxy(SchnetCalculator(model_msg.get_model()))
                else:
                    self.active_model_proxy = SchnetCalculator(model_msg.get_model())

                self.active_updated = True
                self.logger.info('Updated the active model')

                # Signals that inference can start now that we've saved one model
                self.sampling_ready.set()

            if 'train' in self.ps_names:
                # Evict the previous inference model
                inf_store = ps.store.get_store(self.ps_names['train'])
                prev_model = self.inference_proxies[model_id]
                if prev_model is not None:
                    prev_key = ps.proxy.get_key(prev_model)
                    inf_store.evict(prev_key)
                    self.logger.info(f'Evicted the previous model for model {model_id}')

                # Store the next model
                self.inference_proxies[model_id] = inf_store.proxy(model_msg)
            else:
                self.inference_proxies[model_id] = model_msg
            self.logger.info(f'Stored the proxy for model {model_id}')

            # Check if we have a full batch of models
            if not any(x is None for x in self.inference_proxies):
                if not self.inference_ready.is_set():
                    self.inference_ready.set()
                    self.logger.info('Inference is now eligible to start')

        # Check whether training is complete
        self.training_incomplete -= 1
        if self.training_incomplete == 0:
            self.logger.info('All models are trained. Marking that training is complete')
            self.submit_inference()  # Command inference to start first
            self.training_complete.set()
        else:
            self.logger.info(f'{self.training_incomplete} models left to train')

        # Save the result to disk
        _get_proxy_stats(proxy, result)
        with open(self.out_dir / 'training-results.json', 'a') as fp:
            print(result.json(exclude={'inputs', 'value'}), file=fp)

        # Stop if the model training failed
        assert result.success, result.failure_info.exception

        self.logger.info('TIMING - End store_models')

    @task_submitter(task_type='sample')
    def submit_sampler(self):
        """Perform molecular dynamics to generate new structures"""
        self.logger.info('TIMING - Start submit_sampler')
        self.sampling_ready.wait()

        # Pick the next eligible trajectory and start from the last validated structure
        trajectory = self.search_space.popleft()
        starting_point = trajectory.starting

        # Initialize the structure if need be
        if trajectory.current_timestep == 0:
            MaxwellBoltzmannDistribution(starting_point, temperature_K=100)
            self.logger.info('Initialized temperature to 100K')

        # Add the structure to a list of those being validated
        self.to_audit[trajectory.id] = trajectory

        # Determine the run length based on observations of errors
        if len(self.audit_results) > self.n_qc_workers:
            # Predict run length given audit error
            error_per_step = np.median(self.audit_results)
            target_error = self.energy_tolerance * 2
            estimated_run_length = int(target_error / error_per_step)
            self.logger.info(f'Estimated run length of {estimated_run_length} steps to have an error of {target_error:.3f} eV/atom')
            self.run_length = max(self.min_run_length, min(self.max_run_length, estimated_run_length))  # Keep to within the user-defined bounds

        # Submit it with the latest model
        self.logger.info(f'Running trajectory {trajectory.id} for {self.run_length} steps '
                         f'starting at {trajectory.current_timestep}')
        self.queues.send_inputs(
            starting_point,
            self.run_length,
            self.active_model_proxy,
            method='run_sampling',
            topic='sample',
            task_info={'traj_id': trajectory.id, 'run_length': self.run_length}
        )
        self.logger.info('TIMING - Finish submit_sampler')

    def _log_queue_sizes(self):
        """Log the size fo the result queues"""
        self.logger.info(f'Queue sizes - Audit: {len(self.task_queue_audit)}, Active: {len(self.task_queue_active)}')

    @result_processor(topic='sample')
    def store_sampling_results(self, result: Result):
        """Store the results of a sampling run"""
        self.logger.info('TIMING - Start store_sampling_results')

        traj_id = result.task_info['traj_id']
        self.logger.info(f'Received sampling result for trajectory {traj_id}. Success: {result.success}')

        # Determine whether we should re-allocate resources
        if len(self.task_queue_audit) > self.n_qc_workers * 2 and \
                self.rec.allocated_slots('sample') > 0 and \
                not self.reallocating.is_set():
            self.reallocating.set()
            self.logger.info('We have enough sampling tasks, reallocating resources to simulation')
            self.rec.reallocate('sample', 'simulate', 1, block=False, callback=self.reallocating.clear)
        self.rec.release('sample', 1)

        # If successful, submit the structures for auditing
        proxy = result.value
        if result.success:
            audit, traj = result.value
            self.logger.info(f'Produced {len(traj)} new structures')

            # Save how many were produced
            result.task_info['num_produced'] = len(traj)  # First was the initial structure

            # Down-sample to target count if needed
            if len(traj) > self.samples_per_run:
                traj = sample(traj, self.samples_per_run)
                self.logger.info(f'Downsampled to {len(traj)}')

            # Update the state of the trajectory
            self.to_audit[traj_id].update_current_structure(audit, result.task_info['run_length'])

            # Add the latest one to the audit list
            with self.task_queue_lock:
                self.task_queue_audit.append(SimulationTask(
                    atoms=traj[-1], traj_id=traj_id, ml_eng=traj[-1].get_potential_energy()
                ))
                self.has_tasks.set()
                self._log_queue_sizes()

            # Store the trajectory ID as information about the atoms object
            for a in traj:
                a.info['traj_id'] = traj_id

            # Extend the current list of candidates
            self.inference_pool.extend(traj)
            self.logger.info(f'Inference pool now has {len(self.inference_pool)} candidates')

            # Submit if we are not training
            if self.training_complete.is_set():
                self.submit_inference()
        else:
            # If not, push it to the back of the queue
            traj = self.to_audit.pop(traj_id)
            self.search_space.append(traj)

        # Save the result to disk
        _get_proxy_stats(proxy, result)
        with open(self.out_dir / 'sampling-results.json', 'a') as fp:
            print(result.json(exclude={'inputs', 'value'}), file=fp)

        self.logger.info('TIMING - Finish store_sampling_results')

    def submit_inference(self):
        """Submit a list of tasks for inference

        Called by other agents when appropriate
        """
        if self.done.is_set():
            return

        # Submit inference chunks if possible
        while len(self.inference_pool) > self.infer_chunk_size and self.inference_ready.is_set():
            # Split off a chunk
            shuffle(self.inference_pool)  # Nearby samples are correlated, this breaks that up
            inf_chunk = self.inference_pool[:self.infer_chunk_size]
            del self.inference_pool[:self.infer_chunk_size]

            # Proxy the inference chunk
            if 'infer' in self.ps_names:
                store = ps.store.get_store(self.ps_names['infer'])
                inf_proxy = store.proxy(inf_chunk)
            else:
                inf_proxy = inf_chunk

            # Prepare storage for the outputs
            #  Includes a list of the structures and a placeholder for the results
            self.inference_results[self.inference_round] = (inf_chunk, [None] * self.n_models)

            # Submit inference for each model
            for model_id, model_msg in enumerate(self.inference_proxies):
                self.queues.send_inputs(
                    model_msg, inf_proxy, method='evaluate', topic='infer',
                    task_info={'model_id': model_id, 'infer_id': self.inference_round}
                )

            # Increment the inference chunk ID
            self.logger.info(f'Submitted inference round {self.inference_round}')
            self.inference_round += 1

    @result_processor(topic='infer')
    def store_inference(self, result: Result):
        """Collect the results from inference tasks.

        Once all models from an inference batch have completed, check if all were successful.
        Once enough inference batches have completed successfully, pick the next round of tasks"""

        self.logger.info('TIMING - Start collect_inference')
        # Get the batch information
        infer_id = result.task_info['infer_id']
        model_id = result.task_info['model_id']
        proxy = result.value
        self.logger.info(f'Received inference batch {infer_id} model {model_id}. Success: {result.success}')

        # If first result from batch, create storage
        my_batch = self.inference_results[infer_id]

        # Allocate the result in the appropriate spot
        my_batch[1][model_id] = result
        num_complete = sum(x is not None for x in my_batch[1])
        self.logger.info(f'Completed inference from {num_complete}/{self.n_models} models')

        # Check if all results are complete
        if num_complete == self.n_models:
            # If so, remove it from the dictionary holding results
            del self.inference_results[infer_id]

            # If all were successful, add it to the ready queue
            all_success = all(x.success for x in my_batch[1])
            self.logger.info(f'All results from {infer_id} have finished. All successful: {all_success}')
            if all_success:
                self.inference_complete.append(my_batch)
                self.logger.info(f'Completed {len(self.inference_complete)}/{self.infer_pool_size} inference batches')

        # If enough batches have completed, create a new task pool
        if len(self.inference_complete) >= self.infer_pool_size:
            self.logger.info('Selecting new tasks using active learning')

            # Consolidate the structures and predictions across batches
            all_strcs = []
            all_preds = []
            all_traj_id = []
            for strcs, preds in self.inference_complete:
                all_strcs.extend(strcs)
                all_traj_id.extend([s.info['traj_id'] for s in strcs])  # Store the trajectory IDs

                # Store the energy predictions. They are the first return value
                #  We make them into a (n_strcs x n_models) array below
                all_preds.append(np.array([p.value[0] for p in preds]).T)
            all_preds = np.vstack(all_preds)  # Combine
            self.logger.info(f'Consolidated an {all_preds.shape} array of {len(all_strcs)} choices to select from')

            # Clear the inference list now that we're done
            self.inference_complete.clear()

            # Perform the active learning step
            selected_structures = self._select_structures(all_strcs, all_preds, all_traj_id)
            self.logger.info(f'Selected a set of {len(selected_structures)} updated structures')

            # Update the task list
            with self.task_queue_lock:
                self.task_queue_active.extend(selected_structures)
            self.logger.info('Updated task queue')
            self._log_queue_sizes()

        # Store the results to disk
        _get_proxy_stats(proxy, result)
        with open(self.out_dir / 'inference-results.json', 'a') as fp:
            print(result.json(exclude={'value', 'inputs'}), file=fp)

        self.logger.info('TIMING - Finish collect_inference')

    def _select_structures(self, structures: list[ase.Atoms], energies: np.ndarray, traj_ids: list[int]) \
            -> list[SimulationTask]:
        """Select a list of structures for the next batch

        Number of structures is set by ``self.n_to_run``

        Args:
            structures: List of structures to chose from
            energies: Energies predicted for each structure by each model
            traj_ids: ID for the trajectories from which each structure was sampled
        Returns:
            Top list of structures to run
        """

        # Compute the standard deviation of energy per atom across each model
        n_atoms = np.array([len(a) for a in structures])
        energy_std = np.std(np.divide(energies, n_atoms[:, None]), axis=1)

        # Make the structures and traj_ids a ndarray, so we can slice it easier
        structures = np.array(structures, dtype=object)
        traj_ids = np.array(traj_ids)

        # TODO (wardlt): Screen out molecules too similar to training set

        # Gradually add a list of molecules
        output = []
        while len(output) < self.num_to_run and len(structures) > 2:  # Need at least a few structures to pick from
            # Select the most uncertain structures
            worst_pred_ind = np.argmax(energy_std)
            output.append(SimulationTask(
                atoms=structures[worst_pred_ind],
                traj_id=int(traj_ids[worst_pred_ind]),  # Ensure it is a JSON-compatible int
                ml_eng=energies[worst_pred_ind, :].mean(),
                ml_std=energy_std[worst_pred_ind],
            ))

            # Remove the 5% of predictions that are most correlated to this one
            #  We compute the correlation coefficient "by hand" to only compute it against one row
            energies_minus_mean = energies - energies.mean(axis=1, keepdims=True)  # x - x_mean for each row
            assert energies_minus_mean.mean(axis=1).max() < 1e-6, "Messed up the mean computation"
            worst_energies = energies_minus_mean[worst_pred_ind, :]
            corr = (np.dot(energies_minus_mean, worst_energies) /
                    np.sqrt(np.power(energies_minus_mean, 2).sum(axis=1)
                            * np.power(worst_energies, 2).sum()))
            corr = np.abs(corr)

            #  Use argpartition to find the 95% that are least correlated
            worst_to_exclude = int(len(structures) * 0.05) + 1
            sorted_corrs = np.argpartition(-corr, kth=worst_to_exclude)  # largest corrs up front
            to_pick = sorted_corrs[worst_to_exclude:]
            assert corr[to_pick].mean() <= corr.mean(), "You got the sorting backwards"

            #  Only include the least correlated
            energy_std = energy_std[to_pick]
            energies = energies[to_pick, :]
            traj_ids = traj_ids[to_pick]
            structures = structures[to_pick]

        return output

    @task_submitter(task_type="simulate")
    def submit_simulation(self):
        """Submit a new simulation to check results from sampling/gather new training data"""
        self.logger.info('TIMING - Start submit_simulation')

        # Get a simulation to run
        to_run = None
        task_type = None
        while to_run is None:
            self.has_tasks.wait()
            with self.task_queue_lock:  # Wait for another thread to add structures
                task_type = None

                # Enumerate the lists to choose from
                list_choices = [
                    ('audit', self.task_queue_audit, lambda: self.task_queue_audit.pop(0)),
                    ('active', self.task_queue_active, self.task_queue_active.popleft),
                ]

                shuffle(list_choices)  # Shuffle them
                for _task_type, _task_list, _task_pull in list_choices:  # Iterate in the random order
                    if len(_task_list) > 0:
                        to_run = _task_pull()
                        task_type = _task_type
                        self._log_queue_sizes()
                        break

            # If task_type is None, neither were picked
            if task_type is None:
                self.logger.info('No tasks are available to run. Waiting ...')
                self.has_tasks.clear()  # We don't have any tasks to run

        # Submit it
        self.logger.info(f'Selected a {task_type} to run next')
        atoms = to_run.atoms
        atoms.set_center_of_mass([0, 0, 0])
        xyz = write_to_string(atoms, 'xyz')
        self.queues.send_inputs(xyz, method='run_calculator', topic='simulate',
                                keep_inputs=True,  # The XYZ file is not big
                                task_info={'traj_id': to_run.traj_id, 'task_type': task_type,
                                           'ml_energy': to_run.ml_eng, 'xyz': xyz})
        self.logger.info('TIMING - Finish submit_simulation')

    @result_processor(topic='simulate')
    def store_simulation(self, result: Result):
        """Store the results from a simulation"""
        self.logger.info('TIMING - Start store_simulation')

        # Get the associated trajectory
        traj_id = result.task_info['traj_id']
        self.logger.info(f'Received a simulation from trajectory {traj_id}. Success: {result.success}')

        # Reallocate resources if the task queue is getting too small
        if len(self.task_queue_audit) <= self.n_qc_workers and \
                self.rec.allocated_slots('simulate') > 0 and \
                not self.reallocating.is_set():
            self.reallocating.set()
            self.logger.info('Running low on simulation tasks. Reallocating to sampling')
            self.rec.reallocate('simulate', 'sample', 1, block=False, callback=self.reallocating.clear)
        self.rec.release('simulate', 1)

        # Store the result in the database if successful
        task_type = result.task_info['task_type']
        proxy = result.value
        if result.success:
            # Count the completed calculation
            self.num_complete += 1
            self.logger.info(f'Evaluated {self.num_complete}/{self.num_to_run} structures')
            if self.num_complete >= self.num_to_run:
                self.done.set()

            # Store the simulation energy for later analysis
            atoms: ase.Atoms = read_from_string(result.value, 'json')
            dft_energy = atoms.get_potential_energy()
            result.task_info['dft_energy'] = dft_energy

            # See how things compared to 
            ml_eng = result.task_info['ml_energy']
            difference = abs(ml_eng - dft_energy) / len(atoms)
            result.task_info['difference'] = difference

            # If an audit calculation, check whether the energy is close to the ML prediction
            if task_type == 'audit':
                traj = self.to_audit.pop(traj_id)
                was_successful = difference < self.energy_tolerance
                traj.set_validation(was_successful)
                self.logger.info(f'Audit for run of {traj.last_run_length} steps for {traj_id} complete.'
                                 f' Result: {was_successful}. Difference: {difference:.3f} eV/atom')

                # Update the audit history
                self.audit_results.append(difference / traj.last_run_length)

                # Add the trajectory back to the list to sample from
                self.search_space.append(traj)  # Put it to the back of the list
            else:
                # Just print the performance
                self.logger.info(f'Difference between ML and DFT: {difference:.3f} eV/atom')

            # Store in the training set
            if difference < 1e6:
                # Get information about the trajectory
                if traj_id in self.to_audit:
                    traj = self.to_audit[traj_id]
                else:
                    traj = next(x for x in self.search_space if x.id == traj_id)
                with connect(self.db_path) as db:
                    db.write(atoms, runtime=result.time_running, source=task_type, filename=traj.name)
            else:
                self.logger.info('Difference is too large. Not storing as this structure is unrealistic')

            # Trigger actions based on number of tasks completed
            if self.num_complete % self.retrain_freq == 0:
                if self.training_complete.is_set():
                    self.logger.info('Sufficient data collected to retrain. Triggering training to restart.')
                    self.start_training.set()
                else:
                    self.logger.info('Sufficient data collected to retrain, but training is still underway')

        elif task_type == 'audit':
            # If the calculation failed, we mark the validation as failed
            traj = self.to_audit.pop(traj_id)
            traj.set_validation(False)

            # Also add it back to the search space
            self.search_space.append(traj)

            # Add a large error value to the queue
            self.audit_results.append(10 / traj.last_run_length)  # 10 eV/atom is much larger than our typical error

        # Write output to disk regardless of whether we were successful
        _get_proxy_stats(proxy, result)
        with open(self.out_dir / 'simulation-results.json', 'a') as fp:
            print(result.json(exclude={'value', 'inputs'}), file=fp)


if __name__ == '__main__':
    # User inputs
    parser = argparse.ArgumentParser()

    # Network configuration details
    group = parser.add_argument_group(title='Network Configuration',
                                      description='How to connect to the Redis task queues and task servers, etc')
    group.add_argument("--redishost", default="127.0.0.1", help="Address at which the redis server can be reached")
    group.add_argument("--redisport", default="6379", help="Port on which redis is available")

    # Computational infrastructure information
    group = parser.add_argument_group(title='Compute Infrastructure',
                                      description='Information about how to run the tasks')
    group.add_argument("--ml-endpoint", help='FuncX endpoint ID for model training and interface')
    group.add_argument("--qc-endpoint", help='FuncX endpoint ID for quantum chemistry')
    group.add_argument("--num-qc-workers", type=int, help="Number of workers performing chemistry tasks")
    group.add_argument("--parsl", action='store_true', help='Use Parsl instead of FuncX')
    group.add_argument("--parsl-site", choices=['theta-lambda', 'local'], required=True, help='Which configuration to use for Parsl')

    # Problem configuration
    group = parser.add_argument_group(title='Problem Definition',
                                      description='Defining the search space, models and optimizers-related settings')
    group.add_argument('--starting-model', help='Path to the MPNN h5 files', required=True)
    group.add_argument('--training-set', help='Path to ASE DB used to train the initial models', required=True)
    group.add_argument('--search-space', help='Path to ASE DB of starting structures for molecular dynamics sampling',
                       required=True)
    group.add_argument('--num-to-run', default=100, type=int, help='Total number of simulations to perform')
    group.add_argument('--calculator', choices=['ttm', 'dft'], required=True, help='Method used to create training data.')

    # Parameters related to training the models
    group = parser.add_argument_group(title="Training Settings")
    group.add_argument("--num-epochs", type=int, default=32, help="Maximum number of training epochs")
    group.add_argument("--ensemble-size", type=int, default=8, help="Number of models to train to create ensemble")
    group.add_argument("--retrain-freq", type=int, default=10,
                       help="Restart training after this many new training points")
    group.add_argument('--huber-deltas', type=float, default=(0.1, 10), nargs=2,
                       help="Huber delta for the energy and forces")
    group.add_argument('--max-force', type=float, default=None, help='Maximum force allowed in training set. Used to screen out unrealistic structures')

    # Parameters related to sampling for new structures
    group = parser.add_argument_group(title="Sampling Settings")
    #  TODO (wardlt): Switch to using a different endpoint for simulation and sampling tasks?
    group.add_argument("--sampling-method", choices=['md', 'mctbp', 'mhm'], default='mctbp', help='Method used to sample structures')
    group.add_argument("--num-samplers", type=int, default=1, help="Number of agents to use to sample structures")
    group.add_argument("--min-run-length", type=int, default=1,
                       help="Minimum timesteps to run sampling calculations.")
    group.add_argument("--max-run-length", type=int, default=100,
                       help="Maximum timesteps to run sampling calculations.")
    group.add_argument("--num-frames", type=int, default=50, help="Number of frames to return per sampling run")
    group.add_argument("--energy-tolerance", type=float, default=0.1,
                       help="Maximum allowable energy different to accept results of sampling run.")

    # Parameters related to active learning
    group = parser.add_argument_group(title='Active Learning')
    group.add_argument('--infer-chunk-size', type=int, default=100,
                       help='Number of structures to send together')
    group.add_argument('--infer-pool-size', type=int, default=2,
                       help='Number of inference chunks to complete before picking next tasks')

    # Parameters related to ProxyStore
    known_ps = [None, 'redis', 'file', 'globus']
    group = parser.add_argument_group(title='ProxyStore', description='Settings related to ProxyStore')
    group.add_argument('--simulate-ps-backend', default="file", choices=known_ps,
                       help='ProxyStore backend to use with "simulate" topic')
    group.add_argument('--sample-ps-backend', default="file", choices=known_ps,
                       help='ProxyStore backend to use with "sample" topic')
    group.add_argument('--train-ps-backend', default="file", choices=known_ps,
                       help='ProxyStore backend to use with "train" topic')
    group.add_argument('--ps-threshold', default=1000, type=int,
                       help='Min size in bytes for transferring objects via ProxyStore')
    group.add_argument('--ps-globus-config', default=None,
                       help='Globus Endpoint config file to use with the ProxyStore Globus backend')
    group.add_argument('--no-proxies', action='store_true', help='Skip making any proxies.')

    # Parse the arguments
    args = parser.parse_args()
    run_params = args.__dict__

    # Check that the dataset exists
    with connect(args.training_set) as db:
        assert len(db) > 0
        pass

    # Get the hash of the training data and model
    with open(args.training_set, 'rb') as fp:
        run_params['data_hash'] = hashlib.sha256(fp.read()).hexdigest()
    with open(args.starting_model, 'rb') as fp:
        run_params['model_hash'] = hashlib.sha256(fp.read()).hexdigest()

    # Make the calculator
    if args.calculator == 'dft':
        calc = dict(calc='psi4', method='pbe0-d3', basis='aug-cc-pvdz', num_threads=12)
    elif args.calculator == 'ttm':
        from ttm.ase import TTMCalculator

        calc = TTMCalculator()
    else:
        raise ValueError(f'Calculator not yet supported: {args.calculator}')

    # Prepare the output directory and logger
    start_time = datetime.utcnow()
    params_hash = hashlib.sha256(json.dumps(run_params).encode()).hexdigest()[:6]
    out_dir = Path('runs') / f'{args.calculator}-{args.sampling_method}-{start_time.strftime("%y%b%d-%H%M%S")}-{params_hash}'
    out_dir.mkdir(parents=True)

    # Make a copy of the training data
    train_path = out_dir / 'train.db'
    shutil.copyfile(args.training_set, train_path)

    # Set up the logging
    handlers = [logging.FileHandler(out_dir / 'runtime.log'), logging.StreamHandler(sys.stdout)]


    class ParslFilter(logging.Filter):
        """Filter out Parsl debug logs"""

        def filter(self, record):
            return not (record.levelno == logging.DEBUG and '/parsl/' in record.pathname)


    for h in handlers:
        h.addFilter(ParslFilter())

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO, handlers=handlers)

    logger.info(f'Run directory: {out_dir}')
    with open(out_dir / 'runparams.json', 'w') as fp:
        json.dump(run_params, fp)

    # Load in the model
    starting_model = torch.load(args.starting_model, map_location='cpu')
    logger.info(f'Loaded model from {Path(args.starting_model).resolve()}')
    shutil.copyfile(args.starting_model, out_dir / 'starting_model.pth')

    # Make the PS scratch directory
    ps_file_dir = out_dir / 'proxy-store'
    ps_file_dir.mkdir()

    # Init only the required ProxyStore backends
    ps_backends = {args.simulate_ps_backend, args.sample_ps_backend, args.train_ps_backend}
    logger.info(f'Initializing ProxyStore backends: {ps_backends}')
    if 'redis' in ps_backends:
        ps.store.init_store(ps.store.STORES.REDIS, name='redis', hostname=args.redishost, port=args.redisport, stats=True)
    if 'file' in ps_backends:
        ps.store.init_store(ps.store.STORES.FILE, name='file', store_dir=str(ps_file_dir.absolute()), stats=True)
    if 'globus' in ps_backends:
        if args.ps_globus_config is None:
            raise ValueError('Must specify --ps-globus-config to use the Globus ProxyStore backend')
        endpoints = ps.store.globus.GlobusEndpoints.from_json(args.ps_globus_config)
        ps.store.init_store(ps.store.STORES.GLOBUS, name='globus', endpoints=endpoints, stats=True, timeout=600)
    ps_names = {'simulate': args.simulate_ps_backend, 'sample': args.sample_ps_backend,
                'train': args.train_ps_backend, 'infer': args.train_ps_backend}
    if args.no_proxies:
        ps_names = {}
        logger.info('Not making any proxies')

    # Connect to the redis server
    queues = PipeQueues(topics=['simulate', 'sample', 'train', 'infer'],
                        serialization_method='pickle',
                        keep_inputs=False,
                        proxystore_name=ps_names,
                        proxystore_threshold=args.ps_threshold)


    # Apply wrappers to functions that will be used to fix certain requirements
    def _wrap(func, **kwargs):
        out = partial(func, **kwargs)
        update_wrapper(out, func)
        return out


    schnet = GCSchNetForcefield()

    my_train_schnet = _wrap(schnet.train, num_epochs=args.num_epochs, device='cuda',
                            patience=8, reset_weights=False,
                            huber_deltas=args.huber_deltas)
    my_eval_schnet = _wrap(schnet.evaluate, device='cuda')
    my_run_simulation = _wrap(run_calculator, calc=calc, temp_path=str(out_dir.absolute()))

    # Determine which sampling method to use
    sampler_kwargs = {}
    if args.sampling_method == 'md':
        sampler = MolecularDynamics()
        sampler_kwargs = {'timestep': 0.1, 'log_interval': 10}
    elif args.sampling_method == 'mctbp':
        sampler = MCTBP()
    elif args.sampling_method == 'mhm':
        mhm_dir = out_dir / 'mhm'
        mhm_dir.mkdir()
        sampler = MHMSampler(scratch_dir=mhm_dir)
    else:
        raise ValueError(f'Sampling method not supported: {args.sampling_method}')

    my_run_dynamics = _wrap(sampler.run_sampling, device='cuda', **sampler_kwargs)

    # Create the task server
    if args.parsl:
        import config as parsl_configs
        from colmena.task_server import ParslTaskServer

        if args.parsl_site == "theta-lambda":
            # Create the resource configuration
            config = parsl_configs.theta_debug_and_lambda(str(out_dir))

            # Assign tasks to the appropriate executor
            methods = [(f, {'executors': ['v100']}) for f in [my_train_schnet, my_eval_schnet]]
            methods.extend([(f, {'executors': ['knl']}) for f in [my_run_simulation, my_run_dynamics]])
        elif args.parsl_site == "local":
            config = parsl_configs.local_parsl(str(out_dir))
            methods = [my_train_schnet, my_eval_schnet, my_run_dynamics, my_run_simulation]
        else:
            raise ValueError(f'Site not yet implemented: {args.parsl_site}')

        # Create the server
        doer = ParslTaskServer(methods, queues, config)
    else:
        fx_client = FuncXClient()  # Authenticate with FuncX
        task_map = dict((f, args.ml_endpoint) for f in [my_train_schnet, my_eval_schnet])
        task_map.update(dict((f, args.qc_endpoint) for f in [my_run_simulation, my_run_dynamics]))
        doer = FuncXTaskServer(task_map, fx_client, queues)

    # Create the thinker
    thinker = Thinker(
        queues,
        out_dir=out_dir,
        db_path=train_path,
        search_path=Path(args.search_space),
        model=starting_model,
        infer_chunk_size=args.infer_chunk_size,
        infer_pool_size=args.infer_pool_size,
        n_to_run=args.num_to_run,
        n_models=args.ensemble_size,
        n_qc_workers=args.num_qc_workers,
        energy_tolerance=args.energy_tolerance,
        min_run_length=args.min_run_length,
        max_run_length=args.max_run_length,
        samples_per_run=args.num_frames,
        retrain_freq=args.retrain_freq,
        ps_names=ps_names,
        max_force=args.max_force
    )
    logging.info('Created the method server and task generator')

    try:
        # Launch the servers
        doer.start()
        thinker.start()
        logging.info('Launched the servers')

        # Wait for the task generator to complete
        thinker.join()
        logging.info('Task generator has completed')
    finally:
        queues.send_kill_signal()

    # Wait for the method server to complete
    doer.join()
    logging.info('Task server has completed')

    # Cleanup ProxyStore backends (i.e., delete objects on the filesystem
    # for file/globus backends)
    for ps_backend in ps_backends:
        if ps_backend is not None:
            ps.store.get_store(ps_backend).cleanup()
    logging.info('ProxyStores cleaned. Exiting now')
