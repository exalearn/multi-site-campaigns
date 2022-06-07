import argparse
import json
import logging
import os
import sys
import time

import numpy as np

from datetime import datetime
from typing import Any

import proxystore as ps
from funcx import FuncXClient
from colmena.task_server.funcx import FuncXTaskServer
from colmena.redis.queue import make_queue_pairs, ClientQueues
from colmena.thinker import BaseThinker, agent


def get_args():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--endpoint', required=True,
                        help='FuncX endpoint for task execution')
    parser.add_argument('--redis-host', default='localhost',
                        help='Redis server IP')
    parser.add_argument('--redis-port', default='6379',
                        help='Redis server port')
    parser.add_argument('--task-input-size', type=float, default=1,
                        help='Data amount to send to tasks [MB]')
    parser.add_argument('--task-output-size', type=float, default=1,
                        help='Data amount to return from tasks [MB]')
    parser.add_argument('--task-interval', type=float, default=0.001,
                        help='Interval between new task generation [s]')
    parser.add_argument('--task-count', type=int, default=100,
                        help='Number of task to generate')
    parser.add_argument('--task-time', type=int, default=0,
                        help='Time for each task [s]')
    parser.add_argument('--reuse-data', action='store_true', default=False,
                        help='Send the same input to each task')
    parser.add_argument('--ps-backend', default=None, choices=[None, 'redis', 'file', 'globus'],
                        help='ProxyStore backend to use')
    parser.add_argument('--ps-threshold', default=0.1, type=float,
                       help='Threshold size for ProxyStore [MB]')
    parser.add_argument('--ps-file-dir', default=None,
                       help='Filesystem dir to use with the ProxyStore file backend')
    parser.add_argument('--ps-globus-config', default=None, 
                        help='Globus Endpoint config file to use with the '
                        'ProxyStore Globus backend')
    parser.add_argument('--output-dir', type=str, default='runs',
                        help='output dir')

    return parser.parse_args()


def empty_array(size: int) -> np.ndarray:
    return np.empty(int(1000 * 1000 * size / 4), dtype=np.float32)


def target_function(data: np.ndarray, output_size: int, runtime: int) -> np.ndarray:
    import numpy as np
    import time

    time.sleep(runtime)  # simulate more imports/setup
    # Check that ObjectProxy acts as the wrapped np object
    assert isinstance(data, np.ndarray), 'got type {}'.format(data)
    #time.sleep(0.005)  # simulate more computation
    return np.empty(int(1000 * 1000 * output_size / 4), dtype=np.float32)


class Thinker(BaseThinker):

    def __init__(self,
                 queue: ClientQueues,
                 task_input_size: int,
                 task_output_size: int,
                 task_count: int,
                 task_interval: float,
                 task_time: int,
                 reuse_data: bool,
                 ):
        super().__init__(queue)
        self.task_input_size = task_input_size
        self.task_output_size = task_output_size
        self.task_count = task_count
        self.task_interval = task_interval
        self.task_time = task_time
        self.reuse_data = reuse_data
        self.count = 0

    def __repr__(self):
        return ("SyntheticDataThinker(\n" + 
                "    task_input_size={}\n".format(self.task_input_size) +
                "    task_output_size={}\n".format(self.task_output_size) +
                "    task_count={}\n".format(self.task_count) +
                "    task_interval={}\n".format(self.task_interval) +
                "    task_time={}\n)".format(self.task_time)
        )

    @agent
    def consumer(self):
        for _ in range(self.task_count):
            result = self.queues.get_result(topic='generate')
            self.logger.info('Got result: {}'.format(str(result).replace('\n', ' ')))

    @agent
    def producer(self):
        if self.reuse_data:
            input_data = empty_array(self.task_input_size)
        while not self.done.is_set():
            if not self.reuse_data:
                input_data = empty_array(self.task_input_size)
            self.queues.send_inputs(
                input_data, self.task_output_size, self.task_time,
                method='target_function', topic='generate'
            )
            self.count += 1
            if self.count >= self.task_count:
                break
            time.sleep(self.task_interval)


if __name__ == "__main__":
    args = get_args()

    out_dir = os.path.join(args.output_dir, 
            datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(out_dir, exist_ok=True)

    # Set up the logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[logging.FileHandler(os.path.join(out_dir, 'runtime.log')),
                  logging.StreamHandler(sys.stdout)]
    )

    logging.info('Args: {}'.format(args))

    # Init ProxyStore backends
    if args.ps_backend == 'redis':
        ps.store.init_store(
            ps.store.STORES.REDIS,
            name='redis',
            hostname=args.redis_host,
            port=args.redis_port
        )
    elif args.ps_backend == 'file':
        if args.ps_file_dir is None:
            raise ValueError(
                'Must specify --ps-file-dir to use the filesystem ProxyStore backend'
            )
        ps.store.init_store(
            ps.store.STORES.FILE, name='file', store_dir=args.ps_file_dir
        )
    elif args.ps_backend == 'globus':
        if args.ps_globus_config is None:
            raise ValueError(
                'Must specify --ps-globus-config to use the Globus ProxyStore backend'
            )
        endpoints = ps.store.globus.GlobusEndpoints.from_json(args.ps_globus_config)
        ps.store.init_store(
            ps.store.STORES.GLOBUS, name='globus', endpoints=endpoints, timeout=60
        )

    # Make the queues
    client_queues, server_queues = make_queue_pairs(
        args.redis_host,
        args.redis_port,
        topics=['generate'],
        serialization_method='pickle',
        keep_inputs=False,
        proxystore_name=args.ps_backend,
        proxystore_threshold=int(args.ps_threshold * 1000 * 1000)
    ) 

    # Create the task servers
    fx_client = FuncXClient()
    doer = FuncXTaskServer({target_function: args.endpoint}, fx_client, server_queues)

    thinker = Thinker(
        queue=client_queues,
        task_input_size=args.task_input_size,
        task_output_size=args.task_output_size,
        task_count=args.task_count,
        task_interval=args.task_interval,
        task_time=args.task_time,
        reuse_data=args.reuse_data,
    )

    logging.info('Created the task server and task generator')
    logging.info(thinker)

    start_time = time.time()

    try:
        # Launch the servers
        doer.start()
        thinker.start()
        logging.info('Launched the servers')

        # Wait for the task generator to complete
        thinker.join()
        logging.info('Task generator has completed')
    finally:
        client_queues.send_kill_signal()

    # Wait for the task server to complete
    doer.join()

    if args.ps_backend is not None:
        ps.store.get_store(args.ps_backend).cleanup()

    # Print the output result
    logging.info('Finished. Runtime = {}s'.format(time.time() - start_time))
