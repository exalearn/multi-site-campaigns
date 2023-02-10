# Multi-Site Active Learning for IP Optimization

Re-training machine learning models based on new simulations is major bottleneck in
our [previous molecular design application](https://ieeexplore.ieee.org/abstract/document/9653177/)
and one we can reduce by performing training on specialized hardware.
Here, we illustrate how to implement such an application using Colmena backed by FuncX or Parsl.

## Application Description

Our application implements a prototypical active learning algorithm.
There are two key components: a physics code and a set of machine learning models which emulate the physics code.
We perform active learning by using the machine learning emulators to quickly estimate the outcome of the physics code
for a large number of possible inputs,
and use the results of the inference to determine which inputs to run with the physics code.
We then perform those runs of the physics code, use the new outcomes to improve the surrogates, and then start from the
inference step again.
In short, active learning applications involve inference, simulation, and training tasks that are performed
sequentially.

We parallelize the active learning loop described above by running multiple instances of the same task (e.g., running
the physics code) in parallel,
and running each type of task concurrently.
There are some dependencies between the types of task, so they cannot be run completely asynchronously with each other.
Instead, we use the following strategy for running the different types of tasks together

1. Write the highest-priority tasks from inference results to a buffer
    1. _Optional_: Stop refilling the buffer when models are being retrained
2. Start next simulation in buffer as soon as resources are available
3. Begin retraining all models as soon as enough simulations complete
4. Launch new inference tasks as soon as the model completes training

The amount of resources used is controlled through a few separate mechanisms.
The number of parallel resources available for each type of task can be varied independently.
One can also change the policies relating to the concurrency of different tasks.
Decreasing the size of the buffer and not refilling it while models are retraining conserves resources used for
simulation,
whereas increasing the amount of data required to retraining the models conserves machine learning resources.

Each of the policies described above are best stated as events, and we use a set of 6
event-triggered [Colmena](https://colmena.readthedocs.io/) "agents" to implement them.
For example, we have a "simulation launcher" worker which adds tasks to the buffer when a simulation completes.
There is a corresponding "simulation receiver" that marks when a simulation has finished and decides whether to trigger
a model retraining.
The full relationship between these 6 workers and the shared resources between them are illustrated below.

## Running the Code

The `run_*.sh` shell files provided with this repository define a standard problem configuration. 
Each script invokes campaign driver, `run.py` with a different series of command line arguments that control how the campaign is run.

We run a specific set of machine learning models trained using a predefined set of molecules and evaluated over the same set of molecules.
You should not need to modify these if you are studying the workflow system, but there is documentation in `run.py` for what each option does.

The primary seetings you will like modify are the ProxyStore configuration, which is in the "ProxyStore" section of options.
For instance, you can switch between the backend or the threshold sizes at which objects or proxied or turn off ProxyStore altogether.

The code will create a new run directory in `runs` folder and populate it with both temporary files (e.g., copies of objects being proxied) and 
traces of the workflow performances.

Call `python run.py --help` for full details

## ProxyStore

ProxyStore can be enabled on a per-topic basis (e.g., for the simulate, infer, and train tasks) using the provided
command line arguments.
See the ProxyStore argument group with `run.py --help`.

### Example ProxyStore Usage

In this example, `run.py` is executed on a Theta login node, a Redis server is running on a `thetamom1`, simulations are
done on a Theta endpoint `THETA_ENDPOINT`, and inference and training tasks are done on a ThetaGPU
endpoint `THETAGPU_ENDPOINT`.

```
$ run.py \
      --redishost thetamom1 \
      --redisport $REDIS_PORT \
      --qc-endpoint $THETA_ENDPOINT \
      --ml-endpoint $THETAGPU_ENDPOINT \
      --simulate-ps-backend redis \
      --infer-ps-backend file \
      --train-ps-backend file \
      --ps-threshold 500000 \
      --ps-file-dir $PROJECT_DIR/scratch/proxystore-dump
```

With the above configuration, with `simulate` tasks, ProxyStore with Redis will be used (will default to use the same
Redis server that the Task server uses).
When using the Redis ProxyStore backend, the Redis server must be reachable from the Colmena client and workers on the
FuncX endpoint.
This is why we place the Redis server on a Theta MOM node in this example.

For the `infer` and `train` tasks, we use a file system backend with ProxyStore.
This is because workers on the ThetaGPU FuncX endpoint cannot access our Redis server running on the Theta MOM node but
can access the Theta file system.
The `--ps-file-dir` argument specifies a directory that ProxyStore can use for storing serialized objects.

For all ProxyStore backends, only objects greater than 500KB will be proxied (as specified by
the `--ps-threshold 500000` argument).

`globus` is a third ProxyStore backend that is supported in addition to `redis` and `file`.
When using the `globus` backend option, a ProxyStore Globus config file must also be specified via `--ps-globus-config`.
An example is provided in `globus_config.json`.
