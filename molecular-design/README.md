# Multi-Site Active Learning for IP Optimization

Re-training machine learning models based on new simulations is major bottleneck in our [previous molecular design application](https://doi.org/10.1109/MLHPC54614.2021.00007)
and one we can reduce by performing training on specialized hardware.
Here, we illustrate how to implement an application using Colmena that distributes each type of task to a different resource.

## Running the Code

The `run_*.sh` shell files provided with this repository define a standard problem configuration. 
Each script invokes campaign driver, `run.py` with a different series of command line arguments that control how the campaign is run.
Call `python run.py --help` for full details.

We run a specific set of machine learning models trained using a predefined set of molecules and evaluated over the same set of molecules.
You should not need to modify these if you are studying the workflow system, but there is documentation in `run.py` for what each option does.

The primary seetings you will like modify are the ProxyStore configuration, which is in the "ProxyStore" section of options.
For instance, you can switch between the backend or the threshold sizes at which objects or proxied or turn off ProxyStore altogether.

The code will create a new run directory in `runs` folder and populate it with both temporary files (e.g., copies of objects being proxied) and 
traces of the workflow performances.

## Preparing for a Multi-site Run

There are two types of multi-site runs, each requiring differnet configuration paths.

### FuncX

You must [create a FuncX endpoint](https://funcx.readthedocs.io/en/latest/endpoints.html) on the systems used for both the quantum chemistry and 
machine learning tasks. 
Once those are started, you will be given a UUID that is provided as input `run.py`.

We provide the FuncX configurations used in this study in [`funcx-configs`](./funcx-configs)

FuncX experiments can be run from any node with filesystem access to Theta, although that could be changed by using Globus for the ProxyStore backend of simulation tasks.


### Parsl

The Parsl configuration used in this study is defined in [`config.py`](./config.py).
It includes two types of executors: one that requistions nodes from Theta via a job scheduler, 
and a second that connects to a remote GPU machine over an SSH tunnel.

The Theta configuration adheres closely to the [Theta configuration provided in the Parsl documentation](https://parsl.readthedocs.io/en/1.2.0/userguide/configuring.html#theta-alcf).

The GPU node configuration is the less standard configuration.
We specify the ports to communicate with Parsl so that they match up with those of the tunnel (see below).
We also specify "localhost" as the address for the ports so that the workers connect to the tunnel.
The configuration also includes hard-coded work and log directories to paths that exist on the remote system (Parsl does not autodetect where I have write permissions), 
and also uses SSH without password because I have set up SSH keys beforehand (though Parsl [can handle clusters with passwords/2FA](https://parsl.readthedocs.io/en/1.2.0/stubs/parsl.channels.SSHInteractiveLoginChannel.html#parsl.channels.SSHInteractiveLoginChannel))

You must open an SSH tunnel to the GPU machine before running these experiments.
We used the following command.

```
ssh -N -R 6379:localhost:6379 -R 54928:localhost:54928 -R 54875:localhost:54875 lambda1.cels.anl.gov
```

The 6379 port is for the Redis server used by ProxyStore and the other two are Parsl.

Parsl experiments must be run from a Theta login node, as we do not also configure SSH tunnels to Theta.

## ProxyStore

ProxyStore can be enabled on a per-topic basis (e.g., for the simulate, infer, and train tasks) using the provided command line arguments.
See the ProxyStore argument group with `run.py --help`.

### Example ProxyStore Usage

In this example, `run.py` is executed on a Theta login node, a Redis server is running on a `thetamom1`, simulations are done on a Theta endpoint `THETA_ENDPOINT`, and inference and training tasks are done on a ThetaGPU endpoint `THETAGPU_ENDPOINT`.

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

With the above configuration, with `simulate` tasks, ProxyStore with Redis will be used (will default to use the same Redis server that the Task server uses).
When using the Redis ProxyStore backend, the Redis server must be reachable from the Colmena client and workers on the FuncX endpoint.
This is why we place the Redis server on a Theta MOM node in this example.

For the `infer` and `train` tasks, we use a file system backend with ProxyStore.
This is because workers on the ThetaGPU FuncX endpoint cannot access our Redis server running on the Theta MOM node but can access the Theta file system.
The `--ps-file-dir` argument specifies a directory that ProxyStore can use for storing serialized objects.

For all ProxyStore backends, only objects greater than 500KB will be proxied (as specified by the `--ps-threshold 500000` argument).

`globus` is a third ProxyStore backend that is supported in addition to `redis` and `file`.
When using the `globus` backend option, a ProxyStore Globus config file must also be specified via `--ps-globus-config`.
An example is provided in `globus_config.json`.

