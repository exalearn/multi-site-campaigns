# Steering Computaitonal Campaigns across Multiple, Hetereogenous Systems

This repository contains the software used to evaluate the performance of different
systems for executing a computational campaigns across multiple computing sites with different hardware.
As [better described in the associated paper](#), the need to deploy scientific workflows across multiple
sites is driven by the increasing diversity in application requirements
and the available hardware on which to execute them.
We experiment a few different aspects of multi-site campaigns:

1. _Workflow engine_ that delegates tasks across distributed resources
1. _Steering policies_ that control how tasks are executed across different resources
1. _Data transportation system_ that move large inputs or results between compute resources

The copy of this repository as of our HCW'23 paper and all data used in that study are 
[available on the Materials Data Facility.](https://acdc.alcf.anl.gov/mdf/detail/multiresource_ai_v2.1/)


## Organization

Our study includes two major components

### Example Application: Molecular design

Our main example application is a machine-learning guided search for molecules with high resistance to being ionized.
The machine learning tasks (training, inference) in this application are best run on GPU and the software for computing
ionization is best on CPU.
The application also has significant performance vs efficiency tradeoff in how frequently the machine learning tasks are
performed.

Full details and software are in [`molecular-design`](./molecular-design).

The `moldesign` directory contains many utilities for the molecular design workflow (e.g., Python wrappers for
simulation tools, neural network layers).
See [another repo](http://github.com/exalearn/electrolyte-design) for a more up-to-date version.

## Installation

Our computational environment is described using Conda.

```commandline
conda env create --file environment.yml
```

You may need to modify the environments to install versions of Tensorflow optimized for your hardware.
By default, we install the `tensorflow-cpu` as we do not assume CUDA is available on a system.

- _NVIDIA GPUs_: Change the `tensorflow-cpu` to `tensorflow`. Change the paths to the package indicies for PyTorch and PyG to those for your CUDA version.

See the [`envs`](./envs) directory for examples.

## Preparing for a Multi-site Run

There are two types of multi-site runs, each requiring differnet configuration paths.

### FuncX

You must [create a FuncX endpoint](https://funcx.readthedocs.io/en/latest/endpoints.html) on the each resource being used.
Once those are started, you will be given a UUID that is provided as input `run.py`.

We provide the FuncX configurations used in this study in [`funcx-configs`](./configs/funcx-configs)

### Parsl

The Parsl configuration used in this study is defined in [`config.py`](./configs/config.py).
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
