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

- _NVIDIA GPUs_: Change the `tensorflow-cpu` to `tensorflow`.
