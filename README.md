# Steering Computaitonal Campaigns across Multiple, Hetereogenous Systems

This repository contains the software used to evaluate the performance of different
systems for executing a computational campaigns across multiple computing sites with different hardware.
As [better described in the associated paper](#), the need to deploy scientific workflows across multiple
sites is driven by the increasing diversity in application requirements 
and the available hardware on which to execute them.
We experiment a few different aspects of multi-site campaigns:

1. _Workflow engine_ that delegates tasks across distributed resources
1. _Steering policies_ that control how tasks are executed across different resources
1. _Data transportation system_ that move large inputs or results between compute reosurces

## Organization

Our study includes two major components

### Example application: Molecular design

TBD

### Detailed Analysis of Data Transport Layer

TBD

## Installation

We provide different environment for different workflow engines.

- _FuncX_: `environment-funcx.yml`
- _Parsl_: `environment-parsl.yml`

Install them using Anaconda

```bash
conda env create --file <path to .yml> --force
```

