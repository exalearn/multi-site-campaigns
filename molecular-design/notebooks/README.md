# Notebooks for Analyzing Runs

This folder contains all the analyses reported in our paper, and some we did performed but did not write about.

## How to repeat results

The notebooks expect that the runs have been performed and there are a few folders containing links to special runs:

- `prod_runs`: Runs with default settings used to compare performance of workflow systems

The names of the specific runs and associated are provided in the published form of this repository (TBD).

The first task to running the notebooks is to execute [the "post-processing" notebook](./0_post-process-runs.ipynb), which performs some basic analysis of output for each simulation.

The other notebooks perform different analyses described in the paper, and produce figures stored in the [figures folder](./figures)
