# First Attempt

The goal is to rapidly train a machine learning model on relevant regions of a potential energy surface.
We do so by efficiently sampling structures by running molecular dynamics using the trained models
then computing the energy of structures where the model is least certain.

This fitting strategy interleaves four different operations:

- _Training_ an ensemble of models
- _Sampling_ new structures using the molecular dynamics
- _Selecting_ structures 
- _Calculating_ the energy and forces for selected structures

## Running the Code

The application is driven by a main script, [`run.py`](./run.py), that dispatches tasks to FuncX endpoints.

You will need to install FuncX endpoints on the target system in the fff python environment before running this.

Once you do, call: `python run.py -h` to get a list of full options.

## Steering Strategy

The steering strategy is built to simultaneously achive two objectives: training a forcefield, and sampling structures.
We implement the steering strategy using 4 "agents" that coordinate together in managing two loosely-coupled loops.

![coordination-strategy](./figures/thinker-diagram.svg)

The "sampling loop" (inner, green) includes a **Sampler** agents which submits molecular dynamics calculations and then submit the final structure to be audited by a **Calculator** agent before the trajectory is continued. 
The "training loop" beings with **Selector** agents that finds a diverse pool of structures produced by the **Sampler** that are used to retrain a new model libary by **Trainer** agents.

### Agents

Details of the agents.

#### Trainer

The training part creates an ensemble of models periodically during a run.

Training is built using paired agents:

- _Submitter_ launches a many training task from the starting weights each starting 
  with a different bootstrapped sample of the available training data.
- _Storer_ receives the training results and stores the completed models on disk. 
  Once a first model is complete, it allows the sampler to start. 
  Once all complete, it allows the inference tasks to complete

The training agents start at the beginning of the run and when sufficient data have been acquired.

#### Sampler

The sampling operation produces a stream of new atomic configurations to sample.

The sampling calculations are based on molecular dynamics. 
We run molecular dynamics for a moderate number of timestep before halting to audit.
We audit the trajectory by computing the energy of the structure with our physics code.
If the energy between the ML model is close enough to the calculation,
we will restart the trajectory from the last structure.
If not, we repeat from the starting structure.
In this way, we progressively sample regions of the PES and force it to stay in regions that 
are energetically feasible.

Sampling is built using paired agents:

- _Submitting_: Picks a random trajectory to start or continue then submits it to run.
- _Receiving_: Adds an auditing calculation to the list to evaluate 
   and starts inference on the structures produced during sampling.
   
We maintain a constant number of parallel sampling experiments, starting a new one as soon as another completes.

> Should we change MD to the basin-paving? 

#### Selector

Active learning calculations start by running inference for each sampled structure 
from each of the structures produced during training.
We then select a group of structures where there is a large variance between models
and minimal similarity between each other.

We use a heuristic strategy (i.e., one Logan hacked together without thought):

1. Pick the structure with the largest variance
2. Remove the 5% of structures where the energy predictions have 
3. If there are fewer than 5 structures left, halt
4. Repeat from step 1 until you have a target number of structures or fewer than 5 structures are left to choose from. 

We run inference tasks as soon as enough structures are sampled to make a batch large enough to mitigate communication costs.
Active learning starts when enough inference tasks have completed.

### Calculator

The calculation tasks are simple: run the physics code to get energies.

Each calculation starts by first picking a structure the "audit" or "active learning" list randomly.
We launch that calculation to execute remotely and then store the result in an ASE db when complete.


## Implementation Details

There are a few implementation tricks that are worth noting to study if they actually make snese.

### Use of Proxies

The machine learning models are always proxied:

- _Starting Model_: We create a proxy for the initial model at the beginning and never evict it.
- _Sampling Model_: The first model to finish training is proxied for use in the sampling calculations.
  The model is evicted each time a new batch is trained. 
- _Inference Model_: Each model is proxied as soon as it finishes training and is re-used across multiple inference tasks.
  We evict a model from the previous batch each time a new one finishes training.

At present, we do not explicitly proxy any datasets (but probably should) 

### Scheduling

TBD. While write this out once I add in some more options.

## Understanding the Output

TBD
