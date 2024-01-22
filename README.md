# LabelingJSP

Code for IJCAI 2024: Self-Labeling the Job Shop Scheduling Problem (JSP)

> We will provide additional configuration and setup information.

## Project Structure

The project entrypoints are:
- pdrs.py: is the entrypoint for testing Priority Dispatching Rules, both greedy and randomized ones.
- gurobi.py: is the entrypoint for running the Mixed Integer Programming formulation that solves the JSP.
- test.py: is the file for testing the trained Self-labeling Pointer Networks (SPN).
- testL2D.py: contains the code for generating the randomized results of `"Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning"` (L2D).
- train.py: is the entrypoint for training generative model via the proposed self-labeling training strategy.

All the other files contain helper functions and utilities.

The JSP instances are divided into two folders:
- dataset5k: contains the instances used for training models. We also include for each instance the best solution ever found.
- benchmarks: contains the Taillard's benchmark (TA), Demirkol's benchmark (DMU), and the validation instances. 

## Requirements

- PyTorch 13.1
- PyTorch Geometric 2.2
- Tdqm
- Gurobipy
- Pandas
- Gym (necessary for running L2D) 
