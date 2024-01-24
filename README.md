# LabelingJSP

Self-Labeling the Job Shop Scheduling Problem.

We propose a Self-Supervised training strategy specifically designed for combinatorial problems.
Inspired by Semi- and Self-Supervised learning, we show that it is possible to easily train generative models by sampling multiple solutions and using the best one according to the problem objective as a pseudo-label.
In this way, we iteratively improve the model generation capability by relying only on its self-supervision, completely removing the need for optimality information.

We prove the effectiveness of this Self-Labeling strategy on the Job Shop Scheduling (JSP), a complex combinatorial problem that is receiving much attention from the Reinforcement Learning community.
We propose a generative model based on the well-known Pointer Network and train it with our strategy.
Experiments on two popular benchmarks demonstrate the potential of this approach as the resulting models outperform constructive heuristics and current state-of-the-art Reinforcement Learning proposals.

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
