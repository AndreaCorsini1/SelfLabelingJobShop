# Self-Labeling the Job Shop Scheduling Problem

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


## Dataset and benchmark instances

All the instances used for training and testing follow the same structure.
Here is a small example:

```
3 2             # Instance shape (num. jobs and num. machines)
0 4 1 6         # First job
1 9 0 3         # Second job
0 4 1 6         # Third job
39              # Instance upper bound 
0 4 3           # Sequence of operations on machine 0
2 1 5           # Sequence of operations on machine 1
```

The first line gives the number of jobs and machines in the instance. 
In this example, the instance has 3 jobs and 2 machines. 

Then, follow information about the jobs in the instance. Each job is 
given as a sequence of pairs (machine index, processing time). 
For example, the first job starts executing on machine 0 for 4 time units,
and afterward it goes on machine 1 for 6 time units. The second and third jobs 
follow the same structure.

After the instance, there is an upper bound (UB) on the optimal solution of 
the instance. For benchmark instances, we used the best-known UB in the 
literature, while for dataset instances (dataset5k and benchmarks/validation 
folders) we used the best UB ever found by our model during the trainings.
After the UB line, it follows the solution producing that UB value, where 
rows correspond to machines.


## Requirements

- PyTorch 13.1
- PyTorch Geometric 2.2 (check the PyG site for installation instructions)
- Tdqm
- Gurobipy
- Pandas
- Gym (necessary for running L2D) 

It should also work with newer versions of PyTorch and PyTorch Geometric.  


## Cite as:
```
@inproceedings{SelfJSP,
    title = {Self-Labeling the Job Shop Scheduling Problem},
    author = {Corsini, Andrea and Porrello, Angelo and Calderara, Simone and Dell'Amico, Mauro},
    year={2024},
    publisher={Arxiv},
}
```