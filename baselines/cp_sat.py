import numpy as np
from ortools.sat.python import cp_model
from collections import namedtuple

# Multiplicative factor for scaling processing time
FACTOR = 1
#
Task = namedtuple('Task', 'start end interval')


def make_model(machines: np.ndarray,
               p: np.ndarray,
               horizon: int = None):
    """
    Solves the job shop.
    """
    # Creates the solver.
    model = cp_model.CpModel()
    num_m = len(machines[0])
    num_j = len(p)

    # Computes horizon dynamically.
    if horizon is None:
        horizon = sum([sum(p[i] * FACTOR) for i in range(num_j)])

    # Creates jobs.
    tasks = {}
    for i in range(num_j):
        for j in range(num_m):
            start_var = model.NewIntVar(0, horizon, f'st_{i}_{j}')
            end_var = model.NewIntVar(0, horizon, f'et_{i}_{j}')
            int_var = model.NewIntervalVar(start_var, int(p[i, j] * FACTOR),
                                           end_var, f'inter_{i}_{j}')
            #
            tasks[(i, j)] = Task(start=start_var,  end=end_var, interval=int_var)

    # Create disjunctive constraints.
    for m in range(num_m):
        model.AddNoOverlap(tasks[(i, j)].interval
                           for i, j in np.argwhere(machines == m).squeeze())

    # Precedences inside a job.
    for i in range(num_j):
        for j in range(0, num_m - 1):
            model.Add(tasks[(i, j + 1)].start >= tasks[(i, j)].end)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(
        obj_var, [tasks[(i, num_m - 1)].end for i in range(num_j)])
    model.Minimize(obj_var)

    return cp_model.CpSolver(), model, tasks
