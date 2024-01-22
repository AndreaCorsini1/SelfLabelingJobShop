import argparse
import numpy as np
import gurobipy as gp
import os
from gurobipy import GRB

#
DEBUG = False
# Multiplicative factor for scaling processing time
FACTOR = 1


def prepare(instance: dict):
    """
    Prepare a  MIP execution.

    Args:
        instance: Instance to prepare for disjunctive MILP model.
    Returns:
        jobs: 2D matrix where each row gives the precedence among operations of
            a job. The values in this matrix are the absolute indices of
            operations in the instance. Shape (num_j, num_m).
        machines: Machines of operations. Rows correspond to jobs.
            Shape (num_j, num_m).
        p: Processing time of operations. Rows correspond to jobs.
            Shape (num_j, num_m).
    """
    num_j, num_m = instance['j'], instance['m']

    #
    jobs = np.arange(num_j * num_m).reshape(num_j, num_m)

    #
    f_machines = instance['machines'].numpy().reshape(-1)
    machines = -np.ones((num_m, num_j), dtype=int)
    for m in range(num_m):
        machines[m] = np.argwhere(f_machines == m).squeeze()

    #
    p = (instance['costs'].numpy().reshape(-1) * FACTOR).astype(int)

    return jobs, machines, p


def make_model(instance: dict):
    """
    Make Disjunctive model for the Job Shop.
    ---> NOTE: AN ACADEMIC/ENTERPRISE LICENSE IS REQUIRED TO RUN THIS MODEL ON
            LARGE BENCHMARK INSTANCES.

    Args:
        instance: jsp instance to build a disjunctive MIP
    Return:
        - The MIP model.
        - The makespan variable.
        - The start time variable.
    """
    #
    jobs, machines, p = prepare(instance)

    ### PARAMETERS
    num_j, num_m = jobs.shape
    num_ops = num_j * num_m
    J = range(num_j)
    M = range(num_m)
    V = p.sum()     # Big M as the sum of all the processing times

    # Sort for breaking symmetry
    machines.sort(-1)
    disjunctive = [(machines[m, i], machines[m, k])
                   for m in M for i in J for k in range(i + 1, num_j)]

    ### DECISION VARIABLES
    model = gp.Model("DisjunctiveModel")

    # Auxiliary for makespan
    Cmax = model.addVar(lb=0, name="Cmax")
    # Start time of each operation
    x = model.addVars(num_ops, vtype=GRB.INTEGER, lb=0, name="start")
    # Order of jobs on a machine
    z = model.addVars(disjunctive, vtype=GRB.BINARY, name="order")

    ### CONSTRAINTS
    # 1. Precedence constraint on each job
    model.addConstrs((x[k] - x[i] >= p[i]
                      for j in J for i, k in zip(jobs[j, :-1], jobs[j, 1:])),
                     name="PrecedenceJob")

    # 2. Disjunctive constraint 1
    model.addConstrs((x[i] >= x[k] + p[k] - V * z[i, k]
                      for i, k in disjunctive), name="Disjunctive1")

    # 3. Disjunctive constraint 2
    model.addConstrs((x[k] >= x[i] + p[i] - V * (1 - z[i, k])
                      for i, k in disjunctive), name="Disjunctive2")

    # 4. Makespan
    model.addConstrs((Cmax >= x[op] + p[op]
                      for op in range(num_m - 1, num_ops, num_m)),
                     name="Makespan")

    ### OBJECTIVE
    model.setObjective(Cmax, GRB.MINIMIZE)
    return model, Cmax, x


#
parser = argparse.ArgumentParser(description='Test PDR')
parser.add_argument("-benchmark", type=str, required=False,
                    default='TA', help="Either TA or DMU.")
parser.add_argument("-limit", type=int, default=3600,
                    required=False, help="Time limit in seconds.")
args = parser.parse_args()


if __name__ == '__main__':
    from pandas import DataFrame
    from inout import load_dataset
    from pdrs import PDR, MWR

    #
    pdr = PDR(MWR())
    instances = load_dataset(f'benchmarks/{args.benchmark}')
    for ins in instances:

        # DISJUNCTIVE MODEL
        model, cmax, x = make_model(ins)

        # Init solver with SPT solution
        op = 0
        f_costs = ins['costs'].view(-1).numpy()
        sol, times = pdr(ins['j'], ins['m'], ins['costs'].numpy(),
                         ins['machines'].numpy())
        for job_cts in times:
            for op_ct in job_cts:
                st = op_ct - f_costs[op]
                assert st >= 0, "Negative start time!"
                x[op].Start = st
                op += 1
        if DEBUG:
            print(f'--> INIT ms={max(ct[-1] for ct in times):.3f}')

        # Optimize the model
        model.setParam('TimeLimit', args.limit)
        if not DEBUG:
            model.setParam('OutputFlag', 0)
        model.optimize()

        #
        ms = cmax.X
        results = {
            'NAME': ins['name'],
            'MS': ms,
            'OPT': int(model.status == GRB.OPTIMAL),
            'UB': model.ObjBound,
            'TIME': model.RunTime,
            'GAP': (ms / ins['makespan'] - 1) * 100
        }
        if DEBUG:
            print(f"Instance {ins['name']}")
            print(f"\t- MS={cmax.X:.2f}")
            print(f"\t- GAP={(cmax.X / (ins['makespan'] * FACTOR) - 1) * 100:.3f}")
        #
        if not os.path.exists('./output/'):
            os.makedirs('./output/')
        DataFrame([results]).to_csv(
            f'output/MIP_{args.benchmark}.csv',
            mode='a',
            sep=',',
            index=False,
            header=False
        )
