"""
Priority Dispatching Rules:
 - SPT: Shortest Processing Time
 - MWR: Most Work Remaining
 - MOR: Most Operations Remaining
"""
import numpy as np
import os
from time import time
import argparse
import random


class Priority(object):
    """ Base priority class """
    name = 'Priority'       # Name of the priority
    tau = 0.0001            # Step for tie breaking

    def __init__(self):
        self._tie = 0.

    def __call__(self, j, idx, costs, **kwargs):
        raise NotImplementedError()


class SPT(Priority):
    """ Shortest Processing Time """
    name = 'SPT'

    def __call__(self, j, idx, costs, **kwargs):
        self._tie += self.tau       # Tie breaking
        return -costs[j, idx] - self._tie


class MWR(Priority):
    """ Most WorK Remaining """
    name = 'MWR'

    def __call__(self, j, idx, costs, **kwargs):
        self._tie += self.tau       # Tie breaking
        return sum(costs[j, idx:]) + self._tie


class MOR(Priority):
    """ Most Operation Remaining """
    name = 'MOR'

    def __call__(self, j, idx, costs, **kwargs):
        self._tie += self.tau       # Tie breaking
        return len(costs[j, idx:]) + self._tie


class PDR(object):
    """
    Main class for creating Priority Dispatching Rules with time
    index advancing.

    Args:
        priority: Callable to get priority for operations.
        top: Number of operations from which is arbitrary picked the next
            operation to schedule when randomized.
        eps: Tolerance for handling the comparison of float.
    """
    def __init__(self, priority: Priority, top: int = 3,
                 eps: float = 0.001):
        self._eps = eps
        self._active = None
        self._mac_time = None
        self.priority = priority
        self.name = priority.name
        self.top = top

    def __call__(self,
                 num_j: int,
                 num_m: int,
                 costs: np.ndarray,
                 machines: np.ndarray,
                 randomized: bool = False,
                 **kwargs):
        """
        Run the dispatching rule.

        Args:
            num_j: Number of jobs in the instance
            num_m: Number of machines in the instance
            costs: For each operation (job index, operation index), it gives the
                processing time of the operation.
            machines: For each operation (job index, operation index), it gives
                the machine on which the operation must be executed.
        Returns:
            solution: The order of operations on each MACHINE
            times: The completion times of operations for each JOB
        """
        #
        ops = np.array([(0, machines[j, 0]) for j in range(num_j)])
        prio = np.array([self.priority(j, 0, costs, **kwargs)
                         for j in range(num_j)])

        #
        curr_time = -1
        mac_time = [0 for _ in range(num_m)]
        sol = [[] for _ in range(num_m)]
        times = [[] for _ in range(num_j)]

        #
        active_job = num_j
        while active_job > 0:
            # Sort active jobs
            job_order = np.argsort(prio)[::-1][:active_job]
            if randomized:
                # Pick random among first top-k remaining
                rand_order, k = [], self.top
                top_k = list(job_order[:self.top])
                while top_k:
                    item = top_k.pop(random.randint(0, len(top_k) - 1))
                    rand_order.append(item)
                    if k < active_job:
                        top_k.append(job_order[k])
                        k += 1
                job_order = rand_order

                # # Make probabilities and sort base on them (worse than above)
                # p = prio[job_order]
                # if p[-1] < 0:
                #     p -= p[-1] - 1
                # p = p / p.sum()
                # job_order = np.random.choice(job_order, size=active_job,
                #                              replace=False, p=p)

            # Use the time while dispatching
            curr_time = min(ct for ct in mac_time if ct > curr_time)

            #
            for j in job_order:
                idx, m = ops[j]

                # Minimum start time of the operation
                min_st = max(mac_time[m], 0 if idx == 0 else times[j][-1])

                # Schedule if the min start time is lower than the current time
                if min_st - self._eps < curr_time:
                    if idx < num_m - 1:
                        # Insert the next operation of the job
                        ops[j, 0] = idx + 1
                        ops[j, 1] = machines[j, idx + 1]
                        prio[j] = self.priority(j, idx + 1, costs)
                    else:
                        # Deactivate job
                        active_job -= 1
                        prio[j] = -float('inf')

                    # Schedule operation
                    mac_time[m] = min_st + costs[j, idx]
                    times[j].append(mac_time[m])
                    sol[m].append(j * num_m + idx)

        return sol, times


def solve_instance(ins: dict, pdr: PDR, beta: int = 1):
    """
    Helper for testing dispatching rules on an instance.

    Args:
        ins: JSP instance.
        pdr: Dispatching rule to use.
        beta: Number of randomized solutions to generate. If beta = 1, only a
            single greedy solution is constructed.
    Returns:
        - The makespan of the best solution.
        - The completion time of operations.
    """
    costs = ins['costs'].numpy()
    machines = ins['machines'].numpy()
    st = time()

    # Always apply deterministic
    sol, times = pdr(ins['j'], ins['m'], costs, machines)
    best_ms = max(t[-1] for t in times)
    # Generate randomized solution when beta > 1
    if beta > 1:
        np.random.seed(args.seed)
        random.seed(args.seed)
        #
        for _ in range(beta - 1):
            sol, times = pdr(ins['j'], ins['m'], costs, machines,
                             randomized=True)
            ms = max(t[-1] for t in times)
            if ms < best_ms:
                best_ms = ms
    et = time() - st
    return best_ms, et


#
parser = argparse.ArgumentParser(description='Test PDR')
parser.add_argument("-benchmark", type=str, required=False,
                    default='TA', help="Either TA or DMU.")
parser.add_argument("-beta", type=int, default=128, required=False,
                    help="Number of sampled solutions.")
parser.add_argument("-seed", type=int, default=12345,
                    required=False, help="Random seed.")
args = parser.parse_args()


if __name__ == '__main__':
    import pandas as pd
    from inout import load_dataset
    instances = load_dataset(f"benchmarks/{args.benchmark}")

    #
    pdrs = [
        PDR(SPT()),
        PDR(MWR()), PDR(MOR())]
    results = []
    for ins in instances:
        res = {}
        for pdr in pdrs:
            pdr_ms, pdr_time = solve_instance(ins, pdr, args.beta)
            res[f'{pdr.name} MS'] = pdr_ms
            res[f'{pdr.name} TIME'] = pdr_time
        #
        print(ins['name'], res)
        results.append(res)

    # Save results
    if not os.path.exists('./output/'):
        os.makedirs('./output/')
    df = pd.DataFrame(results,
                      index=[i['name'] for i in instances]).sort_index()
    df.to_csv(f"output/pdrs_{args.benchmark}_B{args.beta}.csv",
              sep=',')
