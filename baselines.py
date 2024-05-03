import argparse
import os
from baselines import mip, pdrs, cp_sat
from pandas import DataFrame


#
DEBUG = True


def test_mip(instances: list):
    for ins in instances:
        f_costs = ins['costs'].view(-1).numpy()

        #
        pdr = pdrs.PDR(pdrs.MWR())
        sol, times = pdr(ins['j'], ins['m'], ins['costs'].numpy(),
                         ins['machines'].numpy())

        # DISJUNCTIVE MODEL
        model, cmax, x = mip.make_model(ins)

        # Init solver with MWR solution
        op = 0
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
        results = {'NAME': ins['name'],
                   'MS': ms,
                   'OPT': int(model.status),
                   'UB': model.ObjBound,
                   'TIME': model.RunTime,
                   'GAP': (ms / ins['makespan'] - 1) * 100}
        if DEBUG:
            print(f"Instance {ins['name']}")
            print(f"\t- MS={cmax.X:.2f}")
            print(f"\t- GAP={(cmax.X / ins['makespan'] - 1) * 100:.3f}")
        #
        DataFrame([results]).to_csv(
            f'output/MIP_{args.benchmark}.csv',
            mode='a',
            sep=',',
            index=False,
            header=False
        )


def test_cp(instances: list):
    for ins in instances:
        costs = ins['costs'].numpy()
        machines = ins['machines'].numpy()

        #
        pdr = pdrs.PDR(pdrs.MWR())
        sol, times = pdr(ins['j'], ins['m'], costs, machines)
        pdr_ms = max(t[-1] for t in times)

        # DISJUNCTIVE MODEL
        solver, model, tasks = cp_sat.make_model(machines, costs,
                                                 horizon=int(pdr_ms))

        # Init solver with solution
        for i, job_ct in enumerate(times):
            for j, op_ct in enumerate(job_ct):
                st = op_ct - costs[i, j]
                assert st >= 0, "Negative start time!"
                model.AddHint(tasks[i, j].start, int(op_ct))
        #
        if DEBUG:
            print(f'--> INIT ms={max(ct[-1] for ct in times):.3f}')

        # Model and set params
        solver.parameters.log_search_progress = DEBUG
        solver.parameters.max_time_in_seconds = args.limit
        solver.Solve(model)

        #
        ms = solver.ObjectiveValue()
        results = {'NAME': ins['name'],
                   'MS': ms,
                   'LB': solver.BestObjectiveBound(),
                   'TIME': solver.WallTime(),
                   'GAP': (ms / ins['makespan'] - 1) * 100}
        if DEBUG:
            print(f"Instance {ins['name']}")
            print(f"\t- MS={ms:.2f}")
            print(f"\t- GAP={(ms / ins['makespan'] - 1) * 100:.3f}")
        #
        DataFrame([results]).to_csv(
            f'output/CP_{args.benchmark}.csv',
            mode='a',
            sep=',',
            index=False,
            header=False
        )


def test_pdrs(instances: list):
    algs = [pdrs.PDR(pdrs.SPT()), pdrs.PDR(pdrs.MWR()), pdrs.PDR(pdrs.MOR())]
    results = []
    for ins in instances:
        res = {}
        for pdr in algs:
            pdr_ms, pdr_time = pdrs.solve_instance(ins, pdr,
                                                   args.beta, args.seed)
            res[f'{pdr.name} MS'] = pdr_ms
            res[f'{pdr.name} TIME'] = pdr_time
        #
        print(ins['name'], res)
        results.append(res)

    # Save results
    DataFrame(
        results,
        index=[i['name'] for i in instances]
    ).sort_index().to_csv(
        f"output/pdrs_{args.benchmark}_B{args.beta}.csv",
        sep=','
    )


#
parser = argparse.ArgumentParser(description='Test baseline algorithms')
parser.add_argument("-benchmark", type=str, required=False,
                    default='TA', help="Either TA or DMU.")
parser.add_argument("-limit", type=int, default=3600,
                    required=False, help="Time limit in seconds.")
parser.add_argument("-alg", type=str, default='mip',
                    required=False, help="Baseline algorithm to run.")
parser.add_argument("-beta", type=int, default=1, required=False,
                    help="Number of sampled solutions for pdrs.")
parser.add_argument("-seed", type=int, default=12345,
                    required=False, help="Random seed.")
args = parser.parse_args()


if __name__ == '__main__':
    from inout import load_raw

    # Load raw instances
    instances = load_raw(f'benchmarks/{args.benchmark}')
    if not os.path.exists('../output/'):
        os.makedirs('../output/')

    #
    if 'mip' in args.alg.lower():
        test_mip(instances)
    elif 'cp' in args.alg.lower():
        test_cp(instances)
    elif 'pdr' in args.alg.lower():
        test_pdrs(instances)
    else:
        raise RuntimeError(f'Unknown baseline {args.alg}')
