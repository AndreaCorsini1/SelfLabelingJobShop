import argparse
import torch
import pandas as pd
import os
from tqdm import tqdm
from sampling import sampling, greedy
from inout import load_dataset
from time import time

# Training device
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def validation(encoder: torch.nn.Module,
               decoder: torch.nn.Module,
               val_set,
               beta: int = 32,
               seed: int = None,
               name: str = 'Validation'):
    """

    Args:
        encoder: Encoder.
        decoder: Decoder.
        val_set: Validation set.
        beta: Number of solution to generate for each instance.
        seed: Random seed.
        name: Validation set name.
    """
    if seed is not None:
        torch.manual_seed(seed)
    encoder.eval()
    decoder.eval()
    results, gaps, shape_gap = [], [], {}
    for ins in tqdm(val_set):
        st = time()
        if beta > 1:
            s, mss = sampling(ins, encoder, decoder, bs=beta, device=dev)
        else:
            s, mss = greedy(ins, encoder, decoder, device=dev)
        exe_time = time() - st
        #
        _gaps = (mss / ins['makespan'] - 1) * 100
        min_gap = _gaps.min().item()
        results.append({'NAME': ins['name'],
                        'UB': ins['makespan'],
                        'MS': mss.min().item(),
                        'MS-AVG': mss.mean().item(),
                        'MS-STD': mss.std().item(),
                        'GAP': min_gap,
                        'GAP-AVG': _gaps.mean().item(),
                        'GAP-STD': _gaps.std().item(),
                        'TIME': exe_time})
        #
        gaps.append(min_gap)
        shape = f"{ins['j']}x{ins['m']}"
        if shape in shape_gap:
            shape_gap[shape].append(min_gap)
        else:
            shape_gap[shape] = [min_gap]
    #
    avg_gap = sum(gaps) / len(gaps)
    print(f"\t\t{name} set: AVG Gap={avg_gap:.3f}")
    for shape, g in shape_gap.items():
        print(f"\t\t\t{shape:6}: AVG Gap={sum(g)/len(g):.3f} "
              f"[{min(g):.3f}, {max(g):.3f}]")
    return avg_gap, results


#
parser = argparse.ArgumentParser(description='Test Pointer JSP')
parser.add_argument("-model_path", type=str, required=False,
                    default="./checkpoints/PointerNet-B256f.pt",
                    help="Path to the model.")
parser.add_argument("-benchmarks", type=list, required=False,
                    default=['DMU', 'TA'],
                    help="Output size of the encoder.")
parser.add_argument("-beta", type=int, default=128, required=False,
                    help="Number of sampled solutions.")
parser.add_argument("-seed", type=int, default=12345,
                    required=False, help="Random seed.")


if __name__ == '__main__':
    print(f"Testing on {dev}...")
    _args = parser.parse_args()

    # Load the model
    print(f"Loading {_args.model_path}")
    enc_, dec_ = torch.load(_args.model_path, map_location=dev)
    model_name = _args.model_path.rsplit('/', 1)[1].split('.', 1)[0]

    #
    for b_name in _args.benchmarks:
        d = load_dataset(f'./benchmarks/{b_name}', device=dev)
        _, res = validation(enc_, dec_, d,
                            beta=_args.beta,
                            name=f'{b_name} benchmark',
                            seed=_args.seed)

        # Save results
        if not os.path.exists('./output/'):
            os.makedirs('./output/')
        out_file = f'output/{model_name}_{b_name}-B{_args.beta}_{_args.seed}.csv'
        pd.DataFrame(res).to_csv(out_file, index=False, sep=',')
