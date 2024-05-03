import argparse
import torch
import pandas as pd
import os
from tqdm import tqdm
from sampling import sampling, greedy
from inout import load_dataset
from time import time
from utils import ObjMeter

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
    gaps = ObjMeter()
    results = []

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
        gaps.update(ins, min_gap)
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
    print(f"\t\t{name} set: AVG Gap={gaps.avg:2.3f}")
    print(gaps)
    return results


#
parser = argparse.ArgumentParser(description='Test Pointer Net')
parser.add_argument("-model_path", type=str, required=False,
                    default="./checkpoints/PtrNet-B256.pt",
                    help="Path to the model.")
parser.add_argument("-benchmarks", type=list, required=False,
                    default=['DMU', 'TA'],
                    help="Name of the benchmark to use for testing.")
parser.add_argument("-beta", type=int, default=128, required=False,
                    help="Number of sampled solutions for each instance.")
parser.add_argument("-seed", type=int, default=12345,
                    required=False, help="Random seed.")


if __name__ == '__main__':
    from PointerNet import GATEncoder
    print(f"Testing on {dev}...")
    args = parser.parse_args()

    # Load the model
    print(f"Loading {args.model_path}")
    enc_w, dec_ = torch.load(args.model_path, map_location=dev)
    enc_ = GATEncoder(15).to(dev)   # Load weights to avoid bug with new PyG
    enc_.load_state_dict(enc_w)
    model_name = args.model_path.rsplit('/', 1)[1].split('.', 1)[0]

    #
    for b_name in args.benchmarks:
        d = load_dataset(f'./benchmarks/{b_name}', device=dev)
        res = validation(enc_, dec_, d,
                         beta=args.beta,
                         name=f'{b_name} benchmark',
                         seed=args.seed)

        # Save results
        if not os.path.exists('./output/'):
            os.makedirs('./output/')
        out_file = f'output/{model_name}_{b_name}-B{args.beta}_{args.seed}.csv'
        pd.DataFrame(res).to_csv(out_file, index=False, sep=',')
