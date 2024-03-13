import torch
import random
import sampling as stg
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PointerNet import GATEncoder, MHADecoder
from argparse import ArgumentParser
from inout import load_dataset
from tqdm import tqdm
from utils import *

# Training device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Number of steps to wait before probing for improvements
PROBE_EVERY = 2500


@torch.no_grad()
def validation(encoder: torch.nn.Module,
               decoder: torch.nn.Module,
               val_set: list,
               num_sols: int = 16,
               seed: int = 12345):
    """
    Test the model at the end of each epoch.

    Args:
        encoder: Encoder.
        decoder: Decoder.
        val_set: Validation set.
        num_sols: Number of solution to generate for each instance.
        seed: Random seed.
    """
    if seed is not None:
        torch.manual_seed(seed)
    encoder.eval()
    decoder.eval()
    gaps = ObjMeter()

    # For each instance in the benchmark
    for ins in val_set:
        # Sample multiple solutions
        s, mss = stg.sampling(ins, encoder, decoder, bs=num_sols, device=device)

        # Log info
        min_gap = (mss.min().item() / ins['makespan'] - 1) * 100
        gaps.update(ins, min_gap)

    # Print stats
    avg_gap = gaps.avg
    print(f"\t\tVal set: AVG Gap={avg_gap:.3f}")
    print(gaps)
    return avg_gap


def train(encoder: torch.nn.Module,
          decoder: torch.nn.Module,
          train_set: list,
          val_set: list,
          opti: torch.optim.Optimizer,
          epochs: int = 50,
          virtual_bs: int = 128,
          num_sols: int = 128,
          scheduler=None,
          model_path: str = 'checkpoints/PointerNet.pt'):
    """
    Train the Pointer Network.

    Args:
        encoder: Encoder to train.
        decoder: Decoder to train.
        train_set: Training set.
        val_set: Validation set.
        opti: Optimizer.
        epochs: Number of epochs.
        virtual_bs: Virtual batch size that gives the number of instances
            predicted before back-propagation.
        num_sols: Number of solutions to use in back-propagation.
        scheduler:
        model_path:
    """
    frac, _best = 1. / virtual_bs, None
    size = len(train_set)
    indices = list(range(size))
    #
    c = torch.nn.CrossEntropyLoss(reduction='mean')
    print("Training ...")
    for epoch in range(epochs):
        losses = AverageMeter()
        gaps = ObjMeter()
        random.shuffle(indices)
        cnt = 0
        # For each instance in the training set
        for idx, i in tqdm(enumerate(indices)):
            ins = train_set[i]
            cnt += 1
            # Training step (sample solutions)
            trajs, logits, mss = stg.sample_training(ins, encoder, decoder,
                                                     bs=num_sols, device=device)

            # Compute loss
            ms, argmin = mss.min(-1)
            loss = c(logits[argmin], trajs[argmin])

            # log info
            losses.update(loss.item())
            gaps.update(ins, (ms.item() / ins['makespan'] - 1) * 100)

            # Virtual batching for managing without masking different sizes
            loss *= frac
            loss.backward()
            if cnt == virtual_bs or idx + 1 == size:
                opti.step()
                opti.zero_grad()
                cnt = 0

            # Probe model
            if idx > 0 and idx % PROBE_EVERY == 0:
                val_gap = validation(encoder, decoder, val_set, num_sols=128)
                if _best is None or val_gap < _best:
                    _best = val_gap
                    torch.save((encoder.state_dict(), decoder), model_path)

        # ...log the running loss
        avg_gap = gaps.avg
        print(f'\tEPOCH {epoch:02}: avg loss={losses.avg:.4f}')
        print(f"\t\tTrain: AVG Gap={avg_gap:2.3f}")
        print(gaps)

        # Test model and save
        val_gap = validation(encoder, decoder, val_set, num_sols=128)
        if _best is None or val_gap < _best:
            _best = val_gap
            torch.save((encoder.state_dict(), decoder), model_path)
        if scheduler is not None:
            scheduler.step(avg_gap)


#
parser = ArgumentParser(description='PointerNet arguments for the JSP')
parser.add_argument("-data_path", type=str, default="./dataset5k",
                    required=False, help="Path to the training data.")
parser.add_argument("-model_path", type=str, required=False,
                    default=None, help="Path to the model.")
parser.add_argument("-enc_hidden", type=int, default=64, required=False,
                    help="Hidden size of the encoder.")
parser.add_argument("-enc_out", type=int, default=128, required=False,
                    help="Output size of the encoder.")
parser.add_argument("-mem_hidden", type=int, default=64, required=False,
                    help="Hidden size of the memory network.")
parser.add_argument("-mem_out", type=int, default=128, required=False,
                    help="Output size of the memory network.")
parser.add_argument("-clf_hidden", type=int, default=128, required=False,
                    help="Hidden size of the classifier.")
parser.add_argument("-lr", type=float, default=0.0002, required=False,
                    help="Learning rate in the first checkpoint.")
parser.add_argument("-epochs", type=int, default=50, required=False,
                    help="Number of epochs.")
parser.add_argument("-batch_size", type=int, default=16, required=False,
                    help="Virtual batch size.")
parser.add_argument("-beta", type=int, default=32, required=False,
                    help="Number of sampled solutions.")
args = parser.parse_args()
print(args)


if __name__ == '__main__':
    print(f"Using device: {device}")

    ### TRAINING and VALIDATION
    train_set = load_dataset(args.data_path)
    val_set = load_dataset('./benchmarks/validation', device=device)

    ### MAKE MODEL
    _enc = GATEncoder(train_set[0]['x'].shape[1],
                      hidden_size=args.enc_hidden,
                      embed_size=args.enc_out).to(device)
    _dec = MHADecoder(encoder_size=_enc.out_size,
                      context_size=stg.JobShopStates.size,
                      hidden_size=args.mem_hidden,
                      mem_size=args.mem_out,
                      clf_size=args.clf_hidden).to(device)
    # Load model if necessary
    if args.model_path is not None:
        print(f"Loading {args.model_path}.")
        m_path = f"{args.model_path}"
        _enc_w, _dec = torch.load(args.model_path, map_location=device)
        _enc.load_state_dict(_enc_w)
    else:
        m_path = f"checkpoints/PtrNet-B{args.beta}_1.pt"
    print(_enc)
    print(_dec)

    ### OPTIMIZER
    optimizer = torch.optim.Adam(list(_enc.parameters()) +
                                 list(_dec.parameters()), lr=args.lr)
    _scheduler = ReduceLROnPlateau(optimizer,
                                   mode='min',
                                   verbose=True,
                                   factor=0.5,
                                   patience=5,
                                   cooldown=5,
                                   min_lr=0.000001)
    #
    train(_enc, _dec, train_set, val_set,
          optimizer,
          epochs=args.epochs,
          virtual_bs=args.batch_size,
          num_sols=args.beta,
          scheduler=_scheduler,
          model_path=m_path)
