import torch
import random
import sampling as stg
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PointerNet import GATEncoder, MHADecoder
from argparse import ArgumentParser
from inout import load_dataset
from tqdm import tqdm

# Training device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    gaps, shape_gap = [], {}
    for ins in val_set:
        s, mss = stg.sampling(ins, encoder, decoder, bs=num_sols, device=device)
        #
        ms_min = mss.min().item()
        min_gap = (ms_min / ins['makespan'] - 1) * 100
        gaps.append(min_gap)
        shape = f"{ins['j']}x{ins['m']}"
        if shape in shape_gap:
            shape_gap[shape].append(min_gap)
        else:
            shape_gap[shape] = [min_gap]
    #
    avg_gap = sum(gaps) / len(gaps)
    print(f"\t\tVal set: AVG Gap={avg_gap:.3f}")
    for shape, g in shape_gap.items():
        print(f"\t\t\t{shape:6}: AVG Gap={sum(g)/len(g):2.3f} "
              f"[{min(g):.3f}, {max(g):.3f}]")
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
        encoder.train()
        decoder.train()
        info = {}
        random.shuffle(indices)
        cnt = 0
        #
        for idx, i in tqdm(enumerate(indices)):
            ins = train_set[i]
            cnt += 1
            # Training step
            trajs, logits, mss = stg.sample_training(ins, encoder, decoder,
                                                     bs=num_sols, device=device)
            argmin = mss.argmin()
            loss = c(logits[argmin], trajs[argmin])
            # log info
            shape = ins['shape']
            if shape in info:
                info[shape][0].append(loss.detach())
                info[shape][1].append((mss[argmin] / ins['makespan'] - 1) * 100)
            else:
                info[shape] = [[loss.detach()],
                               [(mss[argmin] / ins['makespan'] - 1) * 100]]
            loss *= frac
            loss.backward()
            # Virtual batching for managing without masking different sizes
            if cnt == virtual_bs or idx + 1 == size:
                opti.step()
                opti.zero_grad()
                cnt = 0
            # Probe model
            if idx > 0 and idx % 2500 == 0:
                val_gap = validation(encoder, decoder, val_set, num_sols=128)
                if _best is None or val_gap < _best:
                    _best = val_gap
                    torch.save((encoder, decoder), model_path)
                encoder.train()
                decoder.train()

        # ...log the running loss
        avg_loss = sum([sum(l) for l, _ in info.values()]) / len(train_set)
        avg_gap = sum([sum(g) for _, g in info.values()]) / len(train_set)
        print(f'\tEPOCH {epoch:02}: avg loss={avg_loss:.4f}')
        print(f"\t\tTrain: AVG Gap={avg_gap:2.3f}")
        for shape, (l, g) in info.items():
            print(f"\t\t\t{shape:5}: AVG Gap={sum(g)/len(g):2.3f} "
                  f"[{min(g):.3f}, {max(g):.3f}]")
            print(f"\t\t\t{shape:5}: AVG loss={sum(l)/len(l):.3f}")

        # Test model and save
        val_gap = validation(encoder, decoder, val_set, num_sols=128)
        if _best is None or val_gap < _best:
            _best = val_gap
            torch.save((encoder, decoder), model_path)
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
parser.add_argument("-beta", type=int, default=128, required=False,
                    help="Number of sampled solutions.")


if __name__ == '__main__':
    print(f"Using device: {device}")
    _args = parser.parse_args()

    ### TRAINING and VALIDATION
    train_set = load_dataset(_args.data_path)
    val_set = load_dataset('./benchmarks/validation', device=device)

    ### MAKE MODEL
    _enc = GATEncoder(train_set[0]['x'].shape[1],
                      hidden_size=_args.enc_hidden,
                      embed_size=_args.enc_out).to(device)
    _dec = MHADecoder(encoder_size=_enc.out_size,
                      context_size=stg.JobShopStates.size,
                      hidden_size=_args.mem_hidden,
                      mem_size=_args.mem_out,
                      clf_size=_args.clf_hidden).to(device)
    # Load model if necessary
    if _args.model_path is not None:
        print(f"Loading {_args.model_path}.")
        m_path = f"{_args.model_path}"
        _encw, _decw = torch.load(_args.model_path, map_location=device)
        _enc.load_state_dict(_encw)
        _dec.load_state_dict(_decw)
    else:
        m_path = f"checkpoints/PointerNet-B{_args.beta}.pt"
    print(_enc)
    print(_dec)

    ### OPTIMIZER
    optimizer = torch.optim.Adam(list(_enc.parameters()) +
                                 list(_dec.parameters()),
                                 lr=_args.lr)
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
          epochs=_args.epochs,
          virtual_bs=_args.batch_size,
          num_sols=_args.beta,
          scheduler=_scheduler,
          model_path=m_path)
