import sys, os, argparse, yaml
from addict import Dict
from glob import glob
import torch
import torch.distributed as dist
from src.model import Model
from src.data import CDDataset, untuple_dataset
from src.data.tokenizer import ProteinAtomTokenizer, FloatTokenizer, TokenizeDataset, ArrayTokenizeDataset, StringTokenizer
from src.utils.train import MAIN_RANK

# arguments
parser = argparse.ArgumentParser()

## trainings
parser.add_argument("--seed", type=int, default=0)

## data
parser.add_argument('--finetune-save-dir', required=True)
parser.add_argument("--pocket-coord-heavy", action='store_true')

## finetune
parser.add_argument("--finetune-name", required=True)
parser.add_argument("--finetune-step", type=int)

args = parser.parse_args()

# get finetune info
finetune_dir = f"finetune/results/{args.finetune_name}"
finetune_config = Dict(yaml.safe_load(open(f"{finetune_dir}/config.yaml")))

## get last pretrain step
auto_pretrain_step = False
if args.pretrain_step is None:
    steps = [os.path.splitext(os.path.basename(step))[0] for step in glob(f"{finetune_dir}/models/*")]
    steps = [int(step) for step in steps if step.isdigit()]
    args.pretrain_step = max(steps)
    auto_pretrain_step = True

# set default args
if args.test: args.studyname+='_test'
if args.record_opt_step is None:
    args.record_opt_step = 1 if args.test else 1000
if args.tokenizer_log_interval is None:
    args.tokenizer_log_interval = 10000 if args.test else int(1e7)
if args.coord_range is None:
    args.coord_range = finetune_config.coord_range



# DDP
dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
rank = dist.get_rank()
size = dist.get_world_size()
device = torch.device('cuda', index=rank % torch.cuda.device_count()) \
    if torch.cuda.is_available() else torch.device('cpu')
is_main = rank == MAIN_RANK

# data
cddata = CDDataset(args.finetune_save_dir, args.seed, mol_atom_h=True,
        mol_coord_h=True, pocket_coord_heavy=args.pocket_coord_heavy)

from src.data import SentenceDataset
class ReinforceDataset(SentenceDataset):
    def __init__(self, cddataset: CDDataset, 
        protein_atom_tokenizer: ProteinAtomTokenizer, 
        float_tokenizer: FloatTokenizer):
        pocket_atom, pocket_coord, lig_smi, lig_coord, score, _ \
            = untuple_dataset(cddataset, 6)
        pocket_atom = TokenizeDataset(pocket_atom, protein_atom_tokenizer)
        pocket_coord = ArrayTokenizeDataset(pocket_coord, float_tokenizer)

        super().__init__('[POCKET]', pocket_atom, '[XYZ]', pocket_coord, '[LIGAND]')
train_data = ReinforceDataset(
    cddata, 
    ProteinAtomTokenizer(), 
    FloatTokenizer(-args.coord_range, args.coord_range))
smiles_tokenizer = StringTokenizer(open("src/data/smiles_tokens.txt").read().splitlines())

# load state dict(for vocs)
state_dict = torch.load(f"{finetune_dir}/models/{args.pretrain_step}.pth", 
    map_location=device, weights_only=True)









