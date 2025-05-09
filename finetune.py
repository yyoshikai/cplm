# 241025 若干argumentを変更した。
import sys, os, shutil
import argparse
from glob import glob

import yaml
from addict import Dict
import torch
import torch.distributed as dist
from torch.utils.data import Subset
from torch.nn.parallel import DistributedDataParallel
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path.append(WORKDIR)

from src.data import *
from src.data.tokenizer import TokenEncodeDataset, VocEncoder
from src.model import Model
from src.utils import set_logtime
from src.utils.path import timestamp
from src.train import WeightedCELoss, train, add_train_args, get_train_logger, sync_train_dir, MAIN_RANK

# arguments
parser = argparse.ArgumentParser()

## settings
parser.add_argument("--studyname", default='default')
parser.add_argument("--test", action='store_true')

## training
add_train_args(parser)
parser.add_argument('--pocket-atom-weight', type=float, default=0.0)
parser.add_argument('--pocket-coord-weight', type=float, default=0.0)
parser.add_argument('--lig-smiles-weight', type=float, default=1.0)
parser.add_argument('--lig-coord-weight', type=float, default=5.0)

## data
parser.add_argument("--coord-range", type=float, help='Defaults to value in training')
parser.add_argument("--pocket-coord-heavy", action='store_true')
parser.add_argument("--finetune-save-dir", required=True)
parser.add_argument("--index-lmdb")
parser.add_argument("--no-score", action='store_true')

## pretrain
parser.add_argument("--pretrain-name", required=True)
parser.add_argument("--pretrain-step", type=int)

args = parser.parse_args()

# get finetune info
pretrain_dir = f"training/results/{args.pretrain_name}"
targs = Dict(yaml.safe_load(open(f"{pretrain_dir}/config.yaml")))

## get last pretrain step
auto_pretrain_step = False
if args.pretrain_step is None:
    steps = [os.path.splitext(os.path.basename(step))[0] for step in glob(f"{pretrain_dir}/models/*")]
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
    args.coord_range = targs.coord_range

batch_first = False
set_logtime(args.logtime)

## DDP
dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
rank = dist.get_rank()
size = dist.get_world_size()
device = torch.device('cuda', index=rank % torch.cuda.device_count()) \
    if torch.cuda.is_available() else torch.device('cpu')
is_main = rank == MAIN_RANK

## make&sync result dir
result_dir = sync_train_dir(f"finetune/results/{timestamp()}_{args.studyname}")


if is_main:
    os.makedirs(f"{result_dir}/cplm", exist_ok=True)
    shutil.copy2('finetune.py', f"{result_dir}/cplm/finetune.py")
    shutil.copytree('src', f"{result_dir}/cplm/src")

## logger
logger = get_train_logger(result_dir)
logger.info(f"num_workers={args.num_workers}")
if auto_pretrain_step:
    logger.info(f"pretrain_step was set to {args.pretrain_step}")

# data
cddata = CDDataset(args.finetune_save_dir, args.seed, mol_atom_h=True,
        mol_coord_h=True, pocket_coord_heavy=args.pocket_coord_heavy)
if args.index_lmdb is not None:
    index_data = PickleLMDBDataset(args.index_lmdb, idx_to_key='str')
    cddata = Subset(cddata, index_data)
train_data = FinetuneDataset(cddata, 
    ProteinAtomTokenizer(), 
    FloatTokenizer(-args.coord_range, args.coord_range), 
    StringTokenizer(open("src/data/smiles_tokens.txt").read().splitlines()),
    not args.no_score, coord_follow_atom=targs.get('coord_follow_atom', False))

vocs = train_data.vocs()
voc_encoder = VocEncoder(vocs)
train_data = TokenEncodeDataset(train_data, voc_encoder)
if not is_main:
    del train_data
    train_data = None

# Make dataset
train_loader = DDPStringCollateLoader(train_data, args.num_workers, args.pin_memory, args.prefetch_factor, 
    args.token_per_batch, batch_first, voc_encoder.pad_token, device, MAIN_RANK)

# model
model = Model(8, 768, 12, 4, 0.1, 'gelu', True, voc_encoder.i2voc, voc_encoder.pad_token)
model.to(torch.bfloat16)
model.to(device)
model = DistributedDataParallel(model)

## load state dict
state_dict = torch.load(f"{pretrain_dir}/models/{args.pretrain_step}.pth", 
    map_location=device, weights_only=True)
model.load_state_dict(state_dict)

criterion = WeightedCELoss(voc_encoder, args.seed, 
        args.pocket_atom_weight, args.pocket_coord_weight, 
        args.lig_smiles_weight, args.lig_coord_weight)

train(args, train_loader, model, criterion, result_dir, voc_encoder.pad_token, device, 
        1 if args.test else 10000)

dist.destroy_process_group()
