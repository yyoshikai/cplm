"""
[pocket atoms][pocket XYZ][ligand SMILES]([ligand XYZ])[vina score]
"""
import sys, os, shutil
import argparse
from glob import glob

import yaml
from addict import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Subset
from torch.nn.parallel import DistributedDataParallel
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [WORKDIR, f"{WORKDIR}/cplm"]

from src.data import *
from src.data.tokenizer import TokenEncodeDataset, VocEncoder
from src.model import Model
from src.utils import set_logtime
from src.utils.path import timestamp2
from src.utils.logger import INFO_WORKER
from src.train import train, add_train_args, get_train_logger, sync_train_dir, MAIN_RANK

# arguments
parser = argparse.ArgumentParser()

## settings
parser.add_argument("--studyname", default='default')
parser.add_argument("--test", action='store_true')

## training
add_train_args(parser)

## data
parser.add_argument("--coord-range", type=float, help='Defaults to value in training')
parser.add_argument("--pocket-coord-heavy", action='store_true')
parser.add_argument("--finetune-save-dir", required=True)
parser.add_argument("--index-lmdb")
parser.add_argument("--no-score", action='store_true')

## criterion
parser.add_argument("--coord-weight", type=float, default=0.1)
parser.add_argument("--score-weight", type=float, default=10)

## pretrain
parser.add_argument("--pretrain-name", required=True)
parser.add_argument("--pretrain-step", type=int)

args = parser.parse_args()

# get finetune info
pretrain_dir = f"{WORKDIR}/cplm/training/results/{args.pretrain_name}"
pretrain_config = Dict(yaml.safe_load(open(f"{pretrain_dir}/config.yaml")))

# set default args
if args.test: args.studyname+='_test'
if args.record_opt_step is None:
    args.record_opt_step = 1 if args.test else 1000
if args.tokenizer_log_interval is None:
    args.tokenizer_log_interval = 10000 if args.test else int(1e7)
if args.coord_range is None:
    args.coord_range = pretrain_config.coord_range

## get last pretrain step
auto_pretrain_step = False
if args.pretrain_step is None:
    steps = [os.path.splitext(os.path.basename(step))[0] for step in glob(f"{pretrain_dir}/models/*")]
    steps = [int(step) for step in steps if step.isdigit()]
    args.pretrain_step = max(steps)
    auto_pretrain_step = True

batch_first = False
set_logtime(args.logtime)

# Environment
## DDP
dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
rank = dist.get_rank()
size = dist.get_world_size()
device = torch.device('cuda', index=rank % torch.cuda.device_count()) \
    if torch.cuda.is_available() else torch.device('cpu')
is_main = rank == MAIN_RANK

## make&sync result dir
result_dir = sync_train_dir(f"results/{timestamp2()}_{args.studyname}")


if is_main:
    os.makedirs(f"{result_dir}/cplm", exist_ok=True)
    shutil.copy2(f'{WORKDIR}/cplm/finetune.py', f"{result_dir}/cplm/finetune.py")
    shutil.copytree(f'{WORKDIR}/cplm/src', f"{result_dir}/cplm/src")

## logger
logger = get_train_logger(result_dir)
logger.info(f"num_workers={args.num_workers}")
if auto_pretrain_step:
    logger.info(f"pretrain_step was set to {args.pretrain_step}")

# data
cddata = CDDataset(args.finetune_save_dir, args.seed, mol_atom_h=True,
        mol_coord_h=True, pocket_coord_heavy=args.pocket_coord_heavy)
if args.index_lmdb is not None:
    index_data = LMDBDataset(args.index_lmdb, key_is_indexed=True)
    cddata = Subset(cddata, index_data)

pocket_atom, pocket_coord, lig_smi, lig_coord, score, _ \
            = untuple_dataset(cddata, 6)
protein_atom_tokenizer = ProteinAtomTokenizer()
float_tokenizer = FloatTokenizer(-args.coord_range, args.coord_range)
smiles_tokenizer = StringTokenizer(open(f"{WORKDIR}/cplm/src/data/smiles_tokens.txt").read().splitlines())

pocket_atom = TokenizeDataset(pocket_atom, protein_atom_tokenizer)
pocket_coord = ArrayTokenizeDataset(pocket_coord, float_tokenizer)
score = TokenizeDataset(score, float_tokenizer)
lig_smi = TokenizeDataset(lig_smi, smiles_tokenizer)
lig_coord = ArrayTokenizeDataset(lig_coord, float_tokenizer)

train_data = SentenceDataset('[POCKET]', pocket_atom, '[XYZ]', pocket_coord, 
        '[LIGAND]', lig_smi, '[XYZ]', lig_coord, '[SCORE]', score, '[END]')

vocs = train_data.vocs()
voc_encoder = VocEncoder(vocs)
train_data = TokenEncodeDataset(train_data, voc_encoder)
if not is_main:
    del train_data
    train_data = None

# Make dataset
train_loader = DDPStringCollateLoader(train_data, args.num_workers, 
    args.pin_memory, args.prefetch_factor, 
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

class PredictCELoss(nn.Module):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, voc_encoder: VocEncoder, seed: int, 
                coord_weight: float, score_weight: float):
        super().__init__()
        self.step = 0
        self.voc_encoder = voc_encoder
        self.seed = seed
        self.coord_weight = coord_weight
        self.score_weight = score_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        input: Tensor[L-1, B, N]
        target: Tensor[L-1, B]
        
        """
        
        # make weight
        L, B = target.shape
        weight = torch.zeros_like(target, dtype=torch.float) # [L, B]
        coord_count = torch.cumsum(target == self.voc_encoder.voc2i['[XYZ]'], dim=0)
        weight[coord_count >= 2] = self.coord_weight
        smi_count = torch.cumsum(target == self.voc_encoder.voc2i['[SCORE]'], dim=0)
        weight[smi_count >= 1] = self.score_weight
        end_count = torch.cumsum(target == self.voc_encoder.voc2i['[END]'], dim=0)
        weight[end_count >= 1] = 0
        weight = torch.cat([
            torch.zeros(1, B, dtype=torch.float, device=weight.device), weight[:-1]
        ], dim=0)

        # log tokens in initial few steps
        if self.step < 5:
            rstate = np.random.RandomState(self.seed+self.step)
            idxs = np.arange(target.shape[1])
            if len(idxs) > 10: 
                idxs = np.sort(rstate.choice(target.shape[1], size=10, replace=False))
            self.logger.log(INFO_WORKER, f"target of step {self.step}:")
            for idx in idxs:
                self.logger.log(INFO_WORKER, f"  [{idx:3}]={','.join(self.voc_encoder.decode(target[:,idx].cpu().tolist()))}")
                self.logger.log(INFO_WORKER, f"  weight[{idx:3}]={weight[:,idx].cpu().tolist()}")
        self.step += 1
        return torch.sum(F.cross_entropy(input.reshape(-1, self.voc_encoder.voc_size), target.ravel(), reduction='none')*weight.ravel())

criterion = PredictCELoss(voc_encoder, args.seed, args.coord_weight, args.score_weight)

train(args, train_loader, model, criterion, result_dir, voc_encoder.pad_token, device, 
    1 if args.test else 10000)

dist.destroy_process_group()




