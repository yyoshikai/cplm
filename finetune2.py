# 241025 若干argumentを変更した。
import sys, os, shutil
import argparse
from glob import glob
from logging import getLogger
import yaml
from addict import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import Subset
from torch.nn.parallel import DistributedDataParallel
import transformers.utils.logging
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path.append(WORKDIR)

from src.data.datasets.crossdocked import CDDataset, CDProteinDataset
from src.data.protein import ProteinProcessDataset
from src.data.molecule import MolProcessDataset
from src.data.datasets.moleculenet import MoleculeNetTrainDataset
from src.data.coord import CoordTransformDataset
from src.data import untuple
from src.data.lmdb import IntLMDBDataset
from src.data.tokenizer import TokenEncodeDataset, VocEncoder, \
        ProteinAtomTokenizer, FloatTokenizer, StringTokenizer
from src.data.collator import DDPStringCollateLoader
from src.data.tokenizer import ProteinAtomTokenizer, FloatTokenizer, StringTokenizer, \
    TokenizeDataset, ArrayTokenizeDataset, SentenceDataset
from src.data.protein import CoordFollowDataset
from src.model import Model, MambaModel2
from src.utils import set_logtime
from src.utils.path import timestamp
from src.train import WeightedCELoss, train, add_train_args, get_train_logger, sync_train_dir, MAIN_RANK
from src.utils.logger import INFO_WORKER

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
parser.add_argument('--protein', action='store_true')

## dataset
parser.add_argument('--dataset-name', required=True)
parser.add_argument('--task-name', required=True)

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
torch.cuda.set_device(rank % torch.cuda.device_count())
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

### logging on other libraries
transformers.utils.logging.enable_propagation()
transformers.utils.logging.disable_default_handler()

# data
## pocket and ligands
rstate = np.random.RandomState(args.seed)
mol, target = untuple(MoleculeNetTrainDataset(args.dataset_name, args.task_name), 2)
mol_smi, mol_coord = untuple(MolProcessDataset(mol, rstate, h_atom=True, h_coord=True, randomize=True), 2)

mol_coord, _center, _rotation_matrix \
    = untuple(CoordTransformDataset(mol_coord, rstate=rstate, normalize_coord=True, random_rotate=True), 3)

## sentence
sentence = []
float_tokenizer = FloatTokenizer(-args.coord_range, args.coord_range)
mol_smi = TokenizeDataset(mol_smi, StringTokenizer(open("src/data/smiles_tokens.txt").read().splitlines()))
mol_coord = ArrayTokenizeDataset(mol_coord, float_tokenizer)
target = TokenizeDataset(target, float_tokenizer)
sentence += ['[LIGAND]', mol_smi, '[XYZ]', mol_coord, '[SCORE]', target, '[END]']
train_data = SentenceDataset(*sentence)

## vocs
vocs = train_data.vocs()
voc_encoder = VocEncoder(vocs)

train_data = TokenEncodeDataset(train_data, voc_encoder)

if not is_main:
    del train_data

# model
if targs.get('mamba', False):
    model = MambaModel2(voc_encoder.i2voc, voc_encoder.pad_token, '[END]')
else:
    model = Model(8, 768, 12, 4, 0.1, 'gelu', True, voc_encoder.i2voc, voc_encoder.pad_token)
model.to(torch.bfloat16)
model.to(device)
model = DistributedDataParallel(model)


# DataLoader
train_loader = DDPStringCollateLoader(train_data, model.module, args.num_workers, args.pin_memory, args.prefetch_factor, args.gpu_size_gb*2**30, batch_first, voc_encoder.pad_token, True, args.sdp_kernel, device, MAIN_RANK, seed=args.seed)

## load state dict
state_dict = torch.load(f"{pretrain_dir}/models/{args.pretrain_step}.pth", 
    map_location=device, weights_only=True)
model.load_state_dict(state_dict)


class WeightedCELoss(nn.Module):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, voc_encoder: VocEncoder, seed: int, split_vocs: list[str], weights: list[float]):
        super().__init__()
        self.step = 0
        self.voc_encoder = voc_encoder
        self.seed = seed
        self.split_is = [self.voc_encoder.voc2i[voc] for voc in split_vocs]
        self.weights = weights

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        input: Tensor[L-1, B, N]
        target: Tensor[L-1, B]
        
        """
        
        # make weight
        L, B = target.shape
        weight = torch.zeros_like(target, dtype=torch.float) # [L, B]
        
        region = torch.zeros_like(target, dtype=torch.int)
        for i in self.split_is:
            region += torch.cumsum(target == i, dim=0)
        
        for ir, w in enumerate(self.weights):
            weight[region == ir] = w
        
        weight = torch.cat([
            torch.full((1, B), fill_value=self.weights[0], dtype=torch.float, 
                    device=target.device), 
            weight[:-1]
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


criterion = WeightedCELoss(voc_encoder, args.seed, ['[SCORE]', '[END]'], 
        [0, 1, 0])

train(args, train_loader, model, criterion, result_dir, voc_encoder.pad_token, device, 1 if args.test else 10000)

dist.destroy_process_group()
