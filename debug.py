import sys, os
import argparse
from time import time
import random
import math
import gc
import psutil
import logging, yaml
from logging.config import dictConfig

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence

from src.data import LMDBDataset, MoleculeDataset, ProteinDataset, RepeatDataset, SliceDataset
from src.tokenizer import MoleculeProteinTokenizer
from src.model import Model
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path.append(WORKDIR)
from tools.path import timestamp


parser = argparse.ArgumentParser()
parser.add_argument("--studyname", default='default')
parser.add_argument("--test", action='store_true')
parser.add_argument("--batch-size", type=int, default=50)
parser.add_argument("--step-size", type=int, default=4000)
parser.add_argument("--max-step", type=int, default=1000000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--clip_grad_norm", type=int, default=1.0)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--pin-memory", action='store_true')
parser.add_argument("--sdp-kernel", choices=['FLASH', 'CUDNN', 'MATH', 'EFFICIENT'])
parser.add_argument("--file-log-level", choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='DEBUG')
parser.add_argument("--gc", action='store_true')
# for debug
parser.add_argument("--lmdb", action='store_true')
parser.add_argument("--mol", action='store_true')
parser.add_argument("--no-env", action='store_true')
parser.add_argument("--no-txn", action='store_true')

args = parser.parse_args()

if args.test: args.studyname+='_test'
result_dir = f"training/results/{timestamp()}_{args.studyname}"
record_opt_step = 1 if args.test else 100
main_rank = 0
batch_first = False

# environments
dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
rank = dist.get_rank()
size = dist.get_world_size()
device = torch.device('cuda', index=rank % torch.cuda.device_count()) \
    if torch.cuda.is_available() else torch.device('cpu')
is_main = rank == main_rank

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if args.test:
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

if args.sdp_kernel is not None:
    torch.backends.cuda.enable_flash_sdp(args.sdp_kernel == 'FLASH')
    torch.backends.cuda.enable_cudnn_sdp(args.sdp_kernel == 'CUDNN')
    torch.backends.cuda.enable_math_sdp(args.sdp_kernel == 'MATH')
    torch.backends.cuda.enable_mem_efficient_sdp(args.sdp_kernel == 'EFFICIENT')



os.makedirs(f"{result_dir}/models", exist_ok=True)
os.makedirs(f"{result_dir}/step_data", exist_ok=True)

log_config = yaml.safe_load(open("src/logging.yaml").read())
log_config['formatters']['default']['format'] = "[{asctime}]"+f"[{rank}/{size}]"+"[{levelname}] {message}"
log_config['handlers']['file']['filename'] = f"{result_dir}/log.txt"
log_config['handlers']['console']['level'] = 'DEBUG' if args.test else 'INFO'
log_config['handlers']['file']['level'] = args.file_log_level
dictConfig(log_config)
logger = logging.getLogger()

# data
train_subset = 'valid' if args.test else 'train'
valid_subset = 'valid'
tokenizer = MoleculeProteinTokenizer()

if args.lmdb:
    train_data = LMDBDataset(f"{WORKDIR}/cheminfodata/unimol/ligands/{train_subset}.lmdb",
            keep_env=not args.no_env, keep_txn=not args.no_txn, key_is_indexed=True)
else:
    train_mol_data = MoleculeDataset(f"{WORKDIR}/cheminfodata/unimol/ligands/{train_subset}.lmdb", 10, tokenizer)
    if args.mol:
        train_data = train_mol_data
    else:
        train_prot_data = ProteinDataset(f"{WORKDIR}/cheminfodata/unimol/pockets/{train_subset}.lmdb", tokenizer)
        train_data = ConcatDataset([train_mol_data, RepeatDataset(train_prot_data, 5)])
        train_data = SliceDataset(train_data, size, rank)

train_loader = DataLoader(train_data, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
train_iter = train_loader.__iter__()
next_item = None
n_accum_token = 0

# model
data_times = []

for step in range(args.max_step):

    # get batch
    data_start = time()
    batch = []
    max_length = 0
    while True:
        if next_item is None:
            try:
                next_item = train_iter.__next__()
            except StopIteration:
                logger.info(f"rank {rank}: epoch finished at step {step}")
                train_iter = train_loader.__iter__()
                next_item = train_iter.__next__()
        if ((len(batch)+1) <= args.batch_size):
            batch.append(next_item)
            max_length = max(max_length, len(next_item))
            n_accum_token += 1
            next_item = None
        else:
            break
    data_end = time()
    data_times.append(data_end-data_start)

    # sum accum_token
    logger.debug(f"n_accum_token={n_accum_token}")
    reduced_accum_token = torch.tensor(n_accum_token, dtype=torch.int, device=device)
    dist.all_reduce(reduced_accum_token)
    logger.debug(f"reduced_accum_token={reduced_accum_token}")

    if reduced_accum_token >= args.step_size:
        logger.debug("optimizer stepped")
        n_accum_token = 0
        accum_loss = 0
    
        if args.gc:
            gc.collect()
    
    mem = psutil.virtual_memory()
    logger.debug(f"memory={mem.used/(2**30):.03f}/{mem.total/(2**30):.03f}")

dist.destroy_process_group()
