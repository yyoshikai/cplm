"""
241017作成
- PDBデータから読み込み, PDBのみ学習するように変更

"""
import sys, os
import argparse
from time import time
import random
import math
import gc
import shutil
import logging, yaml
from logging.config import dictConfig

import psutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence

from src.data import MoleculeDataset, ProteinDataset, RepeatDataset, SliceDataset
from src.tokenizer import MoleculeProteinTokenizer
from src.model import Model
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path.append(WORKDIR)
from tools.path import timestamp, cleardir


parser = argparse.ArgumentParser()
parser.add_argument("--studyname", default='default')
parser.add_argument("--test", action='store_true')
parser.add_argument("--token-per-batch", type=int, default=25000)
parser.add_argument("--token-per-step", type=int, default=int(1.6e6))
parser.add_argument("--max-step", type=int, default=1000000)
parser.add_argument("--record-opt-step", type=int)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--clip_grad_norm", type=int, default=1.0)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--pin-memory", action='store_true')
parser.add_argument("--sdp-kernel", choices=['FLASH', 'CUDNN', 'MATH', 'EFFICIENT'])
parser.add_argument("--file-log-level", choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
parser.add_argument("--gc", action='store_true')
args = parser.parse_args()

if args.test: args.studyname+='_test'
result_dir = f"training/results/{timestamp()}_{args.studyname}"
if args.record_opt_step is None:
    args.record_opt_step = 1 if args.test else 100
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
os.makedirs(f"{result_dir}/optimizers", exist_ok=True)

log_config = yaml.safe_load(open("src/logging.yaml").read())
log_config['formatters']['default']['format'] = "[{asctime}]"+f"[{rank}/{size}]"+"[{levelname}] {message}"
log_config['handlers']['file']['filename'] = f"{result_dir}/log.txt"
log_config['handlers']['console']['level'] = 'DEBUG' if args.test else 'INFO'
log_config['handlers']['file']['level'] = args.file_log_level
dictConfig(log_config)
logger = logging.getLogger()

mem = psutil.virtual_memory()
logger.info(f"Total memory={mem.total/(2**30):.03f}")

# data
"""
train_subset = 'valid' if args.test else 'train'
valid_subset = 'valid'
tokenizer = MoleculeProteinTokenizer()

train_mol_data = MoleculeDataset(f"{WORKDIR}/cheminfodata/unimol/ligands/{train_subset}.lmdb", 10, tokenizer)
train_prot_data = ProteinDataset(f"{WORKDIR}/cheminfodata/unimol/pockets/{train_subset}.lmdb", tokenizer)
train_data = ConcatDataset([train_mol_data, RepeatDataset(train_prot_data, 5)])
train_data = SliceDataset(train_data, size, rank)

valid_mol_data = MoleculeDataset(f"{WORKDIR}/cheminfodata/unimol/ligands/{valid_subset}.lmdb", 10, tokenizer)
valid_prot_data = ProteinDataset(f"{WORKDIR}/cheminfodata/unimol/pockets/{valid_subset}.lmdb", tokenizer)


train_loader = DataLoader(train_data, shuffle=True, num_workers=args.num_workers, pin_memory=args.pin_memory)
train_iter = train_loader.__iter__()
next_item = None
n_accum_token = 0
"""
structure_data = PDBFragmentDataset(f"{WORKDIR}/cheminfodata/pdb/240101/mmCIF", mmcif=True)









# model
tokenizer = MoleculeProteinTokenizer()
model = Model(8, 768, 12, 4, 0.1, 'gelu', True, 
        tokenizer.voc_size, tokenizer.pad_token)
model.to(torch.bfloat16)
model.to(device)
model = DistributedDataParallel(model)
criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=tokenizer.pad_token)
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
optimizer.zero_grad()

def schedule(step):
    if step <= 2000:
        return step / 2000
    elif step <= 55000:
        return math.cos(math.pi*((step-2000)/(55000-2000)))*0.49+0.51
    else:
        return 0.02
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

accum_loss = 0
opt_step = 0
accum_losses = []
accum_n_tokens = []
lrs = []
mems = []

data_times = []
loss_times = []

# n_prot = 0
# n_total = 0
for step in range(args.max_step):

    # get batch
    data_start = time()
    batch = []
    max_length = 0
    while True:
        if next_item is None:
            try:
                next_item = train_iter.__next__().squeeze(0)
            except StopIteration:
                logger.info(f"rank {rank}: epoch finished at step {step}")
                train_iter = train_loader.__iter__()
                next_item = train_iter.__next__().squeeze(0)
        if ((len(batch)+1) * max(max_length, len(next_item)) <= args.token_per_batch):
            batch.append(next_item)
            max_length = max(max_length, len(next_item))
            # if next_item[0] == 3: n_prot += 1
            n_accum_token += len(next_item)
            next_item = None
        else:
            break
    # n_total += len(batch)
    # logger.info(f"n_prot={n_prot}/{n_total}")
    batch = pad_sequence(batch, batch_first=batch_first,
            padding_value=tokenizer.pad_token).to(torch.long)
    batch = batch.to(device)

    logger.info("batch:")

    batch_list = batch.T.cpu().numpy().tolist()
    for item in batch_list:
        if(item[0] == 3):
            logger.info(item)

    data_end = time()
    data_times.append(data_end-data_start)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred = model(batch[:-1])
        loss = criterion(pred.reshape(-1, tokenizer.voc_size), batch[1:].ravel())
        loss.backward()
    accum_loss += loss.item()
    loss_end = time()
    loss_times.append(loss_end-data_end)

    # sum accum_token
    logger.debug(f"n_accum_token={n_accum_token}")
    reduced_accum_token = torch.tensor(n_accum_token, dtype=torch.int, device=device)
    dist.all_reduce(reduced_accum_token)
    logger.debug(f"reduced_accum_token={reduced_accum_token}")

    logger.debug(f"loss={loss.item()}")
    if reduced_accum_token >= args.token_per_step:
        logger.debug("optimizer stepped")
        optim_start = time()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        optim_end = time()
        opt_step += 1
        accum_losses.append(accum_loss)
        accum_n_tokens.append(n_accum_token)
        lrs.append(scheduler.get_last_lr())

        mem = psutil.virtual_memory()
        mems.append(mem.used/(2**30))

        scheduler.step()
        if opt_step % args.record_opt_step == 0:
            df = pd.DataFrame({
                'loss': accum_losses,
                'n_token': accum_n_tokens,
                'lr': lrs,
                'memory': mems
            })
            df.to_csv(f"{result_dir}/step_data/{rank}.csv")
            df = pd.DataFrame({
                'data_time': data_times,
                'forward_time': loss_times,
            })
            df.to_csv(f"{result_dir}/step_data/{rank}_step.csv")
            if is_main:
                torch.save(model.state_dict(), f"{result_dir}/models/{opt_step}.pth")
                cleardir(f"{result_dir}/optimizers")
                torch.save(optimizer.state_dict(), f"{result_dir}/optimizers/{opt_step}.pth")
            if args.gc:
                gc.collect()
        logger.debug(f"optim_time={optim_end-optim_start:.03f}")
        n_accum_token = 0
        accum_loss = 0
    break

dist.destroy_process_group()
