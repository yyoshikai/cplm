import sys, os
import argparse
from time import time
import random

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

parser = argparse.ArgumentParser()
parser.add_argument("--studyname", default='default')
parser.add_argument("--test", action='store_true')
parser.add_argument("--token-per-batch", type=int, default=25000)
parser.add_argument("--token-per-step", type=int, default=int(1.6e5))
parser.add_argument("--max-step", type=int, default=1000000)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
if args.test: args.studyname+='_test'
result_dir = f"training/results/{args.studyname}"
record_opt_step = 1 if args.test else 100
main_rank = 0
batch_first = False

# environments
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if args.test:
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

WORKDIR = os.environ.get('WORKDIR', "/workspace")
dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
rank = dist.get_rank()
size = dist.get_world_size()
device = torch.device('cuda', index=rank % torch.cuda.device_count()) \
    if torch.cuda.is_available() else torch.device('cpu')
is_main = rank == main_rank

if is_main:
    os.makedirs(f"{result_dir}/models", exist_ok=True)
    os.makedirs(f"{result_dir}/step_data", exist_ok=True)


# data
train_subset = 'valid' if args.test else 'train'
valid_subset = 'valid'
tokenizer = MoleculeProteinTokenizer()

train_mol_data = MoleculeDataset(f"{WORKDIR}/cheminfodata/unimol/ligands/{train_subset}.lmdb", 10, tokenizer)
train_prot_data = ProteinDataset(f"{WORKDIR}/cheminfodata/unimol/pockets/{train_subset}.lmdb", tokenizer)
train_data = ConcatDataset([train_mol_data, RepeatDataset(train_prot_data, 5)])
train_data = SliceDataset(train_data, size, rank)

valid_mol_data = MoleculeDataset(f"{WORKDIR}/cheminfodata/unimol/ligands/{valid_subset}.lmdb", 10, tokenizer)
valid_prot_data = ProteinDataset(f"{WORKDIR}/cheminfodata/unimol/pockets/{valid_subset}.lmdb", tokenizer)


train_loader = DataLoader(train_data, shuffle=True)
train_iter = train_loader.__iter__()
next_item = None
n_accum_token = 0

# model
tokenizer = MoleculeProteinTokenizer()
model = Model(8, 768, 12, 4, 0.1, 'gelu', batch_first, True, 
        tokenizer.voc_size, tokenizer.pad_token)
model.to(device)
model = DistributedDataParallel(model)
criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=tokenizer.pad_token)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
optimizer.zero_grad()
accum_loss = 0
opt_step = 0
accum_losses = []
accum_n_tokens = []

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
                print(f"rank {rank}: epoch finished at step {step}", flush=True)
                train_iter = train_loader.__iter__()
                next_item = train_iter.__next__().squeeze(0)
        if ((len(batch)+1) * max(max_length, len(next_item)) <= args.token_per_batch):
            batch.append(next_item)
            max_length = max(max_length, len(next_item))
            n_accum_token += len(next_item)
            next_item = None
        else:
            break
    batch = pad_sequence(batch, batch_first=batch_first,
            padding_value=tokenizer.pad_token).to(torch.long)
    batch = batch.to(device)
    data_end = time()

    pred = model(batch[:-1])
    loss = criterion(pred.reshape(-1, tokenizer.voc_size), batch[1:].ravel())
    loss.backward()
    accum_loss += loss.item()
    loss_end = time()
    print(loss.item(), flush=True)
    if n_accum_token >= args.token_per_step:
        # print("optimizer stepped", flush=True)
        optimizer.step()
        optimizer.zero_grad()
        opt_step += 1
        accum_losses.append(accum_loss)
        accum_n_tokens.append(n_accum_token)
        if opt_step % record_opt_step == 0:
            df = pd.DataFrame({
                'loss': accum_losses,
                'n_token': accum_n_tokens
            })
            df.to_csv(f"{result_dir}/step_data/{rank}.csv")
            if is_main:
                torch.save(model.state_dict(), f"{result_dir}/models/{opt_step}.pth")
        n_accum_token = 0
        accum_loss = 0
    optim_end = time()

    # print(f"rank {rank}: data={data_end-data_start:.03f}, loss={loss_end-data_end:.03f}, optim={optim_end-data_end:.03f}")

dist.destroy_process_group()