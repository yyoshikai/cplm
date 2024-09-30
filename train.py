import sys, os
import argparse

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
args = parser.parse_args()
if args.test: args.studyname+='_test'
result_dir = f"results/{args.studyname}"
record_opt_step = 1 if args.test else 100
main_rank = 0
batch_first = False

# environments
os.makedirs(f"{result_dir}/models", exist_ok=True)
WORKDIR = os.environ.get('WORKDIR', "/workspace")
dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
rank = dist.get_rank()
size = dist.get_world_size()
device = torch.device('cuda', index=rank % torch.cuda.device_count()) \
    if torch.cuda.is_available() else torch.device('cpu')
is_main = rank == main_rank

# data
train_subset = 'valid' if args.test else 'train'
valid_subset = 'valid'
tokenizer = MoleculeProteinTokenizer()

train_mol_data = MoleculeDataset(f"{WORKDIR}/cheminfodata/unimol/ligands/{train_subset}.lmdb", 10, tokenizer)
train_prot_data = ProteinDataset(f"{WORKDIR}/cheminfodata/unimol/pockets/{train_subset}.lmdb", tokenizer)
train_data = ConcatDataset([train_mol_data, RepeatDataset(train_prot_data, 5)])

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
    if is_main:
        for i_process in range(size):
            send_batch = []
            batch_size = 0
            max_length = 0
            while True:
                if next_item is None:
                    try:
                        next_item = train_iter.__next__().squeeze(0)
                    except StopIteration:
                        train_iter = train_loader.__iter__()
                        next_item = train_iter.__next__().squeeze(0)
                if ((len(send_batch)+1) * max(max_length, len(next_item)) <= args.token_per_batch):
                    send_batch.append(next_item)
                    max_length = max(max_length, len(next_item))
                    n_accum_token += len(next_item)
                    next_item = None
                else:
                    break
            send_batch = pad_sequence(send_batch, batch_first=batch_first,
                    padding_value=tokenizer.pad_token).to(torch.long)
            print(f"send_batch: {send_batch.shape}")

            if i_process == rank:
                batch = send_batch.to(device)
            else:
                shape = torch.tensor(send_batch.shape, dtype=torch.int)
                send_batch = torch.cat([
                    send_batch.ravel(), 
                    torch.full(args.token_per_batch-send_batch.numel(), fill_value=tokenizer.pad_token)
                ])
                dist.send(send_batch, dst=i_process, tag=0)
                dist.send(shape, dst=i_process, tag=1)
    else:
        batch = torch.zeros(args.token_per_batch, dtype=torch.long)
        dist.recv(batch, src=main_rank, tag=0)
        shape = torch.zeros(2, dtype=torch.int)
        dist.recv(shape, src=main_rank, tag=1)
        batch = batch[:shape[0]*shape[1]].reshape(batch).to(device)
    
    pred = model(batch[:-1])
    loss = criterion(pred.reshape(-1, tokenizer.voc_size), batch[1:].ravel())
    loss.backward()
    accum_loss += loss.item()
    if n_accum_token >= args.token_per_step:
        print("optimizer stepped", flush=True)
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
            df.to_csv(f"{result_dir}/steps.csv")
            torch.save(model.state_dict(), f"{result_dir}/models/{opt_step}.pth")
        n_accum_token = 0

dist.destroy_process_group()