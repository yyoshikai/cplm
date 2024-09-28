import sys, os
import argparse

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
parser.add_argument("--data-dir")
parser.add_argument("--test", action='store_true')
parser.add_argument("--token-per-batch", type=int, default=25000)
parser.add_argument("--token-per-step", type=int, default=int(1.6e5))
parser.add_argument("--max-step", type=int, default=1000000)
parser.add_argument("--lr", type=float, default=1e-5)
args = parser.parse_args()
main_rank = 0
batch_first = False

# environments
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

train_mol_data = MoleculeDataset(f"{WORKDIR}/cheminfodata/unimol/ligands/{train_subset}.lmdb", tokenizer)
train_prot_data = ProteinDataset(f"{WORKDIR}/cheminfodata/unimol/pockets/{train_subset}.lmdb", tokenizer)
train_data = ConcatDataset([train_mol_data, RepeatDataset(train_prot_data, 5)])

valid_mol_data = MoleculeDataset(f"{WORKDIR}/cheminfodata/unimol/ligands/{valid_subset}.lmdb", tokenizer)
valid_prot_data = MoleculeDataset(f"{WORKDIR}/cheminfodata/unimol/pockets/{valid_subset}.lmdb", tokenizer)

train_iter = DataLoader(train_data).__iter__()
next_item = None

# model
tokenizer = MoleculeProteinTokenizer()
model = Model(8, 768, 12, 4, 0.1, 'gelu', batch_first, True)
model = DistributedDataParallel(model)
criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=tokenizer.pad_token)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
                        next_item = train_iter.__next__()
                    except StopIteration:
                        train_iter = DataLoader(train_data).__iter__()
                        continue
                send_batch.append(next_item)
                batch_size += 1
                max_length = max(max_length, len(next_item))
                if (batch_size * max_length > args.token_per_batch):
                    next_item = send_batch.pop(-1)
                    break
            send_batch = pad_sequence(send_batch, batch_first=batch_first,
                    padding_value=tokenizer.pad_token).to(torch.int)

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
        batch = torch.zeros(args.token_per_batch, dtype=torch.int)
        dist.recv(batch, src=main_rank, tag=0)
        shape = torch.zeros(2, dtype=torch.int)
        dist.recv(shape, src=main_rank, tag=1)
        batch = batch[:shape[0]*shape[1]].reshape(batch).to(device)
    
    pred = model(batch[:-1])
    loss = criterion(pred, batch[1:])
    loss.backward()
    optimizer.step()
