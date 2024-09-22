import sys, os
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from src.data import ChemDataset
from src.tokenizer import MoleculeProteinTokenizer
from src.model import Model

parser = argparse.ArgumentParser()
parser.add_argument("--data")
parser.add_argument("--test", action='store_true')
parser.add_argument("--token_size", type=int, default=25000)
parser.add_argument("--max-step", type=int, default=1000000)
parser.add_argument("--lr", type=float, default=1e-5)
args = parser.parse_args()

# environments
dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
rank = dist.get_rank()
size = dist.get_world_size()
device = torch.device('cuda', index=rank % torch.cuda.device_count()) \
    if torch.cuda.is_available() else torch.device('cpu')    

# data
train_data = ChemDataset('valid' if args.test else 'train')
valid_data = ChemDataset('valid')
# TODO: data_parallelへの対応, mol:proteinを何らかの割合で混ぜる, 
train_iter = DataLoader(train_data).__iter__()

# model
tokenizer = MoleculeProteinTokenizer()
model = Model(8, 768, 4, 0.1, 'gelu', True)
model = DistributedDataParallel(model)
criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=tokenizer.pad_token)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

for step in range(args.max_step):

    data = train_iter.__next__()
    
    pred = model(data[:, :-1])
    loss = criterion(pred, data[:, 1:])
    loss.backward()
    optimizer.step()

    



















