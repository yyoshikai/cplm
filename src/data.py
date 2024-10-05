import pickle
from logging import getLogger

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import lmdb

from .tokenizer import MoleculeProteinTokenizer

class LMDBDataset(Dataset):
    logger = getLogger(f"{__module__}.{__qualname__}")

    def __init__(self, lmdb_path, key_is_indexed=False, keep_env=True, keep_txn=True):
        self.lmdb_path = lmdb_path
        self.env = self.txn = None
        if keep_env:
            self.env = lmdb.open(lmdb_path, subdir=False, readonly=True,
                lock=False, readahead=False, meminit=False, max_readers=256)
            if keep_txn:
                self.txn = self.env.begin()
        self.key_is_indexed = key_is_indexed
        self._lazy_keys = None
    
    def __getitem__(self, idx):
        env = self.env or lmdb.open(self.lmdb_path, subdir=False, readonly=True,
                lock=False, readahead=False, meminit=False, max_readers=256)
        txn = self.txn or env.begin()
        data = pickle.loads(txn.get(self.key(idx)))
    
    def key(self, idx):
        if self.key_is_indexed:
            return str(idx).encode('ascii')
        else:
            if self._lazy_keys is None:
                self.logger.info("Getting all key list...")
                self._lazy_keys = list(self.txn.cursor().iternext(values=False))
                self.logger.info("Done.")
            return self._lazy_keys[idx]

    def __len__(self):
        env = self.env or lmdb.open(self.lmdb_path, subdir=False, readonly=True,
                lock=False, readahead=False, meminit=False, max_readers=256)
        return env.stat()['entries']

class RepeatDataset(Dataset):
    def __init__(self, net_dataset, n_repeat):
        self.net_dataset = net_dataset
        self.net_size = len(net_dataset)
        self.n_repeat = n_repeat

    def __getitem__(self, idx):
        if (idx >= self.net_size*self.n_repeat):
            raise IndexError("Dataset index out of range")
        return self.net_dataset[idx%self.net_size]

    def __len__(self):
        return self.net_size * self.n_repeat
    
class SliceDataset(Dataset):
    def __init__(self, net_dataset, step, start):
        self.net_dataset = net_dataset
        net_size = len(self.net_dataset)
        self.size = net_size // step + (1 if net_size % step > start else 0)
        self.step = step
        self.start = start
    
    def __getitem__(self, idx):
        return self.net_dataset[idx//self.step+self.start]

    def __len__(self):
        return self.size

class MoleculeDataset(Dataset):
    def __init__(self, lmdb_path, n_conformer, tokenizer: MoleculeProteinTokenizer, seed=0):
        self.net_dataset = LMDBDataset(lmdb_path, key_is_indexed=True)
        self.tokenizer = tokenizer
        self.n_conformer = n_conformer
        self.rng = np.random.default_rng(seed)
    
    def __getitem__(self, idx):
        mol_idx, conformer_idx = divmod(idx, self.n_conformer)
        data = self.net_dataset[mol_idx]
        coord = data['coordinates'][conformer_idx]

        tokens = self.tokenizer.tokenize_smi(data['smi'])+self.tokenizer.tokenize_coord(coord)
        return torch.tensor(tokens, dtype=torch.long)
    
    def __len__(self):
        return len(self.net_dataset) * self.n_conformer

class ProteinDataset(Dataset):
    def __init__(self, lmdb_path, tokenizer: MoleculeProteinTokenizer):
        self.net_dataset = LMDBDataset(lmdb_path, key_is_indexed=True)
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx):
        data = self.net_dataset[idx]
        tokens = self.tokenizer.tokenize_protein(data['atoms'], data['coordinates'][0])
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.net_dataset)
