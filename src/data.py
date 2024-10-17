import pickle
from logging import getLogger
import math

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
import lmdb

from .tokenizer import MoleculeProteinTokenizer

class LMDBDataset(Dataset):
    logger = getLogger(f"{__module__}.{__qualname__}")

    def __init__(self, lmdb_path, key_is_indexed=False, keep_env=False, keep_txn=False):
        self.lmdb_path = lmdb_path
        self.keep_env = keep_env
        self.keep_txn = keep_txn
        self.key_is_indexed = key_is_indexed
        
        self._lazy_env = self._lazy_txn = None
        self._lazy_keys = None
    
    def __getitem__(self, idx):
        return pickle.loads(self.txn().get(self.key(idx)))

    def env(self):
        if self._lazy_env is not None:
            return self._lazy_env
        env = lmdb.open(self.lmdb_path, subdir=False, readonly=True,
                lock=False, readahead=False, meminit=False, max_readers=256)
        if self.keep_env:
            self._lazy_env = env
        return env

    def txn(self):
        if self._lazy_txn is not None:
            return self._lazy_txn
        txn = self.env().begin()
        if self.keep_txn:
            self._lazy_txn = txn
        return txn
    
    def key(self, idx):
        if self.key_is_indexed:
            return str(idx).encode('ascii')
        else:
            if self._lazy_keys is None:
                self.logger.info("Getting all key list...")
                self._lazy_keys = list(self.txn().cursor().iternext(values=False))
                self.logger.info("Done.")
            return self._lazy_keys[idx]

    def __len__(self):
        return self.env().stat()['entries']

class CoordTransform:
    def __init__(self, seed:int=0, normalize_coord=False, random_rotate=False, coord_noise_std=0.0):
        self.rng = np.random.default_rng(seed)
        self.normalize_coord = normalize_coord
        self.random_rotate = random_rotate
        self.coord_noise_std = coord_noise_std
    
    def __call__(self, coords: np.ndarray) -> np.ndarray:
        if self.normalize_coord:
            coords = coords - np.mean(coords, axis=1, keepdims=True)
        if self.random_rotate:
            matrix = get_random_rotation_matrix(self.rng)
            coords = np.matmul(coords, matrix)
        if self.coord_noise_std > 0:
            noise = self.rng.normal(size=3, scale=self.coord_noise_std)   
            coords += noise
        return coords

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

def get_random_rotation_matrix(rng: np.random.Generator):
    # get axes
    axes = []
    while(len(axes) < 2):
        new_axis = rng.random(3)
        
        new_norm = np.sqrt(np.sum(new_axis**2))
        if (new_norm < 0.1 or 1 <= new_norm): continue
        new_axis = new_axis / new_norm
        if np.any([np.abs(np.sum(axis*new_axis)) >= 0.9 for axis in axes]):
            continue
        axes.append(new_axis)

    # get rotation matrix
    axis0, axis1b = axes
    axis1 = np.cross(axis0, axis1b)
    axis1 = axis1 / np.linalg.norm(axis1)
    axis2 = np.cross(axis0, axis1)
    axis2 = axis2 / np.linalg.norm(axis2)
    return np.array([axis0, axis1, axis2])

class MoleculeDataset(Dataset):
    def __init__(self, lmdb_path, n_conformer, tokenizer: MoleculeProteinTokenizer, 
                coord_transform: CoordTransform, seed=0, **kwargs):
        self.net_dataset = LMDBDataset(lmdb_path, key_is_indexed=True, **kwargs)
        self.tokenizer = tokenizer
        self.n_conformer = n_conformer
        self.coord_transform = coord_transform
        self.rng = np.random.default_rng(seed)
    
    def __getitem__(self, idx):
        mol_idx, conformer_idx = divmod(idx, self.n_conformer)
        data = self.net_dataset[mol_idx]
        smi = data['smi']
        coord = data['coordinates'][conformer_idx]
        coord = self.coord_transform(coord)
        tokens = self.tokenizer.tokenize_smi(smi)+self.tokenizer.tokenize_coord(coord)
        return torch.tensor(tokens, dtype=torch.long)
    
    def __len__(self):
        return len(self.net_dataset) * self.n_conformer

class ProteinDataset(Dataset):
    def __init__(self, lmdb_path, tokenizer: MoleculeProteinTokenizer, 
            coord_transform: CoordTransform, **kwargs):
        self.net_dataset = LMDBDataset(lmdb_path, key_is_indexed=True, **kwargs)
        self.tokenizer = tokenizer
        self.coord_transform = coord_transform

    def __getitem__(self, idx):
        data = self.net_dataset[idx]
        atoms = data['atoms']
        coords = data['coordinates'][0]
        coords = self.coord_transform(coords)

        tokens = self.tokenizer.tokenize_protein(atoms, coords)
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.net_dataset)
