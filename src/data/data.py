import pickle
from functools import lru_cache
from logging import getLogger
from typing import Generic, TypeVar
from collections.abc import Callable

import numpy as np
from torch.utils.data import Dataset
import lmdb

from ..utils import logtime
from ..utils.lmdb import load_lmdb

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)

class WrapDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
    def __getitem__(self, idx: int) -> T_co:
        raise NotImplementedError
    def __len__(self):
        return len(self.dataset)
    def __getattr__(self, name):
        return self.dataset.__getattribute__(name)

class ApplyDataset(WrapDataset[T_co]):
    def __init__(self, dataset: Dataset[T], func: Callable[[T], T_co]):
        super().__init__(dataset)
        self.func = func
    def __getitem__(self, idx: int):
        return self.func(self.dataset[idx])


class LMDB(Dataset[T_co]):
    logger = getLogger(f"{__module__}.{__qualname__}")

    def __init__(self, path, keep_env=False, keep_txn=False):
        self.path = path
        self.keep_env = keep_env
        self.keep_txn = keep_txn
        
        self._lazy_env = self._lazy_txn = None
        self._lazy_keys = None

    def __getitem__(self, key) -> T_co:
        with logtime(self.logger, f"({self.path})[{key.decode('ascii')}]:"): #?
            return pickle.loads(self.txn().get(key))

    def env(self) -> lmdb.Environment:
        if self._lazy_env is not None:
            return self._lazy_env
        env = lmdb.open(self.path, subdir=False, readonly=True,
                lock=False, readahead=False, meminit=False, max_readers=256)
        if self.keep_env:
            self._lazy_env = env
        return env

    def txn(self) -> lmdb.Transaction:
        if self._lazy_txn is not None:
            return self._lazy_txn
        txn = self.env().begin()
        if self.keep_txn:
            self._lazy_txn = txn
        return txn
    
    def keys(self):
        return list(self.txn().cursor().iternext(values=False))

    def __len__(self):
        return self.env().stat()['entries']

class LMDBDataset(Dataset[T_co]):
    logger = getLogger(f"{__module__}.{__qualname__}")

    def __init__(self, lmdb_path, key_is_indexed=False, keep_env=False, keep_txn=False):
        self.lmdb = LMDB[T_co](lmdb_path, keep_env, keep_txn)
        self.key_is_indexed = key_is_indexed
        self._lazy_keys = None

    def __getitem__(self, idx):
        return self.lmdb[self.key(idx)]
    
    def key(self, idx):
        if self.key_is_indexed:
            return str(idx).encode('ascii')
        else:
            if self._lazy_keys is None:
                self.logger.info("Getting all key list...")
                self._lazy_keys = self.lmdb.keys()
                self.logger.info("Done.")
            return self._lazy_keys[idx]

    def __len__(self):
        return len(self.lmdb)
    
class AsciiLMDBDataset(Dataset[str]):
    logger = getLogger(f"{__module__}.{__qualname__}")

    def __init__(self, lmdb_path, key_is_indexed=False, keep_env=False, keep_txn=False):
        self.lmdb_path = lmdb_path

    def __getitem__(self, idx):
        env, txn = load_lmdb(self.lmdb_path)
        return txn.get(str(idx).encode('ascii')).decode('ascii')
    
    def __len__(self):
        env, _ = load_lmdb(self.lmdb_path)
        return env.stat()['entries']
    

# Indexing
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
        self.size = net_size // step + (1 if (net_size % step) > start else 0)
        self.step = step
        self.start = start
    
    def __getitem__(self, idx):
        return self.net_dataset[idx*self.step+self.start]

    def __len__(self):
        return self.size

class SampleDataset(Dataset):
    def __init__(self, dataset: Dataset, size: int, seed: int=0):
        rstate = np.random.RandomState(seed)
        assert size <= len(dataset)
        self.idxs = rstate.choice(len(dataset), size=size, replace=False)
        self.dataset = dataset

    def __getitem__(self, idx: int):
        return self.dataset[self.idxs[idx]]
    
    def __len__(self):
        return len(self.idxs)

class CacheDataset(WrapDataset):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)
        self.cache_idx = None
        self.cache_item = None
    
    def __getitem__(self, idx: int):
        if idx != self.cache_idx:
            self.cache_idx = idx
            self.cache_item = self.dataset[idx]
        return self.cache_item

class LRUCacheDataset(CacheDataset):
    def __init__(self, dataset: Dataset, maxsize: int=1):
        super().__init__(dataset)
        self._getitem_cached = lru_cache(maxsize=maxsize, typed=True)(self._getitem)

    def _getitem(self, idx: int):
        return self.dataset[idx]
    
    def __getitem__(self, idx: int):
        return self._getitem_cached(idx)

class KeyDataset(WrapDataset):
    def __init__(self, dataset: CacheDataset, key):
        if not isinstance(dataset, CacheDataset):
            raise ValueError(f"KeyDataset not on CacheDataset({type(dataset)}) is slow.")
        super().__init__(dataset)
        self.key = key
    def __getitem__(self, idx):
        return self.dataset[idx][self.key]

def untuple_dataset(dataset: Dataset, size: int):
    if not isinstance(dataset, CacheDataset):
        dataset = CacheDataset(dataset)
    return tuple(KeyDataset(dataset, i) for i in range(size))
    
class IndexDataset(WrapDataset[tuple[int,T_co]]):
    def __init__(self, dataset: Dataset[T_co]):
        super().__init__(dataset)
        self.dataset = dataset

    def __getitem__(self, idx: int):
        return idx, self.dataset[idx]

def index_dataset(dataset: Dataset[T]) -> tuple[Dataset[int], Dataset[T]]:
    dataset = IndexDataset(dataset)
    return untuple_dataset(dataset, 2)

class CoordTransform:
    def __init__(self, seed:int=0, normalize_coord=False, random_rotate=False, coord_noise_std=0.0):
        self.rng = np.random.default_rng(seed)
        self.normalize_coord = normalize_coord
        self.random_rotate = random_rotate
        self.coord_noise_std = coord_noise_std
    
    def __call__(self, coords: np.ndarray) -> np.ndarray:
        if coords.size == 0:
            return coords
        if self.normalize_coord:
            coords = coords - np.mean(coords, axis=0, keepdims=True)
        if self.random_rotate:
            matrix = get_random_rotation_matrix(self.rng)
            coords = np.matmul(coords, matrix)
        if self.coord_noise_std > 0:
            noise = self.rng.normal(size=3, scale=self.coord_noise_std)   
            coords += noise
        return coords

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
