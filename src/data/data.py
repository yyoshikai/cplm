import pickle
from functools import lru_cache
from typing import TypeVar
from collections.abc import Callable, Sized

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from ..utils.lmdb import load_lmdb

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)

class WrapDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset[T]):
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
    
def data_len_to_blen(l: int|Sized):
    if isinstance(l, int):
        return ((l-1).bit_length()+7) // 8
    return data_len_to_blen(len(l))

from typing import Literal
class LMDBDataset(Dataset[T_co]):
    def __init__(self, lmdb_path: str, idx_to_key: Literal['byte', 'str']='byte'):
        self.path = lmdb_path
        match idx_to_key:
            case 'byte':
                blen = data_len_to_blen(self)
                self.idx_to_key = lambda idx: int(idx).to_bytes(blen)
            case 'str':
                self.idx_to_key = lambda idx: str(idx).encode('ascii')
            case _:
                raise ValueError(f"Unsupported {idx_to_key=}")

    def __getitem__(self, idx: int) -> bytes:
        env, txn = load_lmdb(self.path)
        item = txn.get(self.idx_to_key(idx))
        if item is None:
            raise ValueError(f"Key not found: {idx}, {self.idx_to_key(idx)}, {self.path}")
        return item

    def __len__(self):
        env, txn = load_lmdb(self.path)
        return env.stat()['entries']

class PickleLMDBDataset(LMDBDataset[T_co]):
    def __getitem__(self, idx: int) -> T_co:
        return pickle.loads(super().__getitem__(idx))

class StringLMDBDataset(LMDBDataset[str]):
    def __getitem__(self, idx: int) -> str:
        return super().__getitem__(idx).decode('ascii')

class IntLMDBDataset(LMDBDataset[int]):
    def __getitem__(self, idx: int) -> int:
        return int.from_bytes(super().__getitem__(idx))

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

class TensorDataset(Dataset[Tensor]):
    def __init__(self, tensor: Tensor|np.ndarray) -> None:
        self.tensor = torch.tensor(tensor)
    
    def __getitem__(self, idx: int):
        return self.tensor[idx]

    def __len__(self):
        return self.tensor.size(0)
    
