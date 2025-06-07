from functools import lru_cache
from typing import TypeVar
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)

class WrapDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset[T]):
        self.dataset = dataset
    def __getitem__(self, idx: int) -> T_co:
        raise NotImplementedError
    def __len__(self):
        return len(self.dataset)

class ApplyDataset(WrapDataset[T_co]):
    def __init__(self, dataset: Dataset[T], func: Callable[[T], T_co]):
        super().__init__(dataset)
        self.func = func
    def __getitem__(self, idx: int):
        return self.func(self.dataset[idx])
    
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

class ConstantDataset(Dataset[T_co]):
    def __init__(self, value: T_co, size: int):
        self.size = size
        self.value = value
    def __getitem__(self, idx: int):
        return self.value
    def __len__(self):
        return self.size

class TensorDataset(Dataset[Tensor]):
    def __init__(self, tensor: Tensor|np.ndarray) -> None:
        self.tensor = torch.tensor(tensor)
    
    def __getitem__(self, idx: int):
        return self.tensor[idx]

    def __len__(self):
        return self.tensor.size(0)
    
