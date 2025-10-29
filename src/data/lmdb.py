import os, pickle
import math
from typing import TypeVar, Literal
from collections.abc import Sized, Generator
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import get_worker_info
from ..utils.lmdb import load_lmdb, new_lmdb

T_co = TypeVar('T_co', covariant=True)

def data_len_to_blen(l: int|Sized):
    if isinstance(l, int):
        return ((l-1).bit_length()+7) // 8
    return data_len_to_blen(len(l))

class LMDBDataset(Dataset[bytes]):
    def __init__(self, lmdb_path: str, idx_to_key: Literal['byte', 'str']='byte', blen: int|None=None):
        self.path = lmdb_path
        match idx_to_key:
            case 'byte':
                if blen is None: blen = data_len_to_blen(self)
                self.idx_to_key = lambda idx: int(idx).to_bytes(blen, byteorder='big')
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

class PickleLMDBDataset(LMDBDataset):
    def __getitem__(self, idx: int):
        return pickle.loads(super().__getitem__(idx))

class StringLMDBDataset(LMDBDataset):
    def __getitem__(self, idx: int) -> str:
        return super().__getitem__(idx).decode('ascii')

class IntLMDBDataset(LMDBDataset):
    def __getitem__(self, idx: int) -> int:
        return int.from_bytes(super().__getitem__(idx), byteorder='big')

def npy_to_lmdb(npy_path: str):
    assert npy_path.endswith('.npy')
    out_path = npy_path.replace('.npy', '.lmdb')
    assert not os.path.exists(out_path)

    idxs = np.load(npy_path)
    idxs = np.sort(idxs)
    N = len(idxs)
    key_blen = ((N-1).bit_length()+7) // 8
    value_blen = (int(np.max(idxs)).bit_length()+7) // 8

    env, txn = new_lmdb(out_path)
    for i, idx in enumerate(tqdm(idxs)):
        key = i.to_bytes(key_blen)
        value = int(idx).to_bytes(value_blen)
        txn.put(key, value)
    txn.commit()
    env.close()

class IterLMDBSubset(IterableDataset[bytes]):
    def __init__(self, smi_path: str, index_path: str):
        self.smi_path = smi_path
        self.index_path = index_path
    
    def __iter__(self) -> Generator[bytes, None, None]:
        
        smi_env, smi_txn = load_lmdb(self.smi_path, readahead=True)
        smi_cursor = smi_txn.cursor()
        
        idx_env, idx_txn = load_lmdb(self.index_path, readahead=True)
        idx_cursor = idx_txn.cursor()
        idx_blen = data_len_to_blen(idx_env.stat()['entries'])

        # get start idx
        size = idx_env.stat()['entries']
        worker_info = get_worker_info()
        if worker_info is not None:
            per_worker = int(math.ceil(size/worker_info.num_workers))
            min_ = per_worker * worker_info.id
            sup = min(per_worker*(worker_info.id+1), size)
        else:
            min_, sup = 0, size
        
        # move idx_cursor
        idx_min_key = min_.to_bytes(idx_blen)
        idx_sup_key = sup.to_bytes(idx_blen)
        idx_cursor.set_key(idx_min_key)
        
        # move smi_cursor
        smi_min_key = idx_cursor.value()
        smi_cursor.set_key(smi_min_key)

        # iterate
        next_smi_key = smi_min_key
        for key, value in smi_cursor.iternext():
            if key == next_smi_key:
                yield value
                idx_cursor.next()
                idx_key, next_smi_key = idx_cursor.item()
                if idx_key == idx_sup_key: break
    def __len__(self):
        idx_env, idx_txn = load_lmdb(self.index_path)
        return idx_env.stat()['entries']


class StringIterLMDBSubset(IterLMDBSubset):
    def __iter__(self) -> Generator[str, None, None]:
        for value in super().__iter__():
            yield value.decode('ascii')
