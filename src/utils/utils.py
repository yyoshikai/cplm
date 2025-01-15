import random, struct, logging
from functools import partial
from bisect import bisect_right
from time import time
import numpy as np
import torch
from .logger import get_logger

class RandomState:
    def __init__(self, seed: int=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def state_dict(self):
        state_dict = {
            'random': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all()
        }
        return state_dict
    def load_state_dict(self, state_dict: dict):
        random.setstate(state_dict['random'])
        np.random.set_state(state_dict['numpy'])
        torch.set_rng_state(state_dict['torch'])
        torch.cuda.set_rng_state_all(state_dict['cuda'])

def load_gninatypes(path, struct_fmt='fffi'):
    struct_len = struct.calcsize(struct_fmt)
    struct_unpack = struct.Struct(struct_fmt).unpack_from
    with open(path,'rb') as tfile:
        data = [struct_unpack(chunk) for chunk in iter(partial(tfile.read, struct_len), b'')]
    return data

# https://stackoverflow.com/questions/39042214/how-can-i-slice-each-element-of-a-numpy-array-of-strings
def slice_str(x: np.ndarray, end: int):
    b = x.view((str,1)).reshape(len(x),-1)[:, :end]
    return np.fromstring(b.tostring(),dtype=(str,end))

LOGTIME = False
def set_logtime(logtime: bool):
    global LOGTIME
    LOGTIME = logtime

class logtime:
    def __init__(self, logger: logging.Logger, prefix: str='', level=logging.DEBUG, thres: float=0):
        self.logger = logger
        self.prefix = prefix
        self.level = level
        self.thres = thres
    def __enter__(self):
        if LOGTIME:
            self.start = time()
    def __exit__(self, exc_type, exc_value, traceback):
        if LOGTIME:
            elapse = time() - self.start
            if elapse >= self.thres:
                self.logger.log(self.level, f"{self.prefix} {elapse:.4f}") 

class rectime: 
    def __init__(self):
        pass
    def __enter__(self):
        self.start = time()
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.time = time() - self.start

def remove_module(state: dict):
    new_state = {}
    for key, value in state.items():
        assert key[:7] == 'module.'
        new_state[key[7:]] = value
    return new_state



class CompressedArray:
    logger = get_logger(f'{__module__}.{__qualname__}')
    def __init__(self, array: np.ndarray):
        self.logger.info("Compressing array...")
        self.points = np.where(array[1:] != array[:-1])[0]+1
        self.values = np.concatenate([array[[0]], array[self.points]], axis=0)
        self.size = len(array)
        self.logger.info("compressed.")

    def __getitem__(self, idx: int):
        if idx >= self.size or idx < -self.size:
            raise IndexError(f'CompressedArray index out of range({idx}/{self.size})')
        if idx < 0:
            idx += self.size
        bin = bisect_right(self.points, idx)
        return self.values[bin]

    def __len__(self):
        return self.size
