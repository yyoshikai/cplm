import random, struct, logging
from functools import partial
from time import time
import numpy as np
import torch

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

"""
class logtime:
    def __init__(self, logger, prefix):
        pass
    def __enter__(self):
        pass
    def __exit__(self, *excinfo):
        pass

"""

LOGTIME = True

def set_logtime(logtime: bool):
    global LOGTIME
    LOGTIME = logtime

class logtime:
    def __init__(self, logger: logging.Logger, prefix: str=''):
        self.logger = logger
        self.prefix = prefix
    def __enter__(self):
        if LOGTIME:
            self.start = time()
    def __exit__(self, exc_type, exc_value, traceback):
        if LOGTIME:
            self.logger.debug(f"{self.prefix} {time()-self.start:.4f}") 