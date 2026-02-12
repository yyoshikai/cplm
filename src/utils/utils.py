import sys, os, struct, traceback, warnings, subprocess, math, csv
from functools import partial
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from logging import getLogger
from typing import Any, Callable, Literal
import numpy as np
import pandas as pd
try:    
    import torch
except ImportError:
    torch = None

# others
def load_gninatypes(path, struct_fmt='fffi'):
    struct_len = struct.calcsize(struct_fmt)
    struct_unpack = struct.Struct(struct_fmt).unpack_from
    with open(path,'rb') as tfile:
        data = [struct_unpack(chunk) for chunk in iter(partial(tfile.read, struct_len), b'')]
    return data

def slice_str(x: np.ndarray, end: int):
    return np.array([i[:end] for i in x], dtype=(str, end))

def remove_module(state: dict):
    new_state = {}
    for key, value in state.items():
        key = key.removeprefix('module.')
        new_state[key] = value
    return new_state

def setdefault(obj: object, name: str, value: Any):
    if not hasattr(obj, name):
        setattr(obj, name, value)

class CompressedArray:
    logger = getLogger(f'{__module__}.{__qualname__}')
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

module_abbrevs = {'numpy': 'np', 'pandas': 'pd'}
def reveal_data(data, max_iterable_size: int=20, max_str_size: int=160) -> str:
    
    t = type(data)
    module_name = t.__module__
    tname = t.__name__
    if module_name == 'builtins':
        typename = tname
    else:
        module_name = t.__module__
        module_name = module_abbrevs.get(module_name, module_name)
        typename = f"{module_name}.{tname}"
    
    if isinstance(data, str):
        if len(data) > max_str_size:
            data = data[:max_str_size-20] + f"...({len(data)-(max_str_size-10)} letters)..."+data[-10:]
        return data
    elif isinstance(data, (dict, list, tuple, set)):

        if isinstance(data, dict):
            keys = list(data.keys())
            items = list(data.values())
        else:
            items = list(data)
        
        type2bracket = {list: '[]', tuple: '()', set: '{}', dict: '{}'}
        for type_, bracket in type2bracket.items():
            if type(data) == type_:
                start, end = bracket[0], bracket[1]
                break
            if isinstance(data, type_):
                start, end = f"{typename}({bracket[0]}", f"{bracket[1]})"
                break

        n_omit = 0
        n_total = len(items)
        if n_total > max_iterable_size:
            n_omit = len(items) - max_iterable_size
            items = items[:max_iterable_size-1]+[items[-1]]
            if isinstance(data, dict):
                keys = keys[:max_iterable_size-1]+[keys[-1]]
        
        item_strs = [reveal_data(item, max_iterable_size) for item in items]
        if isinstance(data, dict):
            item_strs = [f"{key}: {item_str}" 
                for key, item_str in zip(keys, item_strs)]
        if n_omit > 0:
            item_strs.insert(max_iterable_size-2, f'...({n_total} items in total)...')
        
        total_len = sum(len(item_str) for item_str in item_strs)

        if total_len > max_str_size:
            output = start
            for i, item_str in enumerate(item_strs):
                for line in item_str.split('\n'):
                    output += "\n    "+line
                if i != len(item_strs)-1:
                    output += ','
            output += f'\n{end}'
            return output
        else:
            return start + ', '.join(item_strs) + end
    elif isinstance(data, (int, float, type)):
        return str(data)
    elif isinstance(data, np.ndarray) or (torch is not None and isinstance(data, torch.Tensor)):
        items = data.ravel()
        if len(items) > 10:
            items = items[:8].tolist()+['...']+items[-2:].tolist()

        return f"{typename}(shape={tuple(data.shape)}, dtype={data.dtype}, data=[" \
            +', '.join([str(i) for i in items])+'])'
        
    else:
        data = reveal_data(repr(data), max_iterable_size, max_str_size)
        if data.startswith(tname):
            data = typename + data.removeprefix(tname)
        return data

def traceback_warning():
    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

        log = file if hasattr(file,'write') else sys.stderr
        traceback.print_stack(file=log)
        log.write(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = warn_with_traceback

# Git
def git_commit() -> bool:
    p = subprocess.run('git add . && git commit -m checkpoint_for_training', shell=True, capture_output=True)
    return p.returncode == 0 # 1=nothing to commit, working tree clean

def get_git_hash() -> str:
    p = subprocess.run('git rev-parse --short HEAD', shell=True, capture_output=True)
    return p.stdout.decode().strip()

def should_show(n: int, max_interval: int|float|None=None):
    if max_interval is None:
        max_interval = math.inf
    if n < max_interval:
        return (n&(n-1)) == 0 # True if n == 2**x
    else:
        return n % max_interval == 0

def _flatten_dict(d: Mapping|Sequence) -> dict:
    if isinstance(d, Sequence):
        return _flatten_dict({i: v for i, v in enumerate(d)})
    d_out = {}
    for key, value in d.items():
        if isinstance(value, Mapping) or \
                (isinstance(value, Sequence) and len(value) > 1): # dict, addict.Dict, OrderedDict; 長さ1のtensorをflattenし続けないように
            value = _flatten_dict(value)
            for ckey, cvalue in value.items():
                pkey = (key,)+ckey if isinstance(ckey, tuple) else (key, ckey)
                d_out[pkey] = cvalue
        else:
            d_out[key] = value
    return d_out

class IterateRecorder:
    logger = getLogger(f'{__module__}.{__qualname__}')
    def __init__(self, path: str, max_out_interval: int, cols: list[str]=[]):
        self.path = path
        self.flushed = False
        self.data = {col: [] for col in cols}
        self.data_size = 0
        self.step = 0
        self.max_out_interval = max_out_interval
        self.n_out_level = 1

    def record(self, **kwargs: dict[str, Any]):
        self.step += 1
        kwargs = _flatten_dict(kwargs)
        kwargs = {tuple(str(k) for k in key) if isinstance(key, tuple) else (str(key),): str(value) for key, value in kwargs.items()}
        # record data
        for col, li in self.data.items():
            li.append(kwargs.pop(col, ''))
        
        # add unknown cols
        if len(kwargs) > 0:
            new_cols = list(kwargs.keys())
            new_n_level = max(len(col) for col in new_cols)

            ## Modify DataFrame
            if self.flushed:
                with open(self.path) as f:
                    data = np.array(list(csv.reader(f)), dtype=object)
                n_row, n_col = data.shape
                if new_n_level > self.n_out_level:
                    data = np.insert(data, self.n_out_level, np.full((new_n_level-self.n_out_level, n_col), '', dtype=object))
                    self.n_out_level += new_n_level
                data = np.concatenate([
                    data, 
                    np.array(list(col)+['']*(self.n_out_level+n_row-len(list(col)))),
                ], axis=1)
                with open(self.path, 'w', newline='') as f:
                    csv.writer(f).writerows(data)

            ## Modify self.data
            for col in new_cols:
                self.data[col] = ['']*self.data_size+[kwargs[col]]
        self.data_size += 1

        # Flush
        if should_show(self.step, self.max_out_interval):
            self.flush()
        

    def flush(self):
        pdir = os.path.dirname(self.path)
        if not os.path.exists(pdir): os.makedirs(pdir,  exist_ok=True)
        if not self.flushed:
            self.n_out_level = max(len(col) for col in self.data)
            with open(self.path, 'w', newline='') as f:
                writer = csv.writer(f)
                for level in range(self.n_out_level):
                    writer.writerow([col[level] if level < len(col) else '' for col in self.data])
        with open(self.path, 'a', newline='') as f:
            csv.writer(f).writerows(zip(*list(self.data.values())))
        self.data = {key: [] for key in self.data.keys()}
        self.data_size = 0
        self.flushed = True


def solve_increasing_fn_left(func: Callable[[int], float], start_sup: int) -> int:

    min = 0
    sup = start_sup

    # get max
    while func(sup) <= 0:
        min = sup
        sup = sup*2

    # narrow down
    while sup - min > 1:
        v = (sup+min) // 2
        if func(v) <= 0:
            min = v
        else:
            sup = v
    return min
