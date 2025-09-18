import os, math
import itertools as itr
from bisect import bisect_right
from collections.abc import Callable, Iterable, Generator
from logging import getLogger
from typing import Optional, TypeVar
import torch
from torch import Tensor
from torch.utils.data import DataLoader

import torch.distributed as dist
from ..utils.ddp import dist_send_tensor, dist_recv_tensor
from ..utils.time import TimerTqdm

T = TypeVar('T')
T1 = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)

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

def batched(iterable: Iterable[T_co], n: int) -> Generator[T_co, None, None]: # Same as itr.batched in python >= 3.12
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(itr.islice(iterator, n)):
        yield batch

T_in = TypeVar('T_in')
T_out = TypeVar('T_out')

class InfiniteLoader(Iterable[T_in]):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.epoch = None

        # If True, setting os.environ will have no effect.
        assert self.loader.persistent_workers == False

    def __iter__(self):
        self.epoch = 0
        while True:
            os.environ['EPOCH'] = str(self.epoch)
            yield from self.loader
            self.logger.info(f"Epoch #{self.epoch} finished.")
            self.epoch += 1

class StringCollateIterator(Iterable[list[T_in]]):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, loader: Iterable[T_in], n_batch: int, gpu_size: float, 
            gpuuse_getter: Callable[[int, int], float], 
            length_getter: Callable[[T_in], int],
            log_large_freq: int|float, 
            large_item_file: str|None=None, 
            timer: TimerTqdm|None=None):

        self.loader = loader
        self.n_batch = n_batch
        self.gpu_size = gpu_size
        self.gpuuse_getter = gpuuse_getter
        self.length_getter = length_getter
        self.timer = timer

        # logging
        self.large_item_file = large_item_file
        if large_item_file is not None:
            with open(self.large_item_file, 'w') as f:
                f.write(f"i_item,size\n")
        self.log_large_freq = log_large_freq

    def batch_data_list(self, data_list: list[T_in]) -> tuple[list[list[T_in]], bool]:
        datas = []
        for _ in range(self.n_batch):
            batch_size = self.get_batch_size(self.length_getter(data_list[0]))
            datas.append(data_list[:batch_size])
            data_list = data_list[batch_size:]
            if len(data_list) == 0: break
        return datas, len(data_list) == 0
    
    def start(self, name):
        if self.timer is not None: self.timer.start(f"sc_{name}")

    def collates(self, data_list: list[T_in]) -> list[T_out]:
        data_lists, can_be_batched = self.batch_data_list(data_list)
        assert can_be_batched
        return data_lists
    def __iter__(self) -> Generator[Tensor, None, None]:
        data_list = []
        loader_iter = self.loader.__iter__()
        next_item = None
        n_large_item = n_item = 0
        n_item_last_log = 0
        while True:
            # get next item
            self.start('get_next_item')
            if next_item is None:
                try:
                    next_item = loader_iter.__next__()
                except StopIteration:
                    self.start('output')
                    if len(data_list) > 0:
                        yield from self.collates(data_list)
                    break
                n_item += 1
            
            # check maximum size
            self.start('check_max_size')
            next_length = self.length_getter(next_item)
            if self.gpuuse_getter(1, next_length) > self.gpu_size:
                if self.large_item_file is not None:
                    with open(self.large_item_file, 'a') as f:
                        f.write(f"{n_item-1},{next_length}")
                
                # Log n_large_item
                if n_item - n_item_last_log >= self.log_large_freq:
                    self.logger.info(f"{n_large_item}/{n_item} was too large.")
                    n_item_last_log = n_item
                n_large_item += 1
                next_item = None
                continue
            
            
            # insert size
            self.start('insert_size')
            i = bisect_right(data_list, -next_length, key=lambda x: -self.length_getter(x))
            next_data_list = data_list[:i]+[next_item]+data_list[i:] # most slow

            # check more data can be added
            self.start('check_add')
            _, can_be_batched = self.batch_data_list(next_data_list)
            if can_be_batched:
                data_list = next_data_list
                next_item = None
            else:
                self.start('output')
                yield from self.collates(data_list)
                data_list = []
        if self.log_large_freq < math.inf:
            self.logger.info(f"{n_large_item}/{n_item} was too large.")


    def get_batch_size(self, length: int):
        return solve_increasing_fn_left(lambda bsz: self.gpuuse_getter(bsz, length)-self.gpu_size, 16)



class DDPStringCollateLoader(Iterable[T_out]):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, loader: Optional[Iterable[T_in]], collator: Callable[[T_in], T_out], gpuuse_getter: Callable[[int, int], float], length_getter: Callable[[T_in], int], gpu_size: float, device: torch.device, log_large_freq: int|float, main_rank: int=0, 
    large_item_file: str|None=None, timer: TimerTqdm|None=None):
        self.size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.main_rank = main_rank
        self.device = device
        self.collator = collator
        self.is_main = self.rank == self.main_rank
        self.timer = timer

        if self.is_main:
            self.batch_iterator = StringCollateIterator(loader, self.size, 
                    gpu_size, gpuuse_getter, length_getter, log_large_freq, large_item_file, timer)
        else:
            self.batch_iterator = itr.repeat(None)

    def scatter_batches(self, batches: Optional[list[list[T_in]]]) -> Optional[T_out]:
        if self.is_main:
            assert len(batches) <= self.size

            # Send batch info
            self.start('send_batch_info')
            batch_infos = [torch.tensor(0 if rank < len(batches) else 1, device=self.device) 
                    for rank in range(self.size)]
            dist.scatter(torch.tensor(0, device=self.device), batch_infos, src=self.main_rank)

            # Send batch
            self.start('send_batch')
            this_data = None
            for dst_rank, dst_data in enumerate(batches):
                data = self.collator(dst_data)
                token, weight = data
                token = token.to(self.device)
                weight = weight.to(self.device)
                ## ここはspecialized
                if dst_rank == self.rank:
                    this_data = token, weight
                else:
                    dist_send_tensor(token, dst_rank)
                    dist_send_tensor(weight, dst_rank)
            return this_data
        else:
            # Receive batch info
            self.start('recv_batch_info')
            batch_info = torch.tensor(0, device=self.device)
            dist.scatter(batch_info, src=self.main_rank)

            # Receive batch
            if batch_info.item() == 0:
                ## ここもspecialized
                self.start('recv_batch')
                token = dist_recv_tensor(self.main_rank, self.device)
                weight = dist_recv_tensor(self.main_rank, self.device)
                this_data = token, weight
            else:
                this_data = None
            return this_data

    def start(self, name: str):
        if self.timer is not None: self.timer.start(f"ddp_{name}")

    def __iter__(self) -> Generator[Optional[T_out], None, None]:
        self.start('iter_batches')
        for batches in batched(self.batch_iterator, self.size):
            # Sync StopIteration
            self.start('sync_stop')
            stop_iteration = torch.tensor(False, device=self.device)
            stop_iterations = [stop_iteration for rank in range(self.size)] if self.is_main else None
            dist.scatter(stop_iteration, stop_iterations, src=self.main_rank)
            if stop_iteration.item(): break

            # Send & yield batch

            yield self.scatter_batches(batches)
            self.start('iter_batches')

        # Sync StopIteration from main_rank
        if self.is_main:
            stop_iteration = torch.tensor(True, device=self.device)
            dist.scatter(stop_iteration, [stop_iteration for rank in range(self.size)], src=self.main_rank)
