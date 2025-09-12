import itertools as itr
from bisect import bisect_right
from collections.abc import Callable, Iterable, Generator
from logging import getLogger
from typing import Optional, TypeVar
import torch
from torch import Tensor

import torch.distributed as dist
from ..utils import reveal_data

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

def dist_send_tensor(tensor: Tensor, dst: int):
    dist.send_object_list([tensor.shape, tensor.dtype], dst=dst)
    dist.send(tensor, dst=dst)
def dist_recv_tensor(src: int, recv_device: torch.device) -> Tensor:
    info = [None, None]
    dist.recv_object_list(info, src=src)
    shape, dtype = info
    tensor = torch.zeros(shape, dtype=dtype, device=recv_device)
    dist.recv(tensor, src=src)
    return tensor

def batched(iterable: Iterable[T_co], n: int) -> Generator[T_co, None, None]: # Same as itr.batched in python >= 3.12
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(itr.islice(iterator, n)):
        yield batch

T_in = TypeVar('T_in')
T_out = TypeVar('T_out')
class StringCollateIterator(Iterable[list[T_in]]):
    logger = getLogger(f"{__module__}.{__qualname__}")
    data_logger = getLogger(f"dexs.{__module__}.{__qualname__}")
    def __init__(self, loader: Iterable[T_in], n_batch: int, gpu_size: float, 
            gpuuse_getter: Callable[[int, int], float], 
            length_getter: Callable[[T_in], int]):

        self.loader = loader
        self.n_batch = n_batch
        self.gpu_size = gpu_size
        self.gpuuse_getter = gpuuse_getter
        self.length_getter = length_getter

    def batch_data_list(self, data_list: list[T_in]) -> tuple[list[list[T_in]], bool]:
        datas = []
        for _ in range(self.n_batch):
            batch_size = self.get_batch_size(self.length_getter(data_list[0]))
            datas.append(data_list[:batch_size])
            data_list = data_list[batch_size:]
            if len(data_list) == 0: break
        return datas, len(data_list) == 0

    def collates(self, data_list: list[T_in]) -> list[T_out]:
        data_lists, can_be_batched = self.batch_data_list(data_list)
        assert can_be_batched
        return data_lists

    def __iter__(self) -> Generator[Tensor, None, None]:
        data_list = []
        loader_iter = self.loader.__iter__()
        next_item = None
        n_large_item = n_item = 0
        while True:
            # get next item
            if next_item is None:
                try:
                    next_item = loader_iter.__next__()
                except StopIteration:
                    if len(data_list) > 0:
                        yield from self.collates(data_list)
                    raise StopIteration
                n_item += 1
            
            # check maximum size
            next_length = self.length_getter(next_item)
            if self.gpuuse_getter(1, next_length) > self.gpu_size:
                n_large_item += 1
                self.logger.warning(f"Item was too large even for 1 item per batch({next_length}), and not used ({n_large_item}/{n_item}).")
                next_item = None
                continue

            # insert size
            i = bisect_right(data_list, -next_length, key=lambda x: -self.length_getter(x))
            next_data_list = data_list[:i]+[next_item]+data_list[i:] # most slow

            # check more data can be added
            _, can_be_batched = self.batch_data_list(next_data_list)
            if can_be_batched:
                data_list = next_data_list
                next_item = None
            else:
                yield from self.collates(data_list)
                data_list = []

    def get_batch_size(self, length: int):
        return solve_increasing_fn_left(lambda bsz: self.gpuuse_getter(bsz, length)-self.gpu_size, 16)



class DDPStringCollateLoader(Iterable[T_out]):
    logger = getLogger(f"{__module__}.{__qualname__}")
    data_logger = getLogger(f"dexs.{__module__}.{__qualname__}")
    def __init__(self, loader: Optional[Iterable[T_in]], collator: Callable[[T_in], T_out]|None, gpuuse_getter: Callable[[int, int], T_out]|None, length_getter: Callable[[T_in], int]|None, gpu_size: float, device: torch.device, main_rank: int=0):
        self.size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.main_rank = main_rank
        self.device = device
        self.collator = collator
        self.is_main = self.rank == self.main_rank

        if self.is_main:
            self.batch_iterator = StringCollateIterator(loader, self.size, gpu_size, gpuuse_getter, length_getter)
        else:
            self.batch_iterator = itr.repeat(None)

    def scatter_batches(self, batches: Optional[list[list[T_in]]]) -> Optional[T_out]:
        if self.is_main:
            assert len(batches) <= self.size

            # Send batch info
            batch_infos = [torch.tensor(0 if rank < len(batches) else 1, device=self.device) 
                    for rank in range(self.size)]
            dist.scatter(torch.tensor(0, device=self.device), batch_infos, src=self.main_rank)

            # Send batch
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
            batch_info = torch.tensor(0, device=self.device)
            dist.scatter(batch_info, src=self.main_rank)

            # Receive batch
            if batch_info.item() == 0:
                ## ここもspecialized
                token = dist_recv_tensor(self.main_rank, self.device)
                weight = dist_recv_tensor(self.main_rank, self.device)
                this_data = token, weight
            else:
                this_data = None
            return this_data


    def __iter__(self) -> Generator[Optional[T_out], None, None]:
        for batches in batched(self.batch_iterator, self.size):
            # Sync StopIteration
            stop_iteration = torch.tensor(False, device=self.device)
            stop_iterations = [stop_iteration for rank in range(self.size)] if self.is_main else None
            dist.scatter(stop_iteration, stop_iterations, src=self.main_rank)
            if stop_iteration.item(): break

            # Send & yield batch
            yield self.scatter_batches(batches)

        # Sync StopIteration from main_rank
        if self.is_main:
            stop_iteration = torch.tensor(True, device=self.device)
            dist.scatter(stop_iteration, [stop_iteration for rank in range(self.size)], src=self.main_rank)
