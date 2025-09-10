from bisect import bisect_right
from collections.abc import Callable
from functools import partial
from logging import getLogger
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import torch.distributed as dist
from .data import IndexDataset
from .sampler import InfiniteRandomSampler
from ..model import Model, MambaModel2

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
def dist_recv_tensor(src: int, recv_device: torch.device):
    info = [None, None]
    dist.recv_object_list(info, src=src)
    shape, dtype = info
    tensor = torch.zeros(shape, dtype=dtype, device=recv_device)
    dist.recv(tensor, src=src)
    return tensor

class DDPStringCollateLoader:
    logger = getLogger(f"{__module__}.{__qualname__}")
    data_logger = getLogger(f"dexs.{__module__}.{__qualname__}")
    def __init__(self, dataset: Dataset[tuple[Tensor, Tensor]], model: Model|MambaModel2, num_workers:int, pin_memory: bool, prefetch_factor: int, 
            gpu_size: float, batch_first: bool, padding_value: int, bf16: bool, kernel: str,
            device: torch.device, padding_weight: float=0.0, main_rank: int=0, seed: int=None):
        self.size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.main_rank = main_rank
        self.device = device

        if self.rank == self.main_rank:

            # temp
            dataset = IndexDataset(dataset)
            generator = torch.Generator().manual_seed(seed) if seed is not None else None
            sampler = InfiniteRandomSampler(dataset, generator=generator)
            self.gpu_size = gpu_size
            self.loader = DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=num_workers, 
                pin_memory=pin_memory, persistent_workers=True, prefetch_factor=prefetch_factor)
            
            if isinstance(model, Model): self.get_gpuuse = partial(model.get_gpuuse, bf16=bf16, kernel=kernel)
            elif isinstance(model, MambaModel2): self.get_gpuuse = partial(model.get_gpuuse, bf16=bf16)
            else: raise ValueError(f"Unsupported model: {type(model)}")

            self.iter = self.loader.__iter__()
            self.next_item = None
            self.step = 0
            self.batch_first = batch_first
            self.padding_value = padding_value
            self.padding_weight = padding_weight
            
            self.i_item = 0

    def __next__(self) -> torch.Tensor:
        if self.rank == self.main_rank:
            data_list = []
            while True:
                # get next item
                if self.next_item is None:
                    index, self.next_item = self.iter.__next__()
                    if self.step < 5:
                        self.data_logger.debug(f"{self.i_item}: {index}")
                        self.i_item += 1
                
                # check maximum size
                next_size = len(self.next_item[0])
                if self.get_gpuuse(batch_size=1, length=next_size) > self.gpu_size:
                    self.logger.warning(f"Item was too large even for 1 item per batch({len(self.next_item)}), and not used.")
                    self.next_item = None
                    continue

                # insert size
                i = bisect_right(data_list, -len(self.next_item[0]), key=lambda x: -len(x))
                next_data_list = data_list[:i]+[self.next_item]+data_list[i:] # most slow

                # check more data can be added
                i = 0
                for i_worker in range(self.size):
                    max_len = len(next_data_list[i][0])
                    batch_size = solve_increasing_fn_left(lambda bsz: self.get_gpuuse(bsz, max_len)-self.gpu_size, 16)
                    i += batch_size
                    if i >= len(next_data_list): break
                if i >= len(next_data_list):
                    ## More data may be able to be added.
                    data_list = next_data_list
                    self.next_item = None
                else:
                    break
                
            # If no data can be added, output batch
            i = 0
            batch = None
            weight_batch = None
            if self.step < 5:
                self.logger.debug(f"Shape of step {self.step}: (batch_size, max_len)=")
            for dst_rank in range(self.size):
                max_len = len(data_list[i][0])
                batch_size = solve_increasing_fn_left(lambda bsz: self.get_gpuuse(bsz, max_len)-self.gpu_size, 16)
                if self.step < 5:
                    self.logger.info(f"    rank={dst_rank}: ({batch_size}, {max_len}), gpuuse={self.get_gpuuse(batch_size, max_len)/2**30:.03f}GB")

                # 送ってからpad_sequenceとどちらが速い？
                dst_batch = pad_sequence([data[0] for data in data_list[i:i+batch_size]],
                    batch_first=self.batch_first, padding_value=self.padding_value).to(self.device)
                dst_weight_batch = pad_sequence([data[0] for data in data_list[i:i+batch_size]],
                    batch_first=self.batch_first, padding_value=self.padding_weight).to(self.device)
                if dst_rank == self.rank:
                    batch = dst_batch
                    weight_batch = dst_weight_batch
                else:
                    dist_send_tensor(dst_batch, dst=dst_rank)
                    dist_send_tensor(weight_batch, dst=dst_rank)                    
                i += batch_size
            assert i >= len(data_list)
        else:
            batch = dist_recv_tensor(self.main_rank, self.device)
            weight_batch = dist_recv_tensor(self.main_rank, self.device)
        self.step += 1
        return batch, weight_batch
