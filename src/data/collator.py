from bisect import bisect_right
from collections.abc import Callable, Iterable, Generator
from functools import partial
from logging import getLogger
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import torch.distributed as dist
from .data import IndexDataset
from .tokenizer import VocEncoder
from .sampler import InfiniteRandomSampler
from ..model import Model, MambaModel2
from ..utils import reveal_data

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

class StringCollateIterator(Iterable[Tensor]):
    logger = getLogger(f"{__module__}.{__qualname__}")
    data_logger = getLogger(f"dexs.{__module__}.{__qualname__}")
    def __init__(self, dataset: Dataset[tuple[Tensor, Tensor]], n_batch: int, model: Model|MambaModel2, num_workers:int, pin_memory: bool, prefetch_factor: int, 
            gpu_size: float, batch_first: bool, padding_value: int, bf16: bool, kernel: str,
            padding_weight: float=0.0, seed: int=None, ):

        dataset = IndexDataset(dataset)
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        sampler = InfiniteRandomSampler(dataset, generator=generator)
        self.n_batch = n_batch
        self.gpu_size = gpu_size
        self.loader = DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=num_workers, 
            pin_memory=pin_memory, persistent_workers=True, prefetch_factor=prefetch_factor)
        
        if isinstance(model, Model): 
            self.get_gpuuse = partial(model.get_gpuuse, bf16=bf16, kernel=kernel)
        elif isinstance(model, MambaModel2): 
            self.get_gpuuse = partial(model.get_gpuuse, bf16=bf16)
        else: 
            raise ValueError(f"Unsupported model: {type(model)}")

        self.next_item = None
        self.step = 0
        self.batch_first = batch_first
        self.padding_value = padding_value
        self.padding_weight = padding_weight
        
        self.n_item = 0
        self.n_large_item = 0

    def split_data_list(self, data_list: list[tuple[Tensor, Tensor]]) -> tuple[list[list[tuple[Tensor, Tensor]], bool]]:
        datas = []
        for _ in range(self.n_batch):
            max_len = len(data_list[0][0])
            batch_size = solve_increasing_fn_left(lambda bsz: self.get_gpuuse(bsz, max_len)-self.gpu_size, 16)

            datas.append(data_list[:batch_size])
            if len(data_list) == 0: break
        return datas, len(data_list) == 0


    def collate(self, data_list: list[tuple[Tensor, Tensor]]) -> list[tuple[Tensor, Tensor]]:
        padded_batches = []
        data_lists, can_be_batched = self.split_data_list(data_list)
        assert can_be_batched
        for data_list in data_lists:
            # 送ってからpad_sequenceとどちらが速い？
            batch = pad_sequence([data[0] for data in data_list],
                batch_first=self.batch_first, padding_value=self.padding_value)
            weight_batch = pad_sequence([data[1] for data in data_list],
                batch_first=self.batch_first, padding_value=self.padding_weight)
            padded_batches.append((batch, weight_batch))
        return padded_batches

    def __iter__(self) -> Generator[Tensor, None, None]:
        data_list = []
        loader_iter = self.loader.__iter__()
        while True:
            # get next item
            if self.next_item is None:
                try:
                    index, self.next_item = loader_iter.__next__()
                except StopIteration:
                    if len(data_list) > 0:
                        yield from self.collate(data_list)
                    raise StopIteration

                if self.step < 5:
                    self.data_logger.info(f"item {self.n_item}: idx={index}")
                    self.data_logger.info(f"    {reveal_data(self.next_item)}")
                self.n_item += 1
            
            # check maximum size
            next_size = len(self.next_item[0])
            if self.get_gpuuse(batch_size=1, length=next_size) > self.gpu_size:
                self.n_large_item += 1
                self.logger.warning(f"Item was too large even for 1 item per batch({len(self.next_item[0])}), and not used ({self.n_large_item}/{self.n_item}).")
                self.next_item = None
                continue

            # insert size
            i = bisect_right(data_list, -next_size, key=lambda x: -len(x[0]))
            next_data_list = data_list[:i]+[self.next_item]+data_list[i:] # most slow

            # check more data can be added
            _, can_be_batched = self.split_data_list(next_data_list)
            if can_be_batched:
                data_list = next_data_list
                self.next_item = None
            else:
                yield from self.collate(data_list)
                data_list = []

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
            self.batch_iterator = StringCollateIterator(dataset, self.size, model, num_workers, pin_memory, prefetch_factor, gpu_size, batch_first, padding_value, bf16, kernel, padding_weight, seed)
            


    def __iter__(self) -> Generator[Tensor, None, None]:
        if self.rank == self.main_rank:
            batches = []
            for batch in self.batch_iterator:
                batches.append(batch)

                # Output batch
                if len(batches) == self.size:
                    for dst_rank, batch in enumerate(batches):
                        if dst_rank == self.rank:
                            this_batch = batch
                        else:
                            dist_send_tensor(batch)
                    batches = []
            
            if len(batches) > 0:
                


        if self.rank == self.main_rank:
            data_list = []
            while True:
                # get next item
                if self.next_item is None:
                    try:
                        index, self.next_item = self.iter.__next__()
                    except StopIteration:
                        break
                    if self.step < 5:
                        self.data_logger.info(f"item {self.n_item}: idx={index}")
                        self.data_logger.info(f"    {reveal_data(self.next_item)}")
                    self.n_item += 1
                
                # check maximum size
                next_size = len(self.next_item[0])
                if self.get_gpuuse(batch_size=1, length=next_size) > self.gpu_size:
                    self.n_large_item += 1
                    self.logger.warning(f"Item was too large even for 1 item per batch({len(self.next_item[0])}), and not used ({self.n_large_item}/{self.n_item}).")
                    self.next_item = None
                    continue

                # insert size
                i = bisect_right(data_list, -next_size, key=lambda x: -len(x[0]))
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

            # len(data_list) == 0 means loop was broken by first self.iter.__next__().
            if len(data_list) == 0:
                # Send empty tensor to tell StopIteration to other process
                for dst_rank in range(self.size):
                    if dst_rank != self.rank:
                        dist_send_tensor(torch.tensor([],device=self.device), dst_rank)
                raise StopIteration
                
            # Otherwise, loop was broken by either self.iter.__next__() or no more data can be added.
            i = 0
            batch = None
            weight_batch = None
            if self.step < 5:
                self.logger.debug(f"Shape of step {self.step}: (batch_size, max_len)=")
            for dst_rank in range(self.size):
                max_len = len(data_list[i][0])
                batch_size = solve_increasing_fn_left(lambda bsz: self.get_gpuuse(bsz, max_len)-self.gpu_size, 16)
                if self.step < 5:
                    self.logger.debug(f"    rank={dst_rank}: ({batch_size}, {max_len}), gpuuse={self.get_gpuuse(batch_size, max_len)/2**30:.03f}GB")

                # 送ってからpad_sequenceとどちらが速い？
                dst_batch = pad_sequence([data[0] for data in data_list[i:i+batch_size]],
                    batch_first=self.batch_first, padding_value=self.padding_value).to(self.device)
                dst_weight_batch = pad_sequence([data[1] for data in data_list[i:i+batch_size]],
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
            if batch.numel() == 0:
                raise StopIteration
            weight_batch = dist_recv_tensor(self.main_rank, self.device)
        self.step += 1
        return batch, weight_batch
