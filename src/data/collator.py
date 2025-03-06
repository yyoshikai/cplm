from bisect import bisect_right
from functools import partial
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, BatchSampler
from torch.nn.utils.rnn import pad_sequence

import torch.distributed as dist
from ..utils.logger import get_logger
from .sampler import InfiniteRandomSampler

class StringCollateLoader:
    logger = get_logger(f"{__module__}.{__qualname__}")
    def __init__(self, dataset: Dataset, num_workers:int, pin_memory: bool,
            token_per_batch: int, batch_first: bool,
            padding_value: int, ):
        self.loader = DataLoader(dataset, shuffle=True, num_workers=num_workers, 
            pin_memory=pin_memory, persistent_workers=True)
        self.token_per_batch = token_per_batch
        self.logger.debug("Initializing iterator...")
        self.iter = self.loader.__iter__()
        self.logger.debug("Initialized.")
        self.next_item = None
        self.step = 0
        self.batch_first = batch_first
        self.padding_value = padding_value

    def __next__(self) -> tuple[torch.Tensor, int]:
        # get batch
        batch = []
        max_length = 0
        n_token = 0
        while True:
            if self.next_item is None:
                try:
                    self.next_item = self.iter.__next__().squeeze(0)
                except StopIteration:
                    self.logger.info(f"Epoch finished at {self.step} step.")
                    self.iter = self.loader.__iter__()
                    self.next_item = self.iter.__next__().squeeze(0)
            if ((len(batch)+1) * max(max_length, len(self.next_item)) <= self.token_per_batch):
                batch.append(self.next_item)
                max_length = max(max_length, len(self.next_item))
                n_token += len(self.next_item)
                self.next_item = None
            else:
                if len(batch) == 0:
                    self.logger.warning(f"Item was too large even for single item per batch({len(self.next_item)}), and not used.")
                    self.next_item = None
                    continue
                else:
                    break
        batch = pad_sequence(batch, batch_first=self.batch_first, 
                padding_value=self.padding_value).to(torch.long)
        
        if self.step < 20:
            self.logger.info(f"batch={tuple(batch.shape)}, fill rate: {n_token}/{batch.numel()}")
        self.step+=1

        return batch, n_token

class DDPStringCollateLoader:
    logger = get_logger(f"{__module__}.{__qualname__}")
    def __init__(self, dataset: Dataset, num_workers:int, pin_memory: bool, prefetch_factor: int, 
            token_per_batch: int, batch_first: bool, padding_value: int,
            device: torch.device, main_rank: int=0):
        self.size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.main_rank = main_rank
        self.device = device

        if self.rank == self.main_rank:
            sampler = InfiniteRandomSampler(dataset)
            self.token_per_batch = token_per_batch
            self.loader = DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=num_workers, 
                pin_memory=pin_memory, persistent_workers=True, prefetch_factor=prefetch_factor)
            self.iter = self.loader.__iter__()
            self.next_item = None
            self.step = 0
            self.batch_first = batch_first
            self.padding_value = padding_value

    def __next__(self) -> torch.Tensor:
        if self.rank == self.main_rank:
            data_list = []
            while True:
                # get next item
                if self.next_item is None:
                    self.next_item = self.iter.__next__()

                # check maximum size
                next_size = len(self.next_item)
                if next_size > self.token_per_batch:
                    self.logger.warning(f"Item was too large even for single item per batch({len(self.next_item)}), and not used.")
                    self.next_item = None
                    continue

                # insert size
                i = bisect_right(data_list, -len(self.next_item), key=lambda x: -len(x))
                next_data_list = data_list[:i]+[self.next_item]+data_list[i:] # most slow

                # check more data can be added
                i = 0
                for i_worker in range(self.size):
                    max_len = len(next_data_list[i])
                    i += self.token_per_batch // max_len
                    if i >= len(next_data_list): break
                if i >= len(next_data_list):
                    ## More data may be able to be added.
                    data_list = next_data_list
                    self.next_item = None
                    continue
                break
                
            # If no data can be added, output batch
            i = 0
            batch = None
            for dst_rank in range(self.size):
                max_len = len(data_list[i])
                batch_size = self.token_per_batch // max_len

                # 送ってからpad_sequenceとどちらが速い？
                dst_batch = pad_sequence(data_list[i:i+batch_size],
                    batch_first=self.batch_first, padding_value=self.padding_value).to(self.device)
                if dst_rank == self.rank:
                    batch = dst_batch
                else:
                    dist.send(torch.tensor(dst_batch.shape, dtype=torch.int, device=self.device), dst=dst_rank)
                    dist.send(dst_batch, dst=dst_rank)
                i += batch_size
            assert i >= len(data_list)
        else:
            batch_shape = torch.zeros(2, dtype=torch.int, device=self.device)
            dist.recv(batch_shape, src=self.main_rank)
            batch = torch.zeros(batch_shape[0], batch_shape[1], dtype=torch.long, device=self.device)
            dist.recv(batch, src=self.main_rank)
        return batch
