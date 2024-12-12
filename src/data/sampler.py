"""
Bucketing用に実装したが, 結局短い範囲でbucketingすることにしたので使わなかった。
"""
import sys, os, pickle, math, random, itertools, torch, numpy as np
from collections.abc import Iterator, Sized
import torch.distributed as dist
from bisect import bisect_left
from logging import getLogger
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from ..utils.logger import get_logger, add_file_handler

class BucketBatchList:
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, save_dir: str, seed: int, batch_sizes: list[int]=None):
        self.save_dir = save_dir
        self._lazy_bin2idxs = None
        if batch_sizes is None:
            batch_sizes = pickle.load(open(f"{save_dir}/batch_sizes.pkl", 'rb'))
        self.batch_sizes = batch_sizes
        self.rng = np.random.default_rng(seed)
        self.n_call = 0

    @property
    def bin2idxs(self) -> list[list[int]]:
        if self._lazy_bin2idxs is None:
            with open(f"{self.save_dir}/bin2idxs.pkl", 'rb') as f:
                self._lazy_bin2idxs = pickle.load(f)
            assert len(self._lazy_bin2idxs) == len(self.batch_sizes)
        return self._lazy_bin2idxs

    def get_list(self) -> list[np.ndarray[int]]:
        self.logger.info("Calculating bucket indices...")
        batch_idxs = []
        nbin = len(self.bin2idxs)
        for ibin in range(nbin):
            idxs = np.array(self.bin2idxs[ibin], dtype=int)
            batch_size = self.batch_sizes[ibin]
            self.rng.shuffle(idxs)
            batch_idxs += np.split(idxs, np.arange(batch_size, len(idxs), batch_size))
        self.rng.shuffle(batch_idxs)
        return batch_idxs

    @classmethod
    def preprocess(cls, dataset: Dataset[Sized], save_dir: str, 
            right_biased_bins: list, batch_sizes):
        bins = right_biased_bins
        os.makedirs(save_dir, exist_ok=True)
        logger = get_logger(f"{cls.__module__}.{cls.__qualname__}.preprocess")
        add_file_handler(logger, f"{save_dir}/process.log")
        add_file_handler(getLogger(), f"{save_dir}/root.log")

        logger.info("Getting lengths...")
        if callable(getattr(dataset, 'get_lengths', None)):
            lengths = dataset.get_lengths()
        else:
            logger.info(f"Iterating {type(dataset).__name__}...")
            lengths = (len(dataset[idx]) for idx in tqdm(range(len(dataset))))
        logger.info("Making batch idxs...")
        nbin = len(right_biased_bins)-1
        n_short = n_long = 0
        bin2idxs = [[] for _ in range(len(right_biased_bins)-1)]
        for idx, l in enumerate(tqdm(lengths, total=len(dataset))):
            bin = bisect_left(right_biased_bins, l)-1
            if bin == -1:
                n_short+=1
            elif bin == nbin:
                n_long+=1
            else:
                bin2idxs[bin].append(idx)
        
        with open(f"{save_dir}/bin2idxs.pkl", 'wb') as f:
            pickle.dump(bin2idxs, f)
        with open(f"{save_dir}/batch_sizes.pkl", 'wb') as f:
            pickle.dump(batch_sizes, f)
        logger.info(f"   ~{bins[0]:3}={n_short}")
        n_total_batch = 0
        for ibin in range(nbin):
            nidx = len(bin2idxs[ibin])
            n_batch = math.ceil(nidx/batch_sizes[ibin])
            logger.info(f"{bins[ibin]+1:3}~{bins[ibin+1]:3}={nidx} ({n_batch})")
            n_total_batch += n_batch
        logger.info(f"{bins[nbin]+1:3}~   ={n_long}")
        logger.info(f"{n_total_batch=}")


class DDPBatchSampler(Sampler[list[int]]):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, batch_list: BucketBatchList):
        self.batch_list = batch_list
        self.rank = dist.get_rank()
        self.size = dist.get_world_size()
        self.is_main = self.rank == 0
    
    def __iter__(self):
        if self.is_main:
            batch_list = self.batch_list.get_list()
            self.logger.info("Sending batch list...")
            batch_list += batch_list[:self.size-(len(batch_list)%self.size)]
            for dst_rank in range(1, self.size):
                dist.send_object_list([batch_list[dst_rank::self.size]], dst=dst_rank)
            self.logger.info("Sended.")
            batch_list = batch_list[0::self.size]
        else:
            batch_list = [None]
            dist.recv_object_list(batch_list, 0)
            batch_list = batch_list[0]
        
        return iter(batch_list)
