
import torch
from collections.abc import Iterator, Sized
from logging import getLogger
import numpy as np
from torch.utils.data import Sampler

class InfiniteRandomSampler(Sampler):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, dataset: Sized, generator: torch.Generator=None, mmap_path: str|None=None):
        self.size = len(dataset)
        self.mmap_path = mmap_path
        self.generator = generator
        self.epoch = None
        self._order = None

    def __iter__(self) -> Iterator[int]:
        self.epoch = 0
        while True:
            self.logger.info('Calculating sample order...')
            order = torch.randperm(self.size, generator=self.generator).numpy()
            self.logger.info('calculated.')
            if self.mmap_path is None:
                self._order = order
            else:
                self._order = np.memmap(self.mmap_path, dtype=np.int64, mode='w+', shape=(self.size,))
            yield from map(int, self._order)
            self.epoch += 1
            self.logger.info(f"{self.epoch} epoch finished.")
