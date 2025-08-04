
import torch
from collections.abc import Iterator, Sized
from logging import getLogger
import numpy as np
from torch.utils.data import Sampler, BatchSampler

class InfiniteRandomSampler(Sampler):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, dataset: Sized, generator: torch.Generator=None):
        self.dataset = dataset
        self.generator = generator
        self.epoch = None

    def __iter__(self) -> Iterator[int]:
        self.epoch = 0
        while True:
            self.logger.info('Calculating sample order...')
            order = torch.randperm(len(self.dataset), generator=self.generator).tolist()
            self.logger.info('calculated.')
            yield from order
            self.epoch += 1
            self.logger.info(f"{self.epoch} epoch finished.")
