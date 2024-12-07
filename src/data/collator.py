import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from ..utils.logger import get_logger

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