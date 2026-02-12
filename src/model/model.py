import torch.nn as nn
from torch import Tensor

class Streamer:
    def put(self, tokens: list[int]) -> tuple[bool, int, list[int]]:
        """
        Returns
        -------
        is_remain: bool
            If True, next token must be generated. If False, self.put() must not be called any more.
        position: int
            next position
        token_range: list[int]
            Next token range.
        """
        raise NotImplementedError
    def estimated_n_token(self):
        return None

class WrapperStreamer(Streamer):
    def __init__(self, streamer: Streamer):
        self.streamer = streamer
    def estimated_n_token(self):
        return self.streamer.estimated_n_token()

class LanguageModel(nn.Module):
    def forward(self, src: Tensor, position: Tensor, out_state: bool=False,
            get_mem: bool=False, offset: list[float]=None, mem_path: str=None):
        raise NotImplementedError
    
    def generate2(self, contexts: list[Tensor], positions: list[list[int]], streamers: list[Streamer], max_new_token: int|None):
        raise NotImplementedError
    
    def get_gpuuse(self, batch_size: int, length: int, bf16: bool, kernel: str, 
            capture_rate: bool=True):
    
    @property
    def state_size(self) -> int:
        raise NotImplementedError