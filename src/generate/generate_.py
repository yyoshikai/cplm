
from argparse import Namespace
from collections.abc import Callable
from inspect import getfullargspec
from typing import TypeVar
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, BatchSampler
from src.utils.random import set_random_seed
from src.utils.logger import add_file_handler, get_logger, set_third_party_logger
from src.utils.path import cleardir
from src.utils.rdkit import ignore_rdkit_warning
from src.data.tokenizer import VocEncoder
from src.train import get_model

T = TypeVar('T')
T_Streamer = TypeVar('T_Streamer')

def generate(out_dir: str, targs: Namespace, init_state_path: str, prompt_data: Dataset[T], 
        streamer_fn: Callable[[T, int, VocEncoder], T_Streamer], 
        get_token_position_fn: Callable[[T], tuple[list[str], list[int]]],
        max_n_sample: int,
        max_prompt_len: int,
        max_new_token: int|None,
        batch_size: int,
        seed: int, 
        log_position: bool, 
        log_token_range: bool) -> list[list[T_Streamer]]:

    # Environment
    set_random_seed(seed)
    cleardir(out_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # logger
    logger = get_logger(stream=True)
    add_file_handler(logger, f"{out_dir}/generate.log")
    ignore_rdkit_warning()
    set_third_party_logger()
    logger.debug("args:")
    for name in getfullargspec(generate)[0][2:]:
        logger.debug(f"    {name}: {eval(name)}")

    # model
    model, voc_encoder = get_model(targs, voc_encoder=None, init_state_path=init_state_path, device=device)

    data_size = len(prompt_data)
    sampler = UnfinishedSampler(data_size, max_n_sample)
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)
    ns_sample = np.zeros(data_size, int)

    streamerss = [[] for _ in range(len(prompt_data))]
    for step, raw_data_idxs in enumerate(batch_sampler):
        print(f"{raw_data_idxs=}", flush=True)
        raw_items = list(DataLoader(Subset(prompt_data, raw_data_idxs), shuffle=False, num_workers=0, batch_size=None)) # min(len(raw_data_idxs), 28)
        streamers = []
        token_positions = []
        for data_idx, item in zip(raw_data_idxs, raw_items):
            token_position = get_token_position_fn(item)
            if len(token_position[0]) <= max_prompt_len:
                token_positions.append(token_position)
                streamer = streamer_fn(item, ns_sample[data_idx], voc_encoder)
                streamers.append(streamer)
                streamerss[data_idx].append(streamer)
            ns_sample[data_idx] += 1
        tokens, positions = zip(*token_positions)
        tokens = [torch.tensor(voc_encoder.encode(token), dtype=torch.long, device=device) for token in tokens]
        model.generate2(tokens, positions, streamers, max_new_token, position_log_idxs=[len(tokens)-1] if log_position and step == 0 else [], token_range_log_idxs=[len(tokens)-1] if log_token_range and step == 0 else [])
    logger.info(f"{out_dir} finished!")
    return streamerss

class UnfinishedSampler:
    def __init__(self, data_size: int, max_n_sample: int):
        self.is_remain = np.ones(data_size, int)
        self.max_n_sample = max_n_sample
    def __iter__(self):
        for i_sample in range(self.max_n_sample):
            # fix remain_idxs here
            remain_idxs = np.where(self.is_remain)[0].tolist()
            for idx in remain_idxs:
                if self.is_remain[idx]:
                    yield idx
    def finish(self, idx):
        self.is_remain[idx] = 0



