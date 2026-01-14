import sys, os, itertools, math
from time import time
from argparse import ArgumentParser, Namespace
from collections.abc import Sized, Callable, Generator
from logging import getLogger
from inspect import getfullargspec
from typing import TypeVar
import numpy as np, pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, StackDataset, BatchSampler
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem

sys.path += ["/workspace/cplm"]
from src.utils import should_show
from src.utils.random import set_random_seed
from src.utils.logger import add_file_handler, get_logger, set_third_party_logger
from src.utils.path import cleardir
from src.utils.rdkit import ignore_rdkit_warning
from src.model import Streamer
from src.data.tokenizer import SmilesTokenizer, VocEncoder
from src.finetune import get_finetune_data
from src.train import get_model
from src.evaluate import parse_mol_tokens2

def add_generate_args(parser: ArgumentParser):
    parser.add_argument('--n-trial', type=int, default=1)
    parser.add_argument("--max-len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=True)

def generate0(rdir: str, n_trial: int, batch_size: int, 
        seed: int, max_len: int, model_args, init_state_path, no_score, tqdm_generate: bool=False, max_prompt_len: int=math.inf):

    # Environment
    os.makedirs(rdir, exist_ok=True)
    set_random_seed(seed)
    ## Result dir
    cleardir(rdir)
    ## device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## check if result exists
    if os.path.exists(f"{rdir}/info.csv"):
        print(f"{rdir} already finished.", flush=True)
        sys.exit()

    ## Logger
    logger = get_logger(stream=True)
    add_file_handler(logger, f"{rdir}/generate.log")
    token_logger = get_logger("tokens")
    token_logger.propagate = False
    add_file_handler(token_logger, f"{rdir}/tokens.log")
    token_logger.debug(f"[step][batch_idx][batch_index]=")
    ignore_rdkit_warning()
    set_third_party_logger()

    ## Log args
    logger.info("args:")
    for name in getfullargspec(generate)[0][2:]:
        logger.info(f"    {name}: {eval(name)}")

    # Data
    added_vocs = SmilesTokenizer()
    voc_encoder, _raw, prompt_token_data, _position, _weight, center_data, _rotation, _protein_filename_data, _ligand_filename_data, data_logs \
        = get_finetune_data(model_args, 'test', False, False, added_vocs, prompt_score='none' if no_score else 'low')
    for msg, level in data_logs: 
        logger.log(level, msg)
    data = StackDataset(prompt_token_data, center_data)
    pad_token = voc_encoder.pad_token
    end_token = voc_encoder.voc2i['[END]']


    num_workers = min(28, batch_size)
    
    # model
    model = get_model(model_args, voc_encoder, init_state_path, device)

    # 生成
    sampler = UnfinishedSampler(data, n_trial, max_prompt_len=max_prompt_len)
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)
    smiless = [None] * len(data)
    errors = [None] * len(data)
    os.makedirs(f"{rdir}/sdf", exist_ok=True)

    with torch.inference_mode():
        for step, batch_idxs in enumerate(batch_sampler):
            logger.debug(f"batch_idxs[{step}]={batch_idxs}")

            train_loader = DataLoader(Subset(data, batch_idxs), shuffle=False, 
                    num_workers=num_workers, batch_size=len(batch_idxs), 
                    collate_fn=lambda x: x)
            batch = next(iter(train_loader))
            tokens, centers = list(zip(*batch))
            for idx, token in zip(batch_idxs, tokens):
                token_size = len(token)
                sampler.sizes[idx] = token_size
                if token_size > max_prompt_len:
                    errors[idx] = 'LARGE_PROMPT'
                    logger.warning(f"Too large prompt: {idx}({token_size})")
            
            batch_idxs = [idx for idx, token in zip(batch_idxs, tokens) if len(token) <= max_prompt_len]
            centers = [center for center, token in zip(centers, tokens) if len(token) <= max_prompt_len]
            tokens = [token for token in tokens if len(token) <= max_prompt_len]
            if len(batch_idxs) == 0:
                logger.info(f"step[{step}] All prompts were too large.")
                continue
            batch = pad_sequence(tokens)
            logger.info(f"small batch_idxs[{step}]={batch_idxs}")

            L, B = batch.shape
            batch = batch.to(device)
            
            logger.info(f"step[{step}] generating...")
            outputs = model.generate(batch, '[END]', max_len, pad_token, tqdm=tqdm_generate)
            outputs = [out.cpu().numpy() for out in outputs]

            # Log tokens
            for batch_idx, input, output in zip(batch_idxs, batch.T, outputs):
                token_logger.debug(f"[{step}][{batch_idx}]Input={voc_encoder.decode(input)}")
                token_logger.debug(f"[{step}][{batch_idx}]Output={voc_encoder.decode(output)}")

            logger.debug(f"step[{step}] parsing...")
            for i, output in enumerate(outputs):

                words = ['[LIGAND]']+voc_encoder.decode(itertools.takewhile(lambda x: x != end_token, output))
                idx = batch_idxs[i]

                if not sampler.is_remain[idx]: continue

                error, mol = parse_mol_tokens2(words, centers[i])
                errors[idx] = error
                if error != "": continue
                with open(f"{rdir}/sdf/{idx}.sdf", 'w') as f:
                    f.write(Chem.MolToMolBlock(mol))
                sampler.is_remain[idx] = False
            logger.info(f"batch_errors={[errors[idx] for idx in batch_idxs]}")

    df = pd.DataFrame({'smiles': smiless, 'error': errors})
    df.to_csv(f"{rdir}/info.csv")

    return True

class UnfinishedSampler:
    def __init__(self, dataset: Sized, max_cycle: int=math.inf, max_prompt_len: int=math.inf):
        
        self.iter_idxs = list(range(len(dataset)))
        self.is_remain = np.full(len(dataset), True)
        self.max_cycle = max_cycle
        self.sizes = [None] * len(dataset)
        self.max_prompt_len = max_prompt_len

    def __iter__(self):

        i_cycle = 0
        while True:
            if np.all(~self.is_remain):
                return
            for i in np.where(self.is_remain)[0]:
                if self.sizes[i] is not None and self.sizes[i] > self.max_prompt_len:
                    continue
                if self.is_remain[i]:
                    yield i
            i_cycle += 1
            if i_cycle >= self.max_cycle:
                return

class GeneratorStreamer(Streamer):
    logger = getLogger(f"{__qualname__}")
    def __init__(self, name: str, prompt_token_path: str, new_token_path: str, voc_encoder: VocEncoder):
        self.name = name
        self.voc_encoder = voc_encoder
        self.prompt_token_path = prompt_token_path
        self.new_token_path = new_token_path
        
        self.put_gen = self.put_generator()
        self.is_prompt = True
        self.new_token_dir_made = False
        self.n = 0
        next(self.put_gen)
    def estimated_n_token(self):
        return None
    def put_generator(self) -> Generator[tuple[bool, list[int], list[int]], list[int], None]:
        raise NotImplementedError
    def put(self, token: list[int]) -> tuple[bool, list[int], list[int]]:
        if self.is_prompt:
            os.makedirs(os.path.dirname(self.prompt_token_path), exist_ok=True)
            with open(self.prompt_token_path, 'w') as f:
                f.write(' '.join(self.voc_encoder.decode(token))+'\n')
            self.start = time()
            self.is_prompt = False
        else:
            if not self.new_token_dir_made:
                os.makedirs(os.path.dirname(self.new_token_path), exist_ok=True)
                self.new_token_dir_made = True
            with open(self.new_token_path, 'a') as f:
                f.write(self.voc_encoder.i2voc[token[0]]+' ')
            self.n += 1
            if should_show(self.n):
                t = time() - self.start
                if t >= 1.0:
                    est_n = self.estimated_n_token()
                    if est_n is None:
                        self.logger.info(f"[{self.name}]generated {self.n}/? token in {t:.02f}s")
                    else:
                        est_t = t * est_n / self.n
                        self.logger.info(f"[{self.name}]generated {self.n}/{est_n} token in {t:02f}s (estimated end={est_t:.02f}s)")
        return self.put_gen.send(token)

T = TypeVar('T')

def generate(out_dir: str, targs: Namespace, init_state_path: str, prompt_data: Dataset[T], 
        streamer_fn: Callable[[T, int, VocEncoder], Streamer], 
        get_token_position_fn: Callable[[T], tuple[list[str], list[int]]],
        max_n_sample: int,
        max_prompt_len: int,
        max_new_token: int|None,
        batch_size: int,
        seed: int, ):

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
    sampler = UnfinishedSampler2(data_size, max_n_sample)
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)
    ns_sample = np.zeros(data_size, int)

    n_raw = n_large = 0
    for step, raw_data_idxs in enumerate(batch_sampler):
        
        raw_items = list(DataLoader(Subset(prompt_data, raw_data_idxs), shuffle=False, num_workers=0, batch_size=None)) # min(len(raw_data_idxs), 28)
        n_raw += len(raw_items)
        raw_token_positions = [get_token_position_fn(item) for item in raw_items]
        data_idxs, items, token_positions = zip(*[data for data in zip(raw_data_idxs, raw_items, raw_token_positions) if len(data[2]) <= max_prompt_len])
        streamers = [streamer_fn(item, ns_sample[data_idx], voc_encoder) for item, data_idx in zip(items, data_idxs)]
        tokens, positions = zip(*token_positions)
        tokens = [torch.tensor(voc_encoder.encode(token), dtype=torch.long, device=device) for token in tokens]
        model.generate2(tokens, positions, streamers, max_new_token) # , position_log_idxs=[0] if step == 0 else []

        ns_sample[raw_data_idxs] += 1
    logger.info(f"{out_dir} finished!")

class UnfinishedSampler2:
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



