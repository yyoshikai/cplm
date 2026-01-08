import sys, os, itertools, math
import itertools as itr
from argparse import ArgumentParser, Namespace
from collections.abc import Sized
from inspect import getfullargspec
import numpy as np, pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset, StackDataset, BatchSampler
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem

sys.path += ["/workspace/cplm"]
from src.utils.random import set_random_seed
from src.utils.logger import add_file_handler, get_logger, set_third_party_logger
from src.utils.path import cleardir
from src.utils.rdkit import ignore_rdkit_warning
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





def generate(out_dir: str, targs: Namespace, init_state_path: str, prompt_data: Dataset[T], 
        max_sample: int, 
        max_valid_sample: int|None,
        max_prompt_len: int,
        batch_size: int,
        seed: int, ):

    # Environment
    set_random_seed(seed)
    cleardir(out_dir)
    for subdir in ["generation"]:
        os.makedirs(f"{out_dir}/{subdir}", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # logger
    logger = get_logger(stream=True)
    ignore_rdkit_warning()
    set_third_party_logger()
    logger.debug("args:")
    for name in getfullargspec(generate)[0][2:]:
        logger.debug(f"    {name}: {eval(name)}")

    # model
    model, voc_encoder = get_model(targs, voc_encoder=None, init_state_path=init_state_path, device=device)

    data_size = len(prompt_data)
    sampler = UnfinishedSampler2(data_size, max_sample)
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)
    ns_sample = np.zeros(data_size, int)
    ns_valid_sample = np.zeros(data_size, int)

    result_path = f"{out_dir}/generation.tsv"
    with open(result_path, 'w') as f:
        f.write("prompt_idx\ttrial_idx\tprompt\tgeneration\n")

    n_raw = n_large = 0
    for step, raw_data_idxs in enumerate(batch_sampler):
        
        raw_items = list(DataLoader(Subset(prompt_data, raw_data_idxs), shuffle=False, num_workers=min(len(raw_data_idxs), 28), batch_size=None))
        n_raw += len(raw_items)
        raw_tokens = [get_token(item) for item in raw_items]
        for i_data, token in zip(raw_data_idxs, raw_tokens):
            i_sample = ns_sample[i_data]
            with open(f"generate/{i_sample}/{i_data}/prompt.txt", 'w') as f:
                f.write(' '.join(voc_encoder.decode(token))+'\n') 
        data_idxs, items, tokens = zip(*[data for data in zip(raw_data_idxs, raw_items, raw_tokens) if len(data[2]) <= max_prompt_len])
        n_gen += len(items)
        tokens = [torch.tensor(token, torch.long, device) for token in tokens]
        model.generate2(tokens)


        ns_sample[raw_data_idxs] += 1


        













class UnfinishedSampler2:
    def __init__(self, data_size: int, max_sample: int):
        self.is_remain = np.ones(data_size, int)
        self.max_sample = max_sample
    def __iter__(self):
        for i_sample in range(self.max_sample):
            # fix remain_idxs here
            remain_idxs = np.where(self.is_remain)[0].tolist()
            for idx in remain_idxs:
                if self.is_remain[idx]:
                    yield idx
    def finish(self, idx):
        self.is_remain[idx] = 0



