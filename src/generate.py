import sys, os, itertools, math, logging
from argparse import ArgumentParser
from collections.abc import Sized
from inspect import getfullargspec
from logging import getLogger
import numpy as np, pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader, Subset, StackDataset, BatchSampler
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem, RDLogger

sys.path += ["/workspace/cplm"]
from src.utils.random import set_random_seed
from src.utils.logger import add_stream_handler, add_file_handler, get_logger, disable_openbabel_log
from src.utils.path import cleardir
from src.utils.rdkit import set_rdkit_logger
from src.data.tokenizer import SmilesTokenizer
from src.finetune import get_finetune_data
from src.train import get_model
from src.evaluate import parse_mol_tokens2

def add_generate_args(parser: ArgumentParser):
    parser.add_argument('--n-trial', type=int, default=1)
    parser.add_argument("--max-len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, required=True)

def generate(rdir: str, n_trial: int, batch_size: int, 
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
    logger = get_logger()
    add_stream_handler(logger)
    add_file_handler(logger, f"{rdir}/generate.log")
    token_logger = get_logger("tokens")
    token_logger.propagate = False
    add_file_handler(token_logger, f"{rdir}/tokens.log")
    token_logger.debug(f"[step][batch_idx][batch_index]=")
    RDLogger.DisableLog("rdApp.*")
    disable_openbabel_log()

    # third-party modules
    set_rdkit_logger().setLevel(logging.CRITICAL)
    getLogger('.prody').setLevel(logging.CRITICAL)
    disable_openbabel_log()
    transformers.utils.logging.enable_propagation()
    transformers.utils.logging.disable_default_handler()

    ## Log args
    logger.info("args:")
    for name in getfullargspec(generate)[0][2:]:
        logger.info(f"    {name}: {eval(name)}")

    # Data
    added_vocs = SmilesTokenizer()
    voc_encoder, _raw, prompt_token_data, _weight, center_data, _rotation, _protein_filename_data, _ligand_filename_data \
        = get_finetune_data(model_args, 'test', False, False, added_vocs, prompt_score='none' if no_score else 'low')
    data = StackDataset(prompt_token_data, center_data)
    pad_token = voc_encoder.pad_token
    end_token = voc_encoder.voc2i['[END]']


    num_workers = min(28, batch_size)
    
    # model
    model = get_model(model_args, voc_encoder, init_state_path, device)
    model.to(device)

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
            outputs = model.generate2(batch, '[END]', max_len, pad_token, tqdm=tqdm_generate)
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
