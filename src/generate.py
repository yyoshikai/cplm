import sys, os, itertools, pickle, yaml, math
from collections.abc import Sized
from inspect import getfullargspec
from typing import Optional, Literal
import numpy as np, pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset, StackDataset, BatchSampler, Dataset
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem, RDLogger

sys.path += ["/workspace/cplm"]
from src.utils.random import set_random_seed
from src.utils.logger import add_stream_handler, add_file_handler, get_logger
from src.utils.path import cleardir
from src.utils.time import wtqdm
from src.data.tokenizer import VocEncoder
from src.data import index_dataset
from src.model import Model
from src.model.mamba import MambaModel
from src.evaluate import parse_mol_tokens, parse_mol
PROJ_DIR = "/workspace/cplm"
WORKDIR = "/workspace"

def generate(model: Model|MambaModel, voc_encoder: VocEncoder, prompt_token_data: Dataset[Tensor], center_data: Dataset[Tensor], rdir: str, n_trial: int, batch_size: int, 
        seed: int, max_len: int):
    
    # Environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_random_seed(seed)
    ## Result dir
    cleardir(rdir)

    ## Logger
    logger = get_logger()
    add_stream_handler(logger)
    add_file_handler(logger, f"{rdir}/generate.log")
    token_logger = get_logger("tokens")
    token_logger.propagate = False
    add_file_handler(token_logger, f"{rdir}/tokens.log")
    token_logger.debug(f"[step][batch_idx][batch_index]=")
    RDLogger.DisableLog("rdApp.*")

    ## Log args
    logger.info("args:")
    for name in getfullargspec(generate)[0][2:]:
        logger.info(f"    {name}: {eval(name)}")

    # data
    index_data, prompt_token_data = index_dataset(prompt_token_data)
    data = StackDataset(index_data, prompt_token_data, center_data)
    pad_token = voc_encoder.pad_token
    end_token = voc_encoder.voc2i['[END]']
    def collate_fn(batch):
        indices, batch, centers = list(zip(*batch))
        batch = pad_sequence(batch, padding_value=pad_token)
        return indices, batch, centers

    num_workers = min(28, batch_size)
    model.to(device)

    # 生成
    sampler = UnfinishedSampler(data, n_trial)
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)
    smiless = [None] * len(data)
    errors = [None] * len(data)
    indices = [None] * len(data)
    os.makedirs(f"{rdir}/sdf", exist_ok=True)

    with torch.inference_mode():
        for step, batch_idxs in enumerate(pbar:=wtqdm(batch_sampler)):

            pbar.start('data')
            train_loader = DataLoader(Subset(data, batch_idxs), shuffle=False, 
                    num_workers=num_workers, batch_size=len(batch_idxs), 
                    collate_fn=collate_fn)
            batch_indices, batch, centers = next(iter(train_loader))
            L, B = batch.shape
            batch = batch.to(device)
            logger.debug(f"batch_idxs[{step}]={batch_idxs}")
            logger.debug(f"batch_indices[{step}]={batch_indices}")

            
            pbar.start("generation")
            outputs = model.generate2(batch, '[END]', max_len, pad_token, 10)
            outputs = [out.cpu().numpy() for out in outputs]
            # cut outputs
            cut_outputs = []
            for b in range(B):
                prompt_size = torch.sum(batch[:,b] != pad_token)
                cut_outputs.append(outputs[b][prompt_size:prompt_size+max_len])
            outputs = cut_outputs

            # Log tokens
            for batch_idx, batch_index, input, output in zip(batch_idxs, batch_indices, batch.T, outputs):
                token_logger.debug(f"[{step}][{batch_idx}][{batch_index}]Input={voc_encoder.decode(input)}")
                token_logger.debug(f"[{step}][{batch_idx}][{batch_index}]Output={voc_encoder.decode(output)}")

            pbar.start("parsing")
            n_valid = 0
            batch_errors = []
            for i, output in enumerate(outputs):

                words = ['[LIGAND]']+voc_encoder.decode(itertools.takewhile(lambda x: x != end_token, output))
                center = centers[i]
                idx = batch_idxs[i]

                if not sampler.is_remain[idx]: continue

                indices[idx] = batch_indices[i]

                error, smiles, coords = parse_mol_tokens(words)
                smiless[idx] = smiles
                if error != "":
                    errors[idx] = error
                    continue

                coords += center
                error, mol = parse_mol(smiles, coords)
                errors[idx] = error
                batch_errors.append(error)
                if error != "":
                    continue
                with open(f"{rdir}/sdf/{idx}.sdf", 'w') as f:
                    f.write(Chem.MolToMolBlock(mol))
                n_valid += 1
                sampler.is_remain[idx] = False
            logger.debug(f"{n_valid=}")
            logger.debug(f"{batch_errors=}")

    df = pd.DataFrame({'idx': indices, 'smiles': smiless, 'error': errors})
    df.to_csv(f"{rdir}/info.csv")

    return True

class UnfinishedSampler:
    def __init__(self, dataset: Sized, max_cycle: int=math.inf):
        
        self.iter_idxs = list(range(len(dataset)))
        self.is_remain = np.full(len(dataset), True)
        self.max_cycle = max_cycle

    def __iter__(self):

        i_cycle = 0
        while True:
            if np.all(~self.is_remain):
                return
            for i in np.where(self.is_remain)[0]:
                if self.is_remain[i]:
                    yield i
            i_cycle += 1
            if i_cycle >= self.max_cycle:
                return