import sys, os, math
import itertools as itr
from argparse import ArgumentParser, Namespace
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Subset, StackDataset
from torch.nn.utils.rnn import pad_sequence
from rdkit import RDLogger
from openbabel.openbabel import OBMol

from src.utils.logger import get_logger, add_file_handler, disable_openbabel_log
from src.data import WrapDataset, CacheDataset
from src.data.datasets.pdb import PDBUniMolRandomDataset
from src.data.protein import ProteinTokenizeDataset
from src.data.tokenizer import SentenceDataset, VocEncoder, TokenEncodeDataset
from src.train import get_model
from src.generate import UnfinishedSampler
from src.evaluate import parse_coord_tokens

# arguments
parser = ArgumentParser()
parser.add_argument('--studyname', required=True)
parser.add_argument('--opt', type=int, required=True)
parser.add_argument('--n-trial', type=int, default=10)
parser.add_argument('--max-len', type=int, default=1000)
parser.add_argument('--max-prompt-len', type=int, default=math.inf)
parser.add_argument('--batch-size', type=int, required=True)
parser.add_argument('--num-workers', type=int, required=True)
parser.add_argument('--tqdm', action='store_true')
args = parser.parse_args()

# environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
out_dir = f"predict_structure/results/{args.studyname}/{args.opt}"
os.makedirs(out_dir, exist_ok=True)
logger = get_logger(stream=True)
add_file_handler(logger, f"{out_dir}/predict_structure.log")
token_logger = get_logger("tokens")
token_logger.propagate = False
add_file_handler(token_logger, f"{out_dir}/tokens.log")
token_logger.debug(f"[step][batch_idx][batch_index]=")
RDLogger.DisableLog("rdApp.*")
disable_openbabel_log()

train_dir = f"training/results/{args.studyname}"
targs = Namespace(**yaml.safe_load(open(f"{train_dir}/args.yaml")))
assert targs.PDBUniMolRandom > 0
assert not targs.coord_follow_atom

# dataset
class ProteinSizeDataset(WrapDataset[int]):
    def __init__(self, protein_data: WrapDataset[OBMol]):
        super().__init__(protein_data)
    
    def __getitem__(self, idx: int):
        protein: OBMol = self.dataset[idx]
        return protein.NumAtoms()

protein = PDBUniMolRandomDataset('valid')
protein = Subset(protein, [0]) # temp
protein = CacheDataset(protein)
protein_size = ProteinSizeDataset(protein)
protein = ProteinTokenizeDataset(protein, not targs.no_pocket_heavy_atom, targs.pocket_h_atom, not targs.no_pocket_heavy_coord, targs.pocket_h_coord, False, targs.coord_range, False)
protein = SentenceDataset('[POCKET]', protein, '[XYZ]')
vocs = protein.vocs() | {'[END]'}
voc_encoder = VocEncoder(vocs)
pad_token = voc_encoder.pad_token
end_token = voc_encoder.voc2i['[END]']
protein = TokenEncodeDataset(protein, voc_encoder)
data = StackDataset(protein, protein_size)

# model
model = get_model(targs, voc_encoder, f"{train_dir}/models/{args.opt}.pth", device)

sampler = UnfinishedSampler(data, args.n_trial, args.max_prompt_len)
batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=False)
coords = [None] * len(data)
errors = [None] * len(data)

with torch.inference_mode():
    for step, batch_idxs in enumerate(batch_sampler):
        logger.debug(f"batch_idxs[{step}]={batch_idxs}")

        train_loader = DataLoader(Subset(data, batch_idxs), shuffle=False, 
                num_workers=args.num_workers, batch_size=len(batch_idxs), 
                collate_fn=lambda x: x)
        batch = next(iter(train_loader))
        tokens, ns_atom = zip(*batch)
        for idx, token in zip(batch_idxs, tokens):
            token_size = len(token)
            sampler.sizes[idx] = token_size
            if token_size > args.max_prompt_len:
                errors[idx] = 'LARGE_PROMPT'
                logger.warning(f"Too large prompt: {idx}({token_size})")
        
        batch_idxs = [idx for idx, token in zip(batch_idxs, tokens) if len(token) <= args.max_prompt_len]
        ns_atom = [n_atom for n_atom, token in zip(ns_atom, tokens) if len(token) <= args.max_prompt_len]
        tokens = [token for token in tokens if len(token) <= args.max_prompt_len]
        if len(batch_idxs) == 0:
            logger.info(f"step[{step}] All prompts were too large.")
            continue
        batch = pad_sequence(tokens)
        logger.info(f"small batch_idxs[{step}]={batch_idxs}")

        L, B = batch.shape
        batch = batch.to(device)
        
        logger.info(f"step[{step}] generating...")
        outputs = model.generate2(batch, '[END]', args.max_len, pad_token, tqdm=args.tqdm)
        outputs = [out.cpu().numpy() for out in outputs]

        # Log tokens
        for batch_idx, input, output in zip(batch_idxs, batch.T, outputs):
            token_logger.debug(f"[{step}][{batch_idx}]Input={voc_encoder.decode(input)}")
            token_logger.debug(f"[{step}][{batch_idx}]Output={voc_encoder.decode(output)}")

        logger.debug(f"step[{step}] parsing...")
        for i, output in enumerate(outputs):
            if not sampler.is_remain[idx]: continue
            words = voc_encoder.decode(itr.takewhile(lambda x: x != end_token, output))
            idx = batch_idxs[i]
            error, coords = parse_coord_tokens(words)
            if error == '':
                if len(coords) != ns_atom[i]:
                    error = 'COORD_MISMATCH'
            errors[idx] = error
            if error != '': 
                continue
            np.save(f"{out_dir}/coords/{idx}.npy", coords)
            sampler.is_remain[idx] = False
        logger.info(f"batch_errors={[errors[idx] for idx in batch_idxs]}")
    df = pd.DataFrame({'error': errors})
    df.to_csv(f"{out_dir}/info.csv")
