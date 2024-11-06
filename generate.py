import sys, os
import argparse, logging
import math, itertools
from addict import Dict
from tqdm import tqdm
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path.append(WORKDIR)
from tools.logger import add_stream_handler, add_file_handler

from src.model import Model
from src.tokenizer import MoleculeProteinTokenizer

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--studyname", required=True)
parser.add_argument("--step", required=True, type=int)
parser.add_argument("--protein", action='store_true')
parser.add_argument("--protocol-name", default='241019')
parser.add_argument("--token-per-batch", type=int)
parser.add_argument("--n", type=int, default=25)
parser.add_argument("--max-len", type=int, default=1000)
args = parser.parse_args()
sname = args.studyname
step = args.step

# directories
train_dir = f"./training/results/{args.studyname}"
rdir = f"./generate/results/{args.protocol_name}/{args.studyname}/{args.step}/" \
    +("protein" if args.protein else "mol")  
os.makedirs(rdir, exist_ok=True)

# config
config = yaml.safe_load(open(f"{train_dir}/config.yaml"))
coord_range = config.get('coord_range', 20)
token_per_batch = args.token_per_batch if args.token_per_batch is not None \
    else config['token_per_batch']

# environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger()
add_stream_handler(logger)
add_file_handler(logger, f"{rdir}/log.log")

# data
tokenizer = MoleculeProteinTokenizer(-coord_range, coord_range)

# model
model = Model(8, 768, 12, 4, 0.1, 'gelu', True, tokenizer.voc_size, tokenizer.pad_token)
state = torch.load(f"{train_dir}/models/{step}.pth", weights_only=True)
state2 = {key[7:]: value for key, value in state.items()}
logger.info(model.load_state_dict(state2))
model.to(device)
model.eval()
start_token = tokenizer.prot_start_token if args.protein else tokenizer.mol_start_token


# generate tokens
max_len = args.max_len
batch_size = 25000 // max_len
nbatch = math.floor(args.n/batch_size)

outputs = []
for ibatch in range(nbatch):
    is_finished = torch.full((batch_size,), fill_value=False, device=device)
    with torch.no_grad():
        input = torch.full((1, batch_size), fill_value=start_token, dtype=torch.long, device=device) # [L, B]

        for i in tqdm(range(max_len)):
            output = model(input) # [L, B, D]
            prob = F.softmax(output[-1], dim=1) # [B, D]
            output = torch.multinomial(prob, num_samples=1) # [B, 1]
            output = output.view(1, -1) # [1, B]
            is_finished = torch.logical_or(is_finished, output[0] == tokenizer.pad_token)
            input = torch.cat([input, output], dim=0)
            if torch.all(is_finished): break
        outputs += input.cpu().numpy().tolist()
torch.save(input.cpu(), f'{rdir}/token.pkl')


# detokenize
input = torch.load(f'{rdir}/token.pkl', weights_only=True)
os.makedirs(f"{rdir}/coord", exist_ok=True)
with open(f"{rdir}/string.txt", 'w') as f:
    for i in range(batch_size):
        tokens = input[:,i].detach().cpu().numpy().tolist()
        tokens = itertools.takewhile(lambda x: x != tokenizer.pad_token, tokens)
        if args.protein:
            prot, coord = tokenizer.detokenize_protein(tokens)
            f.write((','.join(prot)+'\n') if prot is not None else 'None')
        else:
            smi, coord = tokenizer.detokenize_mol(tokens)
            f.write(smi+'\n')
        if coord is not None:
            pd.DataFrame(coord).to_csv(f"{rdir}/coord/{i}.csv", index=False, header=False)
