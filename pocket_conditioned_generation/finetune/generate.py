import sys, os, yaml
from argparse import ArgumentParser
from glob import glob
from addict import Dict
import torch
from torch.nn.utils.rnn import pad_sequence
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [f"{WORKDIR}/cplm"]
from src.utils.path import subs_vars
from src.data.tokenizer import StringTokenizer
from src.finetune import get_finetune_data
from src.train import get_model
from src.generate import generate

if __name__ == '__main__':
    # Argument
    parser = ArgumentParser()
    ## study
    parser.add_argument("--sname", required=True)
    parser.add_argument("--opt", type=int)
    ## generation
    parser.add_argument('--n-trial', type=int, default=1)
    parser.add_argument("--max-len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--genname")
    ## Environment
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()

    logs = []
    
    # finetuning / training args
    ## finetuning
    fdir = f"{WORKDIR}/cplm/finetune/results/{args.sname}"
    fargs = Dict(yaml.safe_load(open(f"{fdir}/args.yaml")))
    fargs = subs_vars(fargs, {'/work/02/ga97/a97003': WORKDIR})

    # default args    
    ## opt
    if args.opt is None:
        opts = [int(path.split('/')[-1].split('.')[0]) for path in glob(f"{fdir}/models/*")]
        args.opt = max(opts)
        logs.append(f"opt was set to {args.opt}")
    ## genname
    if args.genname is None:
        args.genname = f"{args.n_trial}_{args.max_len}_{args.seed}"

    # Environment
    rdir = f"{WORKDIR}/cplm/pocket_conditioned_generation/finetune/{args.genname}/{args.sname}/{args.opt}"
    os.makedirs(rdir, exist_ok=True)

    ## check if result exists
    if os.path.exists(f"{rdir}/info.csv"):
        print(f"{rdir} already finished.", flush=True)
        sys.exit()
    ## device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    added_vocs = StringTokenizer(open(f"{WORKDIR}/cplm/src/data/smiles_tokens.txt").read().splitlines()).vocs()
    voc_encoder, _raw, prompt_token_data, _weight, center_data, _rotation_data, \
        _protein_filename_data, _ligand_filename_data \
        = get_finetune_data(fargs, 'test', False, True, added_vocs)
    def collate_fn(batch):
        indices, batch, centers = list(zip(*batch))
        batch = pad_sequence(batch, padding_value=voc_encoder.pad_token)
        return indices, batch, centers
    
    # model
    model = get_model(fargs, voc_encoder, f"{fdir}/models/{args.opt}.pth", device)
    
    # generate
    generate(model, voc_encoder, prompt_token_data, center_data, rdir, args.n_trial, args.batch_size, args.seed, args.max_len)
