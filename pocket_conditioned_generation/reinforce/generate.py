import sys, os
from argparse import ArgumentParser
import yaml
from addict import Dict
import torch
PROJ_DIR = "/workspace/cplm"
sys.path.append(PROJ_DIR)
from src.generate import pocket_conditioned_generate, add_pocket_conditioned_generate_args
from src.utils.path import subs_vars
from src.model import Model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--sname')
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--genname', required=True)
    add_pocket_conditioned_generate_args(parser)
    args = parser.parse_args()

    reinforce_dir = f"{PROJ_DIR}/reinforce/results/{args.sname}"
    model_path = f"{reinforce_dir}/models/{args.step}.pth"
    with open(f"{reinforce_dir}/config.yaml") as f:
        rargs = Dict(yaml.safe_load(f))
    fdir = f"{PROJ_DIR}/finetune/results/{rargs.finetune_name}"
    rdir = f"{PROJ_DIR}/pocket_conditioned_generation/reinforce/results/{args.genname}/{args.sname}/{args.step}/{args.index}"

    
    fargs = Dict(yaml.safe_load(open(f"{fdir}/config.yaml")))
    fargs = subs_vars(fargs, {'/work/02/ga97/a97003': '/workspace'})

    token_per_batch = args.token_per_batch if args.token_per_batch is not None else fargs.token_per_batch
    
    state = torch.load(model_path, weights_only=True)
    state = {key[7:]: value for key, value in state.items()}
    state_vocs = state['vocs']


    pad_token = state_vocs.index('[PAD]')
    end_token = state_vocs.index('[END]')
    model = Model(8, 768, 12, 4, 0.1, 'gelu', True, state_vocs, pad_token)
    print(model.load_state_dict(state))

    pocket_conditioned_generate(model, rdir, model_path, token_per_batch, args.seed, args.max_len, args.index, fargs.pocket_coord_heavy, fargs.coord_range, 'no_score', state_vocs)

