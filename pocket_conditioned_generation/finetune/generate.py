import sys, os, yaml
from argparse import ArgumentParser
import torch
from addict import Dict
from rdkit import RDLogger
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [f"{WORKDIR}/cplm"]
from src.generate import generate
from src.utils.path import subs_vars
from src.model import Model, MambaModel2
from src.model.mamba import mamba2mamba2

if __name__ == '__main__':

    # Argument
    parser = ArgumentParser()
    ## study
    parser.add_argument("--sname", required=True)
    parser.add_argument("--step", type=int, required=True)
    ## scheme
    parser.add_argument("--genname", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument('--from-data-score', action='store_true')

    ## Environment
    parser.add_argument("--token-per-batch", type=int, help='defaults to value in finetuning')
    parser.add_argument("--gtype", type=int, default=2, choices=[1,2,3])
    parser.add_argument("--max-len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rdir = f"{WORKDIR}/cplm/pocket_conditioned_generation/finetune/{args.genname}/{args.index}/{args.sname}/{args.step}"
    os.makedirs(rdir, exist_ok=True)

    # Load finetuning / training
    ## finetuning
    fdir = f"{WORKDIR}/cplm/finetune/results/{args.sname}"
    fargs = Dict(yaml.safe_load(open(f"{fdir}/config.yaml")))
    fargs = subs_vars(fargs, {'/work/02/ga97/a97003': WORKDIR})

    ## training
    tdir = f"{WORKDIR}/cplm/training/results/{fargs.pretrain_name}"
    targs = Dict(yaml.safe_load(open(f"{tdir}/config.yaml")))

    # 引数の保存
    with open(f"{rdir}/finetune_config.yaml", 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # 引数の作成
    token_per_batch = args.token_per_batch if args.token_per_batch is not None \
        else fargs.token_per_batch
    mamba = targs.get('mamba', False)

    ## prompt_score
    if fargs.no_score:
        assert not args.from_data_score
        prompt_score = 'no_score'
    else:
        if args.from_data_score:
            prompt_score = 'data'
        else:
            prompt_score = 'low'

    ## Vocs from state
    print("Loading state ... ", flush=True)
    state = torch.load(f"{fdir}/models/{args.step}.pth", weights_only=True)
    state = {key[7:]: value for key, value in state.items()}
    if mamba:
        state = mamba2mamba2(state)

    
    state_vocs: list = state['vocs']
    pad_token = state_vocs.index('[PAD]')
    end_token = state_vocs.index('[END]')
    
    if mamba:
        model = MambaModel2(state_vocs, pad_token, end_token)
    else:
        model = Model(8, 768, 12, 4, 0.1, 'gelu', True, state_vocs, pad_token)
    print(model.load_state_dict(state))
    
    generate(model, rdir, token_per_batch, args.seed, args.max_len, args.index,
            fargs.pocket_coord_heavy, fargs.coord_range, prompt_score, state_vocs)