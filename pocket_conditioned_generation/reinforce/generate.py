import sys, os
from argparse import ArgumentParser
import yaml
from addict import Dict
PROJ_DIR = "/workspace/cplm"
sys.path.append(PROJ_DIR)
from src.generate import pocket_conditioned_generate, add_pocket_conditioned_generate_args

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
    pocket_conditioned_generate(args, rdir, fdir, model_path)

