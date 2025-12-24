import sys, os, yaml, math
from argparse import ArgumentParser, Namespace
from glob import glob
from addict import Dict
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [f"{WORKDIR}/cplm"]
from src.utils.path import subs_vars
from src.generate import generate, add_generate_args

if __name__ == '__main__':
    
    parser = ArgumentParser()
    add_generate_args(parser)
    parser.add_argument("--sname", required=True)
    parser.add_argument("--opt", type=int)
    parser.add_argument("--genname")
    parser.add_argument("--tqdm-generate", action='store_true')
    parser.add_argument("--max-prompt-len", type=int, default=math.inf)
    args = parser.parse_args()
    
    rdir = f"{WORKDIR}/cplm/reinforce/results/{args.sname}"
    rargs = Dict(yaml.safe_load(open(f"{rdir}/args.yaml")))
    rargs = subs_vars(rargs, {'/work/02/ga97/a97003': WORKDIR})
    fdir = f"{WORKDIR}/cplm/finetune/results/{rargs.finetune_name}"
    fargs = Namespace(**yaml.safe_load(open(f"{fdir}/args.yaml")))
    if args.opt is None:
        opts = [int(path.split('/')[-1].split('.')[0]) for path in glob(f"{rdir}/models/*")]
        args.opt = max(opts)
    if args.genname is None:
        args.genname = f"{args.n_trial}_{args.max_len}_{args.seed}"
    odir = f"{WORKDIR}/cplm/pocket_conditioned_generation/reinforce/{args.genname}/{args.sname}/{args.opt}"

    # generate
    generate(odir, args.n_trial, args.batch_size, args.seed, args.max_len, fargs, f"{rdir}/models/{args.opt}.pth", True, args.tqdm_generate, args.max_prompt_len)
