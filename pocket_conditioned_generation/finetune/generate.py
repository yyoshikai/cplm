import sys, os
from argparse import ArgumentParser
from rdkit import RDLogger
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [f"{WORKDIR}/cplm"]
from src.generate import generate

def add_generate_args(parser: ArgumentParser):
    parser.add_argument("--score-min", type=float)
    parser.add_argument("--score-max", type=float)
    parser.add_argument("--index", required=True)
    parser.add_argument("--gtype", type=int, default=2, choices=[1,2,3])
    parser.add_argument("--max-len", type=int, default=1000)
    parser.add_argument("--token-per-batch", type=int, help='defaults to value in finetuning')
    parser.add_argument("--seed", type=int, default=0)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--sname", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--genname", required=True)

    add_generate_args(parser)
    args = parser.parse_args()

    # Environment
    RDLogger.DisableLog("rdApp.*")


    # Load finetuning
    fdir = f"{WORKDIR}/cplm/finetune/results/{args.sname}"
    rdir = f"{WORKDIR}/cplm/pocket_conditioned_generation/finetune/{args.genname}/{args.index}/{args.sname}/{args.step}"
    generate(args, rdir, fdir, f"{fdir}/models/{args.step}.pth")