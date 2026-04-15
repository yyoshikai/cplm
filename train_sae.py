import yaml
from argparse import ArgumentParser
import torch
from src.train import get_model

parser = ArgumentParser()
parser.add_argument('--studyname', required=True)
parser.add_argument('--studyscript', choices=['training', 'finetune', 'reinforce'], required=True)
parser.add_argument('--opt', type=int)
args = parser.parse_args()

sdir = f"{args.studyscript}/results/{args.studyname}"

with open(f"{sdir}/args.yaml") as f:
    script_args = yaml.safe_load(f)
if args.studyscript == 'reinforce':
    with open(f"finetune/results/{script_args.finetune_name}/args.yaml") as f:
        targs = yaml.safe_load(f)
else:
    targs = script_args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, voc_encoder = get_model(targs, None, f"{sdir}/models/{args.opt}.pth", device)






