import sys, os
import argparse
import torch
sys.path.append(os.environ.get('WORKDIR', "/workspace"))
sys.path.append(".")
from src.data.fragment import process_fragment3, LognormalSampler

parser = argparse.ArgumentParser()
parser.add_argument("--root-dir", required=True)
parser.add_argument("--processname", required=True)
parser.add_argument("--reset", action='store_true')
parser.add_argument("--num-workers", type=int)
parser.add_argument("--tqdm", action='store_true')
parser.add_argument("--max-tasks-per-child", type=int)
parser.add_argument("--idx-min", type=int)
parser.add_argument("--idx-sup", type=int)
parser.add_argument("--max-n-atom", type=int)
parser.add_argument("--device", default='cpu')

# num_atom sampler
parser.add_argument("--natom-logmean", type=float)
parser.add_argument("--natom-logstd", type=float)
parser.add_argument("--natom-min", type=float)
parser.add_argument("--natom-max", type=float)

args = parser.parse_args()
args = vars(args)

sampler = LognormalSampler(logmean=args.pop('natom_logmean'), 
    logstd=args.pop('natom_logstd'),
    min=args.pop('natom_min'), max=args.pop('natom_max'))
args['out_dir'] = f"./preprocess/results/pdb_fragment3/{args.pop('processname')}"
device = args.pop('device')
device = torch.device(device) if torch.cuda.is_available() \
    else torch.device('cpu')

process_fragment3(args, sampler=sampler, device=device, **args)
