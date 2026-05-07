import sys, os
import argparse
sys.path.append(os.environ.get('WORKDIR', "/workspace"))
sys.path.append(".")
from src.data.pretrain.fragment import PDBFragmentDataset

parser = argparse.ArgumentParser()
parser.add_argument("--root-dir", required=True)
parser.add_argument("--radius-mean", type=float, default=10.0)
parser.add_argument("--radius-std", type=float, default=3.0)
parser.add_argument("--processname", required=True)
parser.add_argument("--num-workers", type=int)
parser.add_argument("--range-min", type=int)
parser.add_argument("--range-sup", type=int)
parser.add_argument("--reset", action='store_true')
parser.add_argument("--tqdm", action='store_true')
parser.add_argument("--max-n-atom", type=int)
parser.add_argument("--max-tasks-per-child", type=int)
args = parser.parse_args()


args = vars(args)
args['out_path'] = f"./preprocess/results/pdb_fragment/{args.pop('processname')+'.lmdb'}"

PDBFragmentDataset.process0(**args)
