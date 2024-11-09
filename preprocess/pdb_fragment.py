import sys, os
import math, random, itertools, pickle
from logging.config import dictConfig
import argparse
from collections import defaultdict
from tqdm import tqdm
import yaml, psutil
from addict import Dict
import numpy as np, pandas as pd
import concurrent.futures as cf
sys.path.append(os.environ.get('WORKDIR', "/workspace"))
sys.path.append(".")
from src.data.protein import PDBFragmentDataset
from tools.logger import get_log_config

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

# dictConfig(get_log_config())
PDBFragmentDataset.process0(**args)
