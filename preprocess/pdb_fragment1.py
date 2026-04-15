import sys, os
from logging import getLogger
import argparse
from collections import defaultdict
from tqdm import tqdm
import yaml, psutil
from addict import Dict
import numpy as np, pandas as pd
import concurrent.futures as cf
sys.path.append(os.environ.get('WORKDIR', "/workspace"))
sys.path.append(".")
from src.data.pretrain.protein import PDBFragmentDataset
from tools.logger import add_stream_handler

parser = argparse.ArgumentParser()
parser.add_argument("--processname", required=True)
parser.add_argument("--tqdm", action='store_true')
args = parser.parse_args()


args = vars(args)
args['out_path'] = f"./preprocess/results/pdb_fragment/{args.pop('processname')+'.lmdb'}"
logger = getLogger()
add_stream_handler(logger)
PDBFragmentDataset.process1(**args)
