import sys, os, math, yaml
import itertools as itr
from argparse import Namespace, ArgumentParser
import numpy as np, pandas as pd
from tqdm import tqdm
import torch
sys.path += ["/work/gd43/a97003", "/work/gd43/a97003/cplm" ]
from src.utils.utils import solve_increasing_fn_left
from src.train import get_model

sdir = "/work/gd43/a97003/cplm/training/results/260417_ob/tf_atom_valence_coords"
with open(f"{sdir}/args.yaml") as f:
    args = Namespace(**yaml.safe_load(f))
device = torch.device('cpu')

model, voc_encoder = get_model(args, None, f"{sdir}/models/0.pth", device)

for l in range(-3, 1):
    f = lambda bsz: model.get_gpuuse(bsz, l, bf16=True, kernel='FLASH')-args.gpu_size
    b = solve_increasing_fn_left(f, 16)
    print(f"{l}: {b}")