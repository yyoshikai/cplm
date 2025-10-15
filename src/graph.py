import sys, os, math
import itertools as itr
import yaml
import numpy as np, pandas as pd
from addict import Dict
from tqdm import tqdm
sys.path += ["/workspace", "/workspace/cplm" ]
from src.utils.logger import get_logger
logger = get_logger(stream=True)

import matplotlib.pyplot as plt
from tools.graph import get_scatter_style, COLORS2, tightargs, get_grid

def compare_loss(fname, snames, slabels, script):
    fig, ax = plt.subplots(1,1,figsize=(6,4))
    for si, sname in enumerate(snames):
        dfopt = pd.read_csv(f"/workspace/cplm/{script}/results/{sname}/opts/0.csv")
        color, marker = get_scatter_style(si)
        ax.scatter(dfopt.index, dfopt['loss']/dfopt['weight'], color=color, marker=marker, s=1)

    for si, sname in enumerate(snames):
        dfval = pd.read_csv(f"/workspace/cplm/{script}/results/{sname}/vals/0.csv", header=[0,1])
        color, marker = get_scatter_style(si)
        ax.scatter(dfval['opt'].values.ravel(), dfval['mean_loss'].values.ravel(), 
            color=color, marker=marker, s=30, label=slabels[si])
    ax.legend()
    fpath = f"graph/compare_loss/{fname}.png"
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    ax.set_yscale('log')
    fig.savefig(fpath, **tightargs)
    plt.close(fig)

def compare_loss_train(fname, snames, slabels):
    return compare_loss(fname, snames, slabels, 'training')

def compare_loss_finetune(fname, snames, slabels):
    return compare_loss(fname, snames, slabels, 'finetune')

