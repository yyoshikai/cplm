import sys, os
import argparse

import torch

from src.data import LMDBDataset, MoleculeTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--train-data")
parser.add_argument("--valid-data")
args = parser.parse_args()

# environments
device = torch.device()




















