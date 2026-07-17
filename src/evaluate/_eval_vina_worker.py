import sys, os
from argparse import ArgumentParser
from .evaluate import _eval_vina, DELIM

parser = ArgumentParser()
parser.add_argument('rec_pdbqt_path')
args = parser.parse_args()

input_str = sys.stdin.read()

lig_sdf, rec_pdb = input_str.split(DELIM)

score, min_score = _eval_vina(lig_sdf, rec_pdb, args.rec_pdbqt_path)
print(score, min_score)

