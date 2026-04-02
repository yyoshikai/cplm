"""
finetuneで, いらないcheckpointを消す
残すもの:
    1000 step単位の場合:
        0~5000
        その後, 10000単位
        最後
"""
import sys, os, yaml
from glob import glob
from argparse import ArgumentParser, Namespace

parser = ArgumentParser()
parser.add_argument("--studyname", nargs='*')
args = parser.parse_args()
for sname in args.studyname:
    fdir = f"finetune/results/{sname}"
    with open(f"{fdir}/args.yaml" if os.path.exists(f"{fdir}/args.yaml") 
            else f"{fdir}/config.yaml") as f:
        args = Namespace(**yaml.safe_load(f))
    freq = args.record_opt_step

    paths = glob(f"{fdir}/models/*.pth")
    steps = sorted([int(path.split('/')[-1].split('.')[0]) for path in paths])
    for step in steps:
        if step > freq*5 and step % (freq*10) != 0 and step != max(steps):
            os.remove(f"{fdir}/models/{step}.pth")


