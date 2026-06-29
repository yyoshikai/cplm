"""
finetuneで, いらないcheckpointを消す
残すもの:
    1000 step単位の場合:
        0~10000
        その後, 10000単位
        最後
"""
import sys, os, yaml
from glob import glob
from argparse import ArgumentParser, Namespace

ck_paths = glob("finetune/results/**/models/*.pth", recursive=True)
fnames = sorted({path.rsplit('/', maxsplit=2)[0].removeprefix('finetune/results/') for path in ck_paths})
for sname in fnames:
    fdir = f"finetune/results/{sname}"

    paths = glob(f"{fdir}/models/*.pth")
    steps = sorted([int(path.split('/')[-1].split('.')[0]) for path in paths])
    max_step = max(steps)
    if max_step >= 40000:
        save_steps = list(range(0, 5001, 1000))+list(range(10000, max_step+1, 10000))
    elif max_step >= 20000:
        save_steps = list(range(0, 5001, 1000))+list(range(10000, max_step+1, 5000))
    elif max_step >= 4000:
        save_steps = list(range(0, 501, 100))+list(range(1000, max_step+1, 1000))
    elif max_step >= 2000:
        save_steps = list(range(0, 501, 100))+list(range(1000, max_step+1, 500))
    else:
        save_steps = list(range(0, max_step+1, 100))
    save_steps.append(max_step)
    for step in steps:
        if step not in save_steps:
            os.remove(f"{fdir}/models/{step}.pth")
            # print(f"{fdir}/models/{step}.pth")


