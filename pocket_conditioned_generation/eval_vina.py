import sys, os, logging
import concurrent.futures as cf
from functools import partial
from glob import glob
from argparse import ArgumentParser
from pathlib import Path

import numpy as np, pandas as pd, yaml
from tqdm import tqdm
from openbabel import pybel

GEN_DIR = Path(__file__).resolve().parent
PROJ_DIR = GEN_DIR.parent
WORKDIR = PROJ_DIR.parent

sys.path.append(str(PROJ_DIR))
from src.utils.logger import add_stream_handler, get_logger
from src.evaluate import eval_vina as _eval_vina, eval_qvina as _eval_qvina
from src.utils.path import cleardir
from src.data.datasets.targetdiff import TargetDiffScafCDDataset

logger = get_logger()
add_stream_handler(logger, logging.INFO)
pybel.ob.obErrorLog.SetOutputLevel(0)

parser = ArgumentParser()
parser.add_argument('--gname', nargs='*')
parser.add_argument('--metric', choices={'vina', 'qvina'}, default='vina')
args = parser.parse_args()

if args.gname is None:
    args.gname = []
    logger.info("Finding results...")
    prefix, postfix = f"{GEN_DIR}/", "/info.csv"
    paths = glob(f"{prefix}**{postfix}", recursive=True)
    gnames = [path[len(prefix):][:-len(postfix)] for path in paths]
    for gname in gnames:
        gdir = f"{prefix}{gname}"

        # check if result exists
        if os.path.exists(f"{gdir}/{args.metric}_score.csv"):
            continue
        
        args.gname.append(gname)
    logger.info(f"Found {len(args.gname)} generations.")

def eval_vina(gname):
    gdir = f"{GEN_DIR}/{gname}"
    data = TargetDiffScafCDDataset('test')
    dfg = pd.read_csv(f"{gdir}/info.csv", index_col=0, keep_default_na=False)
    is_ = np.where(dfg['error'] == '')[0]
    if args.metric == 'vina':
        metric = _eval_vina
    else:
        metric = partial(_eval_qvina, timeout=300)
    cleardir(f"{gdir}/eval_{args.metric}")
    scores = []
    with cf.ProcessPoolExecutor(28) as e:
        fs = []
        for i in is_:
            didx=dfg.loc[i, 'idx']

            fs.append(e.submit(metric, lig_path=f"{gdir}/sdf/{i}.sdf", 
                    rec_path=data[didx][3], 
                    out_dir=f"{gdir}/eval_{args.metric}/{i}"))
        scores += [f.result() for f in tqdm(fs)]

    if args.metric == 'vina':
        df = pd.DataFrame(scores, columns=['score', 'min_score'], index=is_)
    else:
        df = pd.DataFrame({'score': scores}, index=is_)
    df.to_csv(f"{gdir}/{args.metric}_score.csv")

for gname in args.gname:
    logger.info(f"evaluating {gname} started.")
    eval_vina(gname)
