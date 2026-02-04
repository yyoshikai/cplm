import sys, os, math, yaml
import itertools as itr
import concurrent.futures as cf
from argparse import ArgumentParser
from collections import defaultdict
from logging import getLogger
from glob import glob
import numpy as np, pandas as pd
from openbabel.openbabel import OBConversion, OBMol
from tqdm import tqdm
from src.utils.path import mwrite
from src.utils.logger import get_logger, add_file_handler
from src.evaluate import eval_vina, eval_qvina

def eval_vina2(out_dir, i, t, lig_sdf):
    logger = getLogger()
    logger.info(f"vina[{i}][{t}]started.")
    with open(f"{out_dir}/prompt_rec_pdb/{i}/{t}.pdb") as f:
        rec_pdb = f.read()
    vina, min_vina, error = eval_vina(
        ligand=lig_sdf, 
        rec=rec_pdb, 
        rec_pdbqt_path=f"{out_dir}/eval/rec_pdbqt/{i}/{t}.pdbqt"
    )
    mwrite(f"{out_dir}/eval/vina_score/{i}/{t}.txt", str(vina))
    mwrite(f"{out_dir}/eval/min_vina_score/{i}/{t}.txt", str(min_vina))
    if error is not None:
        mwrite(f"{out_dir}/eval/vina_error/{i}_{t}.txt", str(error))
    logger.info(f"vina[{i}][{t}]finished.")

def eval_qvina2(out_dir, i, t, lig_sdf):
    logger = getLogger()
    logger.info(f"qvina[{i}][{t}]started.")
    affinity, e, stdout, stderr = eval_qvina(
        ligand=lig_sdf, 
        rec_pdb_path=f"{out_dir}/prompt_rec_pdb/{i}/{t}.pdb", 
        out_dir=f"{out_dir}/eval/qvina_out/{i}/{t}", 
        cpu=1, 
        timeout=60
    )
    mwrite(f"{out_dir}/eval/qvina_score/{i}/{t}.txt", str(affinity))
    if e is not None:
        mwrite(f"{out_dir}/eval/qvina_stdout/{i}_{t}.txt", str(affinity))
        mwrite(f"{out_dir}/eval/qvina_stderr/{i}_{t}.txt", str(affinity))
    logger.info(f"qvina[{i}][{t}]finished.")

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("--gname", required=True)
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()
    out_dir = f"generate/rec_ligand/{args.gname}"
    obc = OBConversion()
    obc.SetInFormat('pdbqt')
    N, T = 100, 5

    if not os.path.exists(f"{out_dir}/generate.log"): # temp
        raise FileNotFoundError(out_dir)

    add_file_handler(get_logger(stream=True), f"{out_dir}/eval.log", mode='a')

    with cf.ProcessPoolExecutor(args.num_workers) as e:
        futures = []
        for lig_sdf_path in sorted(glob(f"{out_dir}/new_sdf/*/*.sdf")):
            *_, i, bname = lig_sdf_path.split('/')
            t = bname.split('.')[0]
            with open(lig_sdf_path) as f:
                lig_sdf = f.read()
            if not os.path.exists(f"{out_dir}/eval/vina_score/{i}/{t}.txt") \
                    or not os.path.exists(f"{out_dir}/eval/min_vina_score/{i}/{t}.txt"):
                futures.append(e.submit(eval_vina2, out_dir, i, t, lig_sdf))
            if not os.path.exists(f"{out_dir}/eval/qvina_score/{i}/{t}.txt"):
                futures.append(e.submit(eval_qvina2, out_dir, i, t, lig_sdf, ))
        for f in futures:
            f.result()
    
    os.makedirs(f"{out_dir}/eval", exist_ok=True)
    for metric in ['vina', 'min_vina', 'qvina']:
        data = defaultdict(dict)
        for i, t in itr.product(range(N), range(T)):
            path = f"{out_dir}/eval/{metric}_score/{i}/{t}.txt"
            if os.path.exists(path):
                with open(path) as f:
                    data[t][i] = eval(f.read())
            else:
                data[t][i] = None
        pd.DataFrame(data).to_csv(f"{out_dir}/eval/{metric}_score.csv")
