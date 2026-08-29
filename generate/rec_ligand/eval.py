import sys, os, math, yaml, logging
import itertools as itr
from contextlib import nullcontext
import concurrent.futures as cf
import multiprocessing as mp
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from logging import getLogger
from glob import glob
import numpy as np, pandas as pd
from openbabel.openbabel import OBConversion
from tqdm import tqdm
from src.utils.path import mwrite
from src.utils.logger import get_logger, add_file_handler
from src.evaluate import eval_vina, eval_qvina
from src.train.data import get_finetune_data
from src.data.molecule import Mol2PDBDataset

def eval_vina2(gdir, idx, t, lig_sdf):
    logger = getLogger("eval_vina")
    print(f"vina[{idx}][{t}] started.", flush=True)
    out_dir = f"{gdir}/eval/{idx}/{t}"
    with open(f"{gdir}/eval/{idx}/rec.pdb") as f:
        rec_pdb = f.read()
    
    vina, min_vina, error = eval_vina(
        lig_sdf=lig_sdf, 
        rec_pdb=rec_pdb, 
        rec_pdbqt_path=f"{out_dir}/rec_pdbqt.pdbqt"
    )
    mwrite(f"{out_dir}/vina_score.txt", str(vina))
    mwrite(f"{out_dir}/min_vina_score.txt", str(min_vina))
    if error is not None:
        mwrite(f"{out_dir}/vina_error.txt", str(error))
    print(f"vina[{idx}][{t}] ended.", flush=True)

def eval_qvina2(gdir, idx, t, lig_sdf):
    out_dir = f"{gdir}/eval/{idx}/{t}"
    logger = getLogger("eval_qvina")
    print(f"qvina[{idx}][{t}] started.", flush=True)
    affinity, e, stdout, stderr = eval_qvina(
        ligand=lig_sdf, 
        rec_pdb_path=f"{gdir}/eval/{idx}/rec.pdb", 
        out_dir=out_dir, 
        cpu=1, 
        timeout=60
    )
    mwrite(f"{out_dir}/qvina_score.txt", str(affinity))
    if e is not None:
        mwrite(f"{out_dir}/qvina_stdout.txt", stdout)
        mwrite(f"{out_dir}/qvina_stderr.txt", stderr)
    print(f"qvina[{idx}][{t}] ended.", flush=True)

def eval_rec_ligand(fargs: Namespace, gdir: str, num_workers: int):
    """
    refについても同じ枠組みで評価したかったので。
    """
    logger = getLogger("eval_rec_ligand")

    # Protein dataset
    rec_data = get_finetune_data(fargs, 'test', sample=1.0, add_ligand=False, random_ligand=False, random_rotate=False, added_vocs=set(), prompt_score='none')[2]
    rec_data = Mol2PDBDataset(rec_data)

    with cf.ProcessPoolExecutor(num_workers) if num_workers > 0 else nullcontext() as e:
        futures = []
        lig_sdf_paths = sorted(glob(f"{gdir}/new_sdf/*/*.sdf"))
        for lig_sdf_path in lig_sdf_paths:
            *_, idx, bname = lig_sdf_path.split('/')
            t = bname.split('.')[0]

            ## set rec
            rpath = f"{gdir}/eval/{idx}/rec.pdb"
            if not os.path.exists(rpath):
                mwrite(rpath, rec_data[int(idx)])
            
            with open(lig_sdf_path) as f:
                lig_sdf = f.read()
            out_dir = f"{gdir}/eval/{idx}/{t}"
            eargs = gdir, idx, t, lig_sdf
            existing_metrics = set()
            for metric in ['vina', 'min_vina', 'qvina']:
                spath = f"{out_dir}/{metric}_score.txt"
                if os.path.exists(spath):
                    with open(spath) as f:
                        if f.read().strip() != 'None':
                            existing_metrics.add(metric)
            if not ({'vina', 'min_vina'} < existing_metrics):
                if e is None:
                    eval_vina2(*eargs)
                else:
                    futures.append(e.submit(eval_vina2, *eargs))
            if 'qvina' not in existing_metrics:
                if e is None:
                    eval_qvina2(*eargs)
                else:
                    futures.append(e.submit(eval_qvina2, *eargs))
        if e is not None:
            for f in futures:
                f.result()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sname", required=True)
    parser.add_argument("--opt", required=True)
    parser.add_argument("--num-workers", type=int)
    args = parser.parse_args()
    gdir = f"generate/rec_ligand/{args.sname}/{args.opt}"
    obc = OBConversion()
    obc.SetInFormat('pdbqt')

    # Environment
    os.makedirs(f"{gdir}/eval", exist_ok=True)
    logger = get_logger(stream=True)
    add_file_handler(logger, f"{gdir}/eval.log", mode='a')

    ## load training args
    with open(f"reinforce/results/{args.sname}/args.yaml") as f:
        rargs = Namespace(**yaml.safe_load(f))
    with open(f"finetune/results/{rargs.finetune_name}/args.yaml") as f:
        fargs = Namespace(**yaml.safe_load(f))

    eval_rec_ligand(fargs, gdir, args.num_workers)
