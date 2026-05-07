"""
finetune.pyと統合した方がよい？
"""
import sys, os, logging, argparse, pandas as pd, numpy as np
import concurrent.futures as cf
from tqdm import tqdm
sys.path.append(".")
from src.utils.lmdb import new_lmdb, load_lmdb

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--size", type=int, required=True)
args = parser.parse_args()

dfs = []
out_path = f"preprocess/results/finetune/{args.output}/main.lmdb"
env, txn = new_lmdb(out_path, map_size=1000000000000)

start_idx = 0
for i in range(args.size):
    env0, txn0 = load_lmdb(f"preprocess/results/finetune/{args.output}/{i}/main.lmdb", 
        readahead=True)
    L = env0.stat()['entries']
    for key0, data0 in tqdm(txn0.cursor().iternext(), total=env0.stat()['entries']):
        key = str(int(key0.decode('ascii'))+start_idx).encode('ascii')
        txn.put(key, data0)
    df = pd.read_csv(f"preprocess/results/finetune/{args.output}/{i}/filenames.csv", 
        index_col=0, dtype=str, keep_default_na=False)
    df.index += start_idx
    dfs.append(df)
    start_idx += L
df = pd.concat(dfs, axis=0)
df.to_csv(f"preprocess/results/finetune/{args.output}/filenames.csv") 
assert np.all(df.index == np.arange(len(df), dtype=int))
txn.commit()
env.close()
