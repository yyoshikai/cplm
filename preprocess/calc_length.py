import sys, os
import argparse
import concurrent.futures as cf
import pickle
import lmdb
from src.data import MoleculeTokenizer, LMDBDataset

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output")
parser.add_argument("--key-is-indexed", action='store_true')
parser.add_argument("--overwrite", action='store_true')
parser.add_argument("--max-workers", type=int, default=1)
args = parser.parse_args()
if args.output is None:
    args.output = os.path.splitext(args.input)+'_len.lmdb'
if os.path.exists(args.output):
    if args.overwrite:
        os.remove(args.output)
    else:
        raise FileExistsError(f"`--output` file already exists.")
chunksize = 100

def calc_lengths(path, key_is_indexed, idx_start, idx_end):

    dataset = LMDBDataset(path, key_is_indexed=key_is_indexed)
    tokenizer = MoleculeTokenizer()

    lengths = []
    for idx in range(idx_start, idx_end):
        data = dataset[idx]
        lengths.append(pickle.dumps(tokenizer.get_length(data['smi'], data['coord'])))
    return lengths

in_data = LMDBDataset(args.input, key_is_indexed=args.key_is_indexed)
in_size = len(in_data)

out_env = lmdb.open(args.output, subdir=False, readonly=False,
    lock=False, readahead=False, meminit=False, max_readers=1, map_size=int(100e9))
out_txn = out_env.begin(write=True)

print(f"Processing with {args.max_workers} workers...")
with cf.ProcessPoolExecutor(args.max_workers) as e:
    futures = []
    for idx_start in range(0, in_size, chunksize):
        futures.append(e.submit(calc_lengths, 
            path=args.input, key_is_indexed=args.key_is_indexed, 
            idx_start=idx_start, idx_end=min(idx_start+chunksize, in_size)))
    
    idx = 0
    for f in futures:
        for l in f.result():
            out_env.put(in_data.key(idx), l)
            idx+=1

out_txn.commit()
out_env.close()


