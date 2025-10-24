import pickle
from tqdm import tqdm
from src.utils.lmdb import load_lmdb

env, txn = load_lmdb("/workspace/cplm/ssd/preprocess/results/finetune/r4_all/main.lmdb")

sizes = []
for data in tqdm(txn.cursor().iternext(keys=False)):
    data = pickle.loads(data)
    sizes.append(len(data['pocket_atoms']))
    if len(sizes) == 100000: break

import numpy as np
print(np.histogram(sizes))