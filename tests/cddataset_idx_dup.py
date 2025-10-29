import sys, os
import itertools as itr
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path.append(f"{WORKDIR}/cplm")
from src.utils.lmdb import load_lmdb
dlargs = dict(batch_size=None, shuffle=False, num_workers=16)

# idxが重複していないか そのままだと時間がかかるのでlmdbを直接調べる
idxs = {}
for split in ['valid', 'test', 'train']:
    env, txn = load_lmdb(f"{WORKDIR}/cheminfodata/crossdocked/pockets/mask/{split}_idxs.lmdb", readahead=True)
    idxs[split] = {int.from_bytes(key) for key in txn.cursor().iternext(values=False)}

for split0, split1 in itr.combinations(['train', 'valid', 'test'], 2):
    print(f"{split0=}, {split1=}, {len(idxs[split0])=}, {len(idxs[split1])=}, {len(idxs[split0]&idxs[split1])=}")