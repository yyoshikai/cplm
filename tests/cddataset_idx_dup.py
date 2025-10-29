import sys, os
import itertools as itr
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path.append(f"{WORKDIR}/cplm")
from src.data.datasets.crossdocked import CDDataset
from src.data import ApplyDataset
from src.utils.logger import get_logger

logger = get_logger(stream=True)
dlargs = dict(batch_size=None, shuffle=False, num_workers=16)

# test setのdnameがsplit_by_nameと一致しているか？
idxs = {}
for split in ['valid', 'test', 'train']:
    indices = CDDataset(split).indices
    idxs[split] = set(tqdm(DataLoader(indices, **dlargs)))

for split0, split1 in itr.combinations(['train', 'valid', 'test'], 2):
    print(f"{split0=}, {split1=}, {len(idxs[split0])=}, {len(idxs[split1])=}, {len(idxs[split0]&idxs[split1])=}, ")