import sys, os
import itertools as itr
import torch
from torch.utils.data import DataLoader
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path.append(f"{WORKDIR}/cplm")
from src.data.datasets.crossdocked import CDDataset
from src.utils.logger import get_logger

logger = get_logger(stream=True)
dlargs = dict(batch_size=None, shuffle=False, num_workers=16)
# indexが重複していないか？
datas = {}
idxs = {}
for split in ['valid', 'test', 'train']:
    datas[split] = CDDataset(split)
    logger.info(f"{split=}, {len(datas[split])=}")
    idxs[split] = list(DataLoader(datas[split].indices, **dlargs))

for split0, split1 in itr.combinations(['train', 'valid', 'test'], 2):
    idx0 = idxs[split0]
    idx1 = idxs[split1]
    logger.info(f"{split0=}, {split1=}, {len(idx0)=}, {len(idx1)=}, {len(idx0&idx1)=}, {len(idx0)|len(idx1)=}")

# test setのdnameがsplit_by_nameと一致しているか？
loader = DataLoader(datas['test'], **dlargs)
test_dnames = {item[3].split('/')[-2] for item in loader}
split_by_name = torch.load("/workspace/cheminfodata/crossdocked/targetdiff/split_by_name.pt", weights_only=True)
test_dnames0 = {item[0].split('/')[0] for item in split_by_name['test']}
logger.info(f"{len(test_dnames)=}, {len(test_dnames0)=}, {len(test_dnames0&test_dnames)=}")

# (scaffoldの調査は面倒なので省略)


