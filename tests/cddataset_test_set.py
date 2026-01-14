import sys, os
import itertools as itr
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
WORKDIR = os.environ.get('WORKDIR', "/workspace")
from src.data.datasets.crossdocked import CDDataset
from src.utils.logger import get_logger

logger = get_logger(stream=True)
dlargs = dict(batch_size=None, shuffle=False, num_workers=16)

# test setのdnameがsplit_by_nameと一致しているか？
loader = DataLoader(CDDataset('test'), **dlargs)
test_dnames = {item[3].split('/')[-2] for item in tqdm(itr.islice(loader, 100000))}
split_by_name = torch.load(f"{WORKDIR}/cheminfodata/crossdocked/targetdiff/split_by_name.pt", weights_only=True)
test_dnames0 = {item[0].split('/')[0] for item in split_by_name['test']}
logger.info(f"{len(test_dnames)=}, {len(test_dnames0)=}, {len(test_dnames0&test_dnames)=}")

