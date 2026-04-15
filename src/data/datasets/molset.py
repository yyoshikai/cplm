import sys, os
from pathlib import Path
from torch.utils.data import Subset
from ..lmdb import StringLMDBDataset, IntLMDBDataset
from ...utils.path import WORKDIR
WORKDIR = str(Path(__file__).parents[4])
MOLSET_DIR = f"{WORKDIR}/tvae/data/moses"

class MolsetWholeDataset(StringLMDBDataset):
    def __init__(self, molset_dir: str=MOLSET_DIR):
        super().__init__(f"{molset_dir}/main.lmdb")

class MolsetDataset(Subset[str]):
    def __init__(self, split: str, molset_dir: str=MOLSET_DIR):
        index_data = IntLMDBDataset(f"{molset_dir}/idxs/{split}.lmdb")
        super().__init__(MolsetWholeDataset(molset_dir), index_data)

