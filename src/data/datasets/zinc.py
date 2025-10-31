import sys, os
from pathlib import Path
from torch.utils.data import Subset
from ..lmdb import StringLMDBDataset, IntLMDBDataset
from ...utils.path import WORKDIR
ZINC_DIR = f"{WORKDIR}/cheminfodata/zinc/251030/ZINC20-2D"

class ZincDataset(Subset[str]):
    def __init__(self, split: str):
        index_data = IntLMDBDataset(f"{ZINC_DIR}/idxs/{split}.lmdb")
        super().__init__(StringLMDBDataset(f"{ZINC_DIR}/smi.lmdb"), index_data)
