"""
251028 前処理のコードは消したので, 古いcommitを参照してください。
"""
import os
from logging import getLogger
from typing import Literal
from time import time

import pandas as pd
from prody import parsePDB, confProDy, addMissingAtoms
from ..lmdb import PickleLMDBDataset, IntLMDBDataset, data_len_to_blen
from ..data import WrapDataset, TupleDataset, Subset
from rdkit import Chem
confProDy(verbosity='none')
from ...utils.utils import CompressedArray
from ...utils.path import WORKDIR
from ..protein import Protein

SAVE_DIR = f"{WORKDIR}/cplm/ssd/preprocess/results/finetune/r4_all"
CDDIR = f"{WORKDIR}/cheminfodata/crossdocked"

class CDWholeDataset(TupleDataset[tuple[Protein, Chem.Mol, float, str, str]]):
    def __init__(self):
        self.raw_data = PickleLMDBDataset(f"{CDDIR}/pockets/main.lmdb")
        super().__init__(5)

    def __getitem__(self, idx):
        data = self.raw_data[idx]

        # pocket
        pocket_atoms, pocket_coord = data['pocket_atoms'], data['pocket_coordinate']
        pocket = Protein(pocket_atoms, pocket_coord)
        
        # path
        ligand_path = f"{CDDIR}/CrossDocked2020/{data['dname']}/{data['lig_name']}"
        protein_path = f"{CDDIR}/CrossDocked2020/{data['dname']}/{data['protein_name']}"

        score = float(data['score'])
        return pocket, data['lig_mol'], score, protein_path, ligand_path
    
    def __len__(self):
        return len(self.raw_data)

class CDDataset(TupleDataset[tuple[Protein, Chem.Mol, float, str, str]]):
    def __init__(self, split: Literal['train', 'valid', 'test']):
        self.indices = IntLMDBDataset(f"{CDDIR}/pockets/mask/{split}_idxs.lmdb")
        self.dataset = CDWholeDataset()
    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]
    def __len__(self):
        return len(self.indices)
    
class CDProteinWholeDataset(TupleDataset[tuple[Protein, Chem.Mol, float, str, str]]):
    def __init__(self):
        self.raw_data = PickleLMDBDataset(f"{CDDIR}/pockets/main.lmdb")
        super().__init__(5)
        os.makedirs("./tmp", exist_ok=True)

    def __getitem__(self, idx):
        data = self.raw_data[idx]
        
        # path
        ligand_path = f"{CDDIR}/CrossDocked2020/{data['dname']}/{data['lig_name']}"
        protein_path = f"{CDDIR}/CrossDocked2020/{data['dname']}/{data['protein_name']}"
        tmp_path = f"./tmp/{idx}_{time()}.pdb"
        
        addMissingAtoms(protein_path, outfile=tmp_path)
        protein = parsePDB(tmp_path)
        protein = Protein(protein.getData('name'), protein.getCoords())

        score = float(data['score'])
        os.remove(tmp_path)
        return protein, data['lig_mol'], score, protein_path, ligand_path

class CDProteinDataset(TupleDataset[tuple[Protein, Chem.Mol, float, str, str]]):
    def __init__(self, split: Literal['train', 'valid', 'test']):
        self.indices = IntLMDBDataset(f"{CDDIR}/pockets/mask/{split}_idxs.lmdb")
        self.dataset = CDProteinWholeDataset()
    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]
    def __len__(self):
        return len(self.indices)
