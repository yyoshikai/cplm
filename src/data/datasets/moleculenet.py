import os, pickle, io, logging, yaml
from logging import getLogger
from collections import defaultdict
from time import time
from torch.utils.data import Dataset, Subset

import numpy as np
import pandas as pd
from prody import parsePDB, parsePDBStream, confProDy, Contacts, addMissingAtoms
from ...utils.lmdb import new_lmdb
from ..lmdb import PickleLMDBDataset
from ..data import WrapDataset
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem import rdDepictor
from ...utils.logger import add_file_handler, get_logger
from ...utils.rdkit import ignore_warning
from ...utils.utils import CompressedArray
from ..protein import Protein
from copy import copy

WORKDIR = os.environ.get('WORKDIR', "/workspace")
MOLNET_DIR = f"{WORKDIR}//cheminfodata/molnet"

class MoleculeNetDataset(Dataset[tuple[Chem.Mol, float]]):
    def __init__(self, dataset_name: str, task_name: str):
        with open(f"{MOLNET_DIR}/strict/{dataset_name}/canonical.txt") as f:
            self.smiles = f.read().splitlines()
        self.target = pd.read_csv(f"{MOLNET_DIR}/strict/{dataset_name}/info.csv")[task_name].values.astype(float)

    def __getitem__(self, idx: int):

        mol = Chem.MolFromSmiles(self.smiles[idx])
        rdDepictor.Compute2DCoords(mol)
        assert mol.GetNumConformers() >= 1, mol
        mol = Chem.AddHs(mol, addCoords=True)
        mol_2d = copy(mol)
        
        
        rdDistGeom.EmbedMolecule(mol)
        if mol.GetNumConformers() >= 1:
            pass
        else:
            mol = mol_2d
        # assert mol.GetNumConformers() >= 1, mol
        return mol, self.target[idx]

    def __len__(self, idx):
        return len(self.target)
    

class MoleculeNetTrainDataset(Subset[tuple[Chem.Mol, float]]):
    def __init__(self, dataset_name: str, task_name: str):
        folds = pd.read_csv(f"{MOLNET_DIR}/strict/{dataset_name}/info2.csv")['gem_sfold']
        idxs = np.where(folds == 0)[0]
        dataset = MoleculeNetDataset(dataset_name, task_name)
        super().__init__(dataset, idxs)
