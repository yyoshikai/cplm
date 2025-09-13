import os
from copy import copy

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset

from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem import rdDepictor

WORKDIR = os.environ.get('WORKDIR', __file__.split('/cplm/')[0])
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
