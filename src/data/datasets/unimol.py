import sys, os
from typing import Optional
from logging import getLogger
import numpy as np, pandas as pd
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import Conformer
from rdkit.Geometry import Point3D
from ..lmdb import PickleLMDBDataset
from ..protein import Protein

class UniMolLigandDataset(Dataset[Chem.Mol]):
    logger = getLogger(f'{__module__}.{__qualname__}')
    def __init__(self, lmdb_path, n_conformer, sample_save_dir: Optional[str]=None):
        self.net_dataset = PickleLMDBDataset(lmdb_path, idx_to_key='str')
        self.n_conformer = n_conformer
        self.getitem_count = 0
        self.sample_save_dir = sample_save_dir

    def __getitem__(self, idx) -> Chem.Mol:
        mol_idx, conformer_idx = divmod(idx, self.n_conformer)
        data = self.net_dataset[mol_idx]

        smi = data['smi']
        coord: np.ndarray = data['coordinates'][conformer_idx]
        coord = coord.astype(float)

        # Generate mol with conformer
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        n_atom  = mol.GetNumAtoms()
        # rdkitのバージョンにより水素の数が違う場合, 重原子の座標から水素の座標を推定する。
        # experiments/241202_241201_mol_pocket5_debugの `2. 原子のconformerを追加する方法を調べる。`より。
        if n_atom != len(coord):
            mol_heavy = Chem.RemoveHs(mol)
            n_heavy = mol_heavy.GetNumAtoms()
            conf = Conformer(n_heavy)
            for i in range(n_heavy):
                conf.SetAtomPosition(i, Point3D(*coord[i]))
            mol_heavy.AddConformer(conf)
            mol = Chem.AddHs(mol_heavy, addCoords=True)
        else:
            conf = Conformer(n_atom)
            for i in range(n_atom):
                conf.SetAtomPosition(i, Point3D(*coord[i]))
            mol.AddConformer(conf)

        # save sample
        if self.sample_save_dir is not None and self.getitem_count < 5:
            save_dir = f"{self.sample_save_dir}/{idx}"
            os.makedirs(save_dir, exist_ok=True)
            with open(f"{save_dir}/data_smi.txt", 'w') as f:
                f.write(data['smi'])
            pd.DataFrame(data['coordinates'][conformer_idx]) \
                .to_csv(f"{save_dir}/data_coord.csv", header=False, index=False)
        self.getitem_count += 1
        return mol
    
    def __len__(self):
        return len(self.net_dataset) * self.n_conformer

class UniMolPocketDataset(Dataset[Protein]):
    def __init__(self, lmdb_path, **kwargs):
        self.dataset = PickleLMDBDataset(lmdb_path, **kwargs)
    
    def __getitem__(self, idx) -> Protein:
        data = self.dataset[idx]
        atoms = np.array(data['atoms'])
        coord =  data.pop('coordinates')[0] # * np.array([0, 1, 2])
        return Protein(atoms=atoms, coord=coord)

    def __len__(self):
        return len(self.dataset)
