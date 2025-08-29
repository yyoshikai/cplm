import os, random
from typing import Optional
import numpy as np, pandas as pd
from torch.utils.data import Dataset
from rdkit import Chem
from .data import WrapDataset

class MolProcessDataset(WrapDataset[tuple[str, np.ndarray]]):
    def __init__(self, mol_data: Dataset[Chem.Mol], rstate: np.random.RandomState, h_atom: bool, h_coord: bool, randomize: bool, sample_save_dir: Optional[str]=None):
        super().__init__(mol_data)
        self.mol_data = mol_data
        self.h_atom = h_atom
        self.h_coord = h_coord
        assert not ((not self.h_atom) and self.h_coord), 'Not supported.'
        self.randomize = randomize
        self.rng = rstate

        self.sample_save_dir = sample_save_dir
        self.getitem_count = 0
        
    def __getitem__(self, idx: int):
        mol = self.mol_data[idx]

        # remove/add hydrogen
        if self.h_atom:
            mol = Chem.AddHs(mol, addCoords=True)
        else:
            mol = Chem.RemoveHs(mol)
        
        # randomize
        if self.randomize:
            idxs = np.arange(mol.GetNumAtoms(), dtype=int)
            self.rng.shuffle(idxs)
            mol = Chem.RenumberAtoms(mol, idxs.tolist())
            smi = Chem.MolToSmiles(mol, canonical=False)
        else:
            smi = Chem.MolToSmiles(mol)
        atom_order = mol.GetProp('_smilesAtomOutputOrder', autoConvert=True)
        if self.h_atom and not self.h_coord:
            atom_order = [o for o in atom_order if mol.GetAtomWithIdx(o).GetSymbol() != 'H']
        coord = mol.GetConformer().GetPositions()
        coord = coord[atom_order]

        # save sample
        if self.sample_save_dir is not None and self.getitem_count < 5:
            save_dir = f"{self.sample_save_dir}/{idx}"
            os.makedirs(save_dir, exist_ok=True)
            with open(f"{save_dir}/out_smi.txt", 'w') as f:
                f.write(smi)
            pd.DataFrame(coord) \
                .to_csv(f"{save_dir}/out_coord.csv", header=False, index=False)

        self.getitem_count += 1
        return smi, coord
    

class RandomScoreDataset(Dataset[float]):
    def __init__(self, min: float, max: float, size: int, seed: int):
        self.min = min
        self.max = max
        self.size = size
        self.rng = random.Random(seed)

    def __getitem__(self, idx: int):
        return self.rng.uniform(self.min, self.max)

    def __len__(self):
        return self.size

