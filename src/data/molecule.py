import os
from typing import Optional
import numpy as np, pandas as pd
from torch.utils.data import Dataset, get_worker_info
from rdkit import Chem
from .data import WrapTupleDataset, is_main_worker, get_rng

class MolProcessDataset(WrapTupleDataset[tuple[str, np.ndarray]]):
    def __init__(self, mol_data: Dataset[Chem.Mol], base_seed: int, h_atom: bool, h_coord: bool, randomize: bool, sample_save_dir: Optional[str]=None):
        super().__init__(mol_data, 2)
        self.mol_data = mol_data
        self.h_atom = h_atom
        self.h_coord = h_coord
        assert not ((not self.h_atom) and self.h_coord), 'Not supported.'
        self.randomize = randomize
        self.seed = base_seed

        self.sample_save_dir = sample_save_dir
        self.getitem_count = 0
        
    def __getitem__(self, idx: int):
        mol = self.mol_data[idx]

        # rng
        epoch = int(os.environ.get('EPOCH', 0))
        rng = get_rng(self.seed, idx)

        # remove/add hydrogen
        if self.h_atom:
            mol = Chem.AddHs(mol, addCoords=True)
        else:
            mol = Chem.RemoveHs(mol)
        
        # randomize
        if self.randomize:
            idxs = np.arange(mol.GetNumAtoms(), dtype=int)
            rng.shuffle(idxs)
            mol = Chem.RenumberAtoms(mol, idxs.tolist())
            smi = Chem.MolToSmiles(mol, canonical=False)
        else:
            smi = Chem.MolToSmiles(mol)
        try:
            atom_order = eval(mol.GetProp('_smilesAtomOutputOrder'))
        except Exception as e:
            print(f"{mol=}")
            raise e
        if self.h_atom and not self.h_coord:
            atom_order = [o for o in atom_order if mol.GetAtomWithIdx(o).GetSymbol() != 'H']
        coord = mol.GetConformer().GetPositions()
        coord = coord[atom_order]

        # save sample
        if self.sample_save_dir is not None and self.getitem_count < 5 and is_main_worker():
            worker_info = get_worker_info()
            if worker_info is None or worker_info.id == 0:
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
        self.seed = seed

    def __getitem__(self, idx: int):
        return get_rng(self.seed, idx).uniform(self.min, self.max)

    def __len__(self):
        return self.size

class RandomClassDataset(Dataset[bool]):
    def __init__(self, size: int, seed: int):
        self.seed = seed
        self.size = size
    def __getitem__(self, idx: int):
        return get_rng(self.seed, idx).uniform() < 0.5
    def __len__(self):
        return self.size