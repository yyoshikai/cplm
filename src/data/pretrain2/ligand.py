import sys, os
from typing import Optional
from logging import getLogger
import numpy as np, pandas as pd
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import Conformer
from rdkit.Geometry import Point3D
from ..coord_transform import CoordTransform
from ..lmdb import PickleLMDBDataset
from ..tokenizer import FloatTokenizer, StringTokenizer
from ...utils import logtime

# Pretrain時, 分子の処理用のデータセット
class MoleculeDataset(Dataset):
    logger = getLogger(f'{__module__}.{__qualname__}')
    def __init__(self, dataset: Dataset, coord_transform: CoordTransform, 
            smiles_tokenizer: StringTokenizer, coord_tokenizer: FloatTokenizer
        ):
        self.dataset = dataset
        self.coord_transform = coord_transform
        self.smiles_tokenizer = smiles_tokenizer
        self.coord_tokenizer = coord_tokenizer

    def __getitem__(self, idx):
        data = self.dataset[idx]
        with logtime(self.logger, f"[{idx}]"):
            smi = data['smi']
            coord = data['coordinate']
            coord = self.coord_transform(coord)
            return ['[LIGAND]']+self.smiles_tokenizer.tokenize(smi)+['[XYZ]']+self.coord_tokenizer.tokenize_array(coord.ravel())+['[END]']

    def __len__(self):
        return len(self.dataset)

    def vocs(self):
        return self.smiles_tokenizer.vocs()|self.coord_tokenizer.vocs()|{'[LIGAND]', '[XYZ]', '[END]'}

class UniMolLigandDataset(Dataset):
    logger = getLogger(f'{__module__}.{__qualname__}')
    def __init__(self, lmdb_path, n_conformer, seed: int, 
            atom_h: bool=True, coord_h: bool=True, randomize: bool=False, 
            sample_save_dir: Optional[str]=None):
        """
        ややdata specificな処理を行っている。
        本当はもう少し上の段階でランダム化を行った方がよいかもしれない。
                    
        Parameters
        ----------
        atom_h: SMILESに水素を追加するかどうか
        coord_h: 座標に水素を追加するかどうか

        atom_h=True, coord_h=True とするとBindGPT
        """
        self.net_dataset = PickleLMDBDataset(lmdb_path, idx_to_key='str')
        self.n_conformer = n_conformer
        self.atom_h = atom_h
        self.coord_h = coord_h
        if (not self.atom_h) and self.coord_h:
            raise ValueError(f"atom_h=False and coord_h=True is not supported.")
        self.randomize = randomize
        self.rng = np.random.default_rng(seed)

        self.sample_save_dir = sample_save_dir
        self.getitem_count = 0

    def __getitem__(self, idx):
        mol_idx, conformer_idx = divmod(idx, self.n_conformer)
        data = self.net_dataset[mol_idx]
        with logtime(self.logger, f"[{idx}]"):
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

            # remove hydrogen
            if not self.atom_h:
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
            if self.atom_h and not self.coord_h:
                atom_order = [o for o in atom_order if mol.GetAtomWithIdx(o).GetSymbol() != 'H']
            coord = mol.GetConformer().GetPositions()
            coord = coord[atom_order]

            # save sample
            if self.sample_save_dir is not None and self.getitem_count < 5:
                save_dir = f"{self.sample_save_dir}/{idx}"
                os.makedirs(save_dir, exist_ok=True)
                with open(f"{save_dir}/data_smi.txt", 'w') as f:
                    f.write(data['smi'])
                with open(f"{save_dir}/out_smi.txt", 'w') as f:
                    f.write(smi)
                pd.DataFrame(data['coordinates'][conformer_idx]) \
                    .to_csv(f"{save_dir}/data_coord.csv", header=False, index=False)
                pd.DataFrame(coord) \
                    .to_csv(f"{save_dir}/out_coord.csv", header=False, index=False)

            output = {'smi': smi, 'coordinate': coord}
            self.getitem_count += 1
            return output
    
    def __len__(self):
        return len(self.net_dataset) * self.n_conformer
