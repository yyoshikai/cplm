"""
251028 前処理のコードは消したので, 古いcommitを参照してください。
"""
import os, gzip
from logging import getLogger
from typing import Literal
from time import time

import pandas as pd
import torch
from prody import parsePDB, confProDy, addMissingAtoms
from ..lmdb import PickleLMDBDataset, IntLMDBDataset, data_len_to_blen
from ..data import TupleDataset
from rdkit import Chem
confProDy(verbosity='none')
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
        if split == 'test':
            raise NotImplementedError(f"test set is not supported for pockets.")
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
        return protein, data['lig_mol'], score, protein_path, None
        # ligand_path: 251113 indexがないのはおかしい気がするので一旦こうする

class CDProteinTestDataset(TupleDataset[tuple[Protein, Chem.Mol, float, str, str]]):
    """
    TargetDiffのtest setを利用
    (_idx.lmdbを使うと, targetdiffと同じディレクトリのデータが全て取られる。)
    現状, 生成にしか使わなさそうなので毎回読み込むようにしている。
    """
    def __init__(self):
        super().__init__(5)
        self.split_by_name = torch.load(f"{CDDIR}/targetdiff/split_by_name.pt", 
                weights_only=True)['test']
        os.makedirs("./tmp", exist_ok=True)
    def __getitem__(self, idx):

        pname, lname = self.split_by_name[idx]
        # 'BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3_pocket10.pdb', 'BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3.sdf
        pname = pname.split('_rec_')[0] + "_rec.pdb"
        lname = lname.removesuffix('.sdf')
        lname, sdf_idx = lname.rsplit('_', maxsplit=1)
        ligands_path = f"{CDDIR}/CrossDocked2020_v1.1/{lname}.sdf.gz"
        protein_path = f"{CDDIR}/CrossDocked2020_v1.1/{pname}"
        sdf_idx = int(sdf_idx)

        # ligand
        with gzip.open(ligands_path) as f:
            sup = iter(Chem.ForwardSDMolSupplier(f))
            for _ in range(sdf_idx):
                next(sup)
            mol = next(sup)

        # protein
        tmp_path = f"./tmp/{idx}_{time()}.pdb"
        addMissingAtoms(protein_path, outfile=tmp_path)
        protein = parsePDB(tmp_path, )
        assert protein is not None
        protein = Protein(protein.getData('name'), protein.getCoords())
        os.remove(tmp_path)

        return protein, mol, None, protein_path, ligands_path
    
    def __len__(self):
        return len(self.split_by_name)


class CDProteinDataset(TupleDataset[tuple[Protein, Chem.Mol, float, str, str]]):
    def __new__(cls, split: Literal['train', 'valid', 'test']):
        if split == 'test':
            return CDProteinTestDataset()
        elif split in ['train', 'valid']:
            return super().__new__(cls, split)
        else:
            raise ValueError(f"Unsupported {split=}")
    def __init__(self, split: Literal['train', 'valid']):
        super().__init__(5)
        self.indices = IntLMDBDataset(f"{CDDIR}/pockets/mask/{split}_idxs.lmdb")
        self.dataset = CDProteinWholeDataset()
    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]
    def __len__(self):
        return len(self.indices)
