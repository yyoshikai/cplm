"""
251028 前処理のコードは消したので, 古いcommitを参照してください。
"""
import os
from logging import getLogger
from typing import Literal
from time import time

import pandas as pd
from prody import parsePDB, confProDy, addMissingAtoms
from ..lmdb import PickleLMDBDataset, IntLMDBDataset
from ..data import WrapDataset, TupleDataset, Subset
from rdkit import Chem
confProDy(verbosity='none')
from ...utils.utils import CompressedArray
from ...utils.path import WORKDIR
from ..protein import Protein

SAVE_DIR = f"{WORKDIR}/cplm/ssd/preprocess/results/finetune/r4_all"
CDDIR = f"{WORKDIR}/cheminfodata/crossdocked"

finetune_data_type = tuple[Protein, Chem.Mol, float, str, str]
class CDWholeDataset(TupleDataset[finetune_data_type]):
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
        pocket_path = f"{CDDIR}/CrossDocked2020/{data['dname']}/{data['protein_name']}"

        score = float(data['score'])
        return pocket, data['mol'], score, pocket_path, ligand_path

class CDDataset(Subset[finetune_data_type]):
    def __init__(self, split: Literal['train', 'valid', 'test']):
        dataset = CDWholeDataset()
        indices = IntLMDBDataset(f"{CDDIR}/pockets/mask/{split}_idxs.lmdb")
        super().__init__(dataset, indices)

class CDProteinDataset(WrapDataset[tuple[Protein, Chem.Mol, float]]):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, save_dir: str, cddir: str=f"{WORKDIR}/cheminfodata/crossdocked/CrossDocked2020",  out_filename: bool=False):
        self.lmdb_dataset = PickleLMDBDataset(f"{save_dir}/main.lmdb", idx_to_key='str')
        super().__init__(self.lmdb_dataset)
        self.out_filename = out_filename
        self.cddir = cddir
        os.makedirs("tmp", exist_ok=True)

        self.logger.info("Loading filenames.csv.gz ... ")
        df = pd.read_csv(f"{save_dir}/filenames.csv.gz")
        self.logger.info("loaded.")
        self.df = {'idx': df['idx'].values}
        for key in ['dname', 'lig_name', 'protein_name']:
            self.df[key] = CompressedArray(df[key].values)
        self.df['sdf_idx'] = df['sdf_idx'].values
        del df

    def __getitem__(self, idx: int):
        data = self.lmdb_dataset[idx]

        # ligand
        lig_mol: Chem.Mol = data['lig_mol']
        score = float(data['score'])

        # protein
        dname = self.df['dname'][idx]
        protein_name = self.df['protein_name'][idx]
        addMissingAtoms(f"{self.cddir}/{dname}/{protein_name}", outfile=f"./tmp/{dname}_{protein_name}.pdb")
        protein = parsePDB(f"./tmp/{dname}_{protein_name}.pdb")
        protein = Protein(protein.getData('name'), protein.getCoords())
        
        output = (protein, lig_mol, score)
        if self.out_filename:
            output += ({key: self.df[key][idx] for key in self.df}, )
        return output
