import os, gzip, pickle
from typing import Literal
from logging import getLogger
import numpy as np
from openbabel.openbabel import OBMol
from rdkit import Chem
from torch.utils.data import Dataset, Subset
from ...utils.lmdb import load_lmdb
from ..lmdb import StringLMDBDataset
from ...chem import read_pdb_text
from ...utils.path import WORKDIR

DEFAULT_PDB_DIR = f"{WORKDIR}/cheminfodata/pdb/220103"
DEFAULT_VALID_SIZE = 100

class PDBDataset(Dataset[OBMol|Chem.Mol|str]):
    def __init__(self, pdbid_name: str, cls: Literal['ob', 'rdkit', 'text', 'id'], pdb_dir: str=DEFAULT_PDB_DIR):
        self.pdb_dir = pdb_dir
        self.pdbid_data = StringLMDBDataset(f"{pdb_dir}/{pdbid_name}.lmdb")
        self.cls = cls

    def __getitem__(self, idx: int):
        # get pdbid
        pdbid = self.pdbid_data[idx]
        if self.cls == 'id':
            return pdbid
        # load protein
        path = f"{self.pdb_dir}/pdb/{pdbid[1:3]}/pdb{pdbid}.ent.gz"
        with gzip.open(path, 'rt') as f:
            pdb_text = f.read()
        mol = read_pdb_text(pdb_text, self.cls)
        return mol
    
    def __len__(self):
        return len(self.pdbid_data)

# 誰も使ってない
class PDBUniMolDataset(Subset[OBMol|Chem.Mol|str]):
    def __init__(self, split: Literal['train', 'valid'], cls: Literal['ob', 'rdkit', 'text', 'id'], pdb_dir: str=DEFAULT_PDB_DIR):
        # Whole data
        whole_data = PDBDataset("unimol_valid_count_order_pdbids", cls, pdb_dir)

        # Get index
        if split == 'train':
            indices = range(DEFAULT_VALID_SIZE, len(whole_data))
        elif split == 'valid':
            indices = range(DEFAULT_VALID_SIZE)
        else:
            raise ValueError
        
        super().__init__(whole_data, indices)

class PDBUniMolRandomDataset(Subset[OBMol|Chem.Mol|str]):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, 
            split: Literal['train', 'valid'], 
            cls: Literal['ob', 'rdkit', 'text', 'id'], 
            h: Literal['all', 'atom', 'none'], 
            max_n_token: int,
            ion: bool, ligand: bool, water: bool, 
            pdb_dir: str=DEFAULT_PDB_DIR, 
    ):
        mol_data = PDBDataset("unimol_random_order_pdbids", cls, pdb_dir)
        
        # path = f"{pdb_dir}/natom_masks/{max_n_token}_{int(ion)}_{int(ligand)}_{int(water)}"
        self.logger.info("Making mask index...")
        env, txn = load_lmdb(f"{pdb_dir}/natom.lmdb")

        ntokens = []
        for i in range(len(mol_data)):
            pdbid = mol_data.pdbid_data[i]
            ns = pickle.loads(txn.get(pdbid.encode()))
            natom = (ns['amino']
                + ns['ion'] if ion else 0
                + ns['ligand'] if ligand else 0
                + ns['water'] if water else 0
            )
            ntoken = natom*7
            if h != 'none':
                natom_h = (ns['amino_h']
                    + ns['ion_h'] if ion else 0
                    + ns['ligand_h'] if ligand else 0
                    + ns['water_h'] if water else 0
                )
                if h == 'all':
                    ntoken += natom_h*7
                elif h == 'atom':
                    ntoken += natom_h
            ntokens.append(ntoken)

        idxs = np.where(np.array(ntokens) <= max_n_token)[0]
        self.logger.info("mask index made.")
        self.logger.info(f"Masked data size={len(idxs)}/{len(mol_data)}")

        if split == 'train':
            idxs = idxs[DEFAULT_VALID_SIZE:]
        elif split == 'valid':
            idxs = idxs[:DEFAULT_VALID_SIZE]
        else:
            raise ValueError

        super().__init__(mol_data, idxs)

    def __str__(self):
        return type(self).__name__
