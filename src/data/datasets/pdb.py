import os, gzip
from typing import Literal

from openbabel.openbabel import OBMol
from rdkit import Chem
from torch.utils.data import Dataset, Subset
from ..lmdb import StringLMDBDataset
from ...chem import read_pdb_text
from ...utils.path import WORKDIR

DEFAULT_PDB_DIR = f"{WORKDIR}/cheminfodata/pdb/220103"
DEFAULT_VALID_SIZE = 100

class PDBDataset(Dataset[OBMol|Chem.Mol]):
    def __init__(self, pdbid_name: str, cls: Literal['ob', 'rdkit', 'text'], pdb_dir: str=DEFAULT_PDB_DIR):
        self.pdb_dir = pdb_dir
        self.pdbid_data = StringLMDBDataset(f"{pdb_dir}/{pdbid_name}.lmdb")
        self.cls = cls

    def __getitem__(self, idx: int) -> OBMol:
        # get pdbid
        pdbid = self.pdbid_data[idx]

        # load protein
        path = f"{self.pdb_dir}/pdb/{pdbid[1:3]}/pdb{pdbid}.ent.gz"
        with gzip.open(path, 'rt') as f:
            pdb_text = f.read()
        mol = read_pdb_text(pdb_text, self.cls)
        return mol
    
    def __len__(self):
        return len(self.pdbid_data)

# 誰も使ってない
class PDBUniMolDataset(Subset[OBMol]):
    def __init__(self, split: Literal['train', 'valid'], cls: Literal['ob', 'rdkit', 'text'], pdb_dir: str=DEFAULT_PDB_DIR):
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

class PDBUniMolRandomDataset(Subset[OBMol]):
    def __init__(self, split: Literal['train', 'valid'], cls: Literal['ob', 'rdkit', 'text'], pdb_dir: str=DEFAULT_PDB_DIR):
        # Whole data
        whole_data = PDBDataset("unimol_random_order_pdbids", cls, pdb_dir)

        # Get index
        if split == 'train':
            indices = range(DEFAULT_VALID_SIZE, len(whole_data))
        elif split == 'valid':
            indices = range(DEFAULT_VALID_SIZE)
        else:
            raise ValueError
        
        super().__init__(whole_data, indices)

    def __str__(self):
        return type(self).__name__