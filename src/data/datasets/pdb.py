import os, gzip
from typing import Literal

from openbabel.openbabel import OBMol, OBConversion
from torch.utils.data import Dataset, Subset
from ..lmdb import StringLMDBDataset

WORKDIR = os.environ.get('WORKDIR', __file__.split('/cplm/')[0])

DEFAULT_PDB_DIR = f"{WORKDIR}/cheminfodata/pdb/220103"
DEFAULT_VALID_SIZE = 100

class PDBDataset(Dataset[OBMol]):
    def __init__(self, pdbid_name: str, pdb_dir: str=DEFAULT_PDB_DIR):
        self.pdb_dir = pdb_dir
        self.pdbid_data = StringLMDBDataset(f"{pdb_dir}/{pdbid_name}.lmdb")
        self.obc = OBConversion()
        self.obc.SetInFormat('pdb')

    def __getitem__(self, idx: int) -> OBMol:
        # get pdbid
        pdbid = self.pdbid_data[idx]

        # load protein
        path = f"{self.pdb_dir}/pdb/{pdbid[1:3]}/pdb{pdbid}.ent.gz"
        mol = OBMol()
        with gzip.open(path, 'rt') as f:
            self.obc.ReadString(mol, f.read())
        return mol
    
    def __len__(self):
        return len(self.pdbid_data)
    
class PDBUniMolDataset(Subset[OBMol]):
    def __init__(self, split: Literal['train', 'valid'], pdb_dir: str=DEFAULT_PDB_DIR):
        # Whole data
        whole_data = PDBDataset("unimol_valid_count_order_pdbids", pdb_dir)

        # Get index
        if split == 'train':
            indices = range(DEFAULT_VALID_SIZE, len(whole_data))
        elif split == 'valid':
            indices = range(DEFAULT_VALID_SIZE)
        else:
            raise ValueError
        
        super().__init__(whole_data, indices)

class PDBUniMolRandomDataset(Subset[OBMol]):
    def __init__(self, split: Literal['train', 'valid'], pdb_dir: str=DEFAULT_PDB_DIR):
        # Whole data
        whole_data = PDBDataset("unimol_random_order_pdbids", pdb_dir)

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