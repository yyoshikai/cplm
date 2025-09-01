import sys, os, gzip
import numpy as np
from logging import getLogger
from prody import parsePDBStream, AtomGroup
from torch.utils.data import Dataset, Subset
from ..protein import Protein

WORKDIR = os.environ.get('WORKDIR', "/workspace")

DEFAULT_PDB_DIR = f"{WORKDIR}/cheminfodata/pdb/240101"
class PDBDataset(Dataset[Protein]):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, pdb_dir: str=DEFAULT_PDB_DIR):
        self.pdb_dir = pdb_dir
        with open(f"{pdb_dir}/files_pdb.txt") as f:
            self.fnames = np.array(f.read().splitlines())

    def __getitem__(self, idx: int) -> Protein:
        path = f"{self.pdb_dir}/{self.fnames[idx]}"
        with gzip.open(path, 'rt') as f:
            protein: AtomGroup = parsePDBStream(f)
            atoms = protein.getData('name')
            coord = protein.getCoords()
        return Protein(atoms, coord)
    
    def __len__(self):
        return len(self.fnames)

class PDBNoTDTestDataset(Subset[Protein]):
    def __init__(self, pdb_dir: str=DEFAULT_PDB_DIR):
        dataset = PDBDataset(pdb_dir)
        idxs = np.load(f"{pdb_dir}/remove_targetdiff_test.npy")
        super().__init__(dataset, idxs)
