from typing import Literal
from rdkit import Chem
from openbabel.openbabel import OBMol, OBConversion
from torch.utils.data import Dataset
from ...utils.path import WORKDIR
from ...chem import read_pdb_path
DEFAULT_POSEBUSTERS_DIR = f"{WORKDIR}/cheminfodata/posebusters"

class PosebustersV2ProteinDataset(Dataset[OBMol|Chem.Mol]):
    def __init__(self, out_cls: Literal['ob', 'rdkit', 'text'], posebusters_dir: str=DEFAULT_POSEBUSTERS_DIR):
        self.pb_dir = posebusters_dir
        with open(f"{self.pb_dir}/posebusters_pdb_ccd_ids.txt") as f:
            self.ids = f.read().splitlines()
        self.out_cls = out_cls

    def __getitem__(self, idx: int):
        id = self.ids[idx]
        return read_pdb_path(f"{self.pb_dir}/posebusters_benchmark_set/{id}/{id}_protein.pdb", self.out_cls)

    def __len__(self):
        return len(self.ids)

class PosebustersV2LigandDataset(Dataset[Chem.Mol]):
    def __init__(self, posebusters_dir: str=DEFAULT_POSEBUSTERS_DIR):
        self.pb_dir= posebusters_dir
        with open(f"{self.pb_dir}/posebusters_pdb_ccd_ids.txt") as f:
            self.ids = f.read().splitlines()
    
    def __getitem__(self, idx: int):
        id = self.ids[idx]
        mol = next(Chem.SDMolSupplier(f"{self.pb_dir}/posebusters_benchmark_set/{id}/{id}_ligand.sdf", removeHs=False))
        return mol
    
    def __len__(self):
        return len(self.ids)


