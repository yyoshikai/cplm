
from rdkit import Chem
from openbabel.openbabel import OBMol, OBConversion
from torch.utils.data import Dataset
from ...utils.path import WORKDIR

DEFAULT_POSEBUSTERS_DIR = f"{WORKDIR}/cheminfodata/posebusters"

class PosebustersV2ProteinDataset(Dataset[OBMol]):
    def __init__(self, posebusters_dir: str=DEFAULT_POSEBUSTERS_DIR):
        self.pb_dir = posebusters_dir
        with open(f"{self.pb_dir}/posebusters_pdb_ccd_ids.txt") as f:
            self.ids = f.read().splitlines()
        self.obc = OBConversion()
        self.obc.SetInFormat('pdb')

    def __getitem__(self, idx: int):
        id = self.ids[idx]
        mol = OBMol()
        self.obc.ReadFile(mol, f"{self.pb_dir}/posebusters_benchmark_set/{id}/{id}_protein.pdb")
        return mol

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


