from torch.utils.data import Dataset
from rdkit import Chem
from openbabel import openbabel as ob
from .data import WrapDataset, get_rng

class RemoveIsotopeDataset(WrapDataset[ob.OBMol|Chem.Mol]):
    def __init__(self, mol_data: Dataset[ob.OBMol|Chem.Mol]):
        super().__init__(mol_data)
    
    def __getitem__(self, idx: int):
        mol = self.dataset[idx]
        if isinstance(mol, ob.OBMol):
            for atom in ob.OBMolAtomIter(mol):
                atom.SetIsotope(0)
        elif isinstance(mol, Chem.Mol):
            for atom in mol.GetAtoms():
                atom.SetIsotope(0)
        else:
            raise ValueError(f"Unknown {type(mol)=}")
        return mol

class SetHydrogenDataset(WrapDataset[ob.OBMol|Chem.Mol]):
    def __init__(self, dataset: Dataset[ob.OBMol|Chem.Mol], h: bool):
        super().__init__(dataset)
        self.h = h
    def __getitem__(self, idx):
        mol = self.dataset[idx]
        if isinstance(mol, ob.OBMol):
            if self.h:
                success = mol.AddHydrogens()
            else:
                success = mol.DeleteHydrogens()
            assert success
        else:
            if self.h:
                mol = Chem.AddHs(mol, addCoords=True)
            else:
                mol = Chem.RemoveHs(mol)
        return mol

class Mol2PDBDataset(WrapDataset[str]):
    def __init__(self, dataset: Dataset[ob.OBMol|Chem.Mol]):
        super().__init__(dataset)
        self.obc = ob.OBConversion()
        self.obc.SetOutFormat('pdb')
    def __getitem__(self, idx: int):
        protein = self.dataset[idx]
        if isinstance(protein, ob.OBMol):
            pdb = self.obc.WriteString(protein)
        else:
            pdb = Chem.MolToPDBBlock(protein)
        return pdb

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
