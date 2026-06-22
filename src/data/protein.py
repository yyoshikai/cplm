import io
import itertools as itr
from typing import Literal
from openbabel.openbabel import OBMol
from rdkit import Chem
from Bio import PDB
from Bio.PDB.Residue import Residue
from torch.utils.data import Dataset
from .data import WrapDataset
from ..chem import obmol2pdb, pdb2obmol

AtomRepr = Literal['none', 'atom', 'all']

non_metals = [
    'H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Si', 'P', 'S', 'Cl', 'Ar', 'As', 'Se', 'Br', 'Kr',
    'Te', 'I', 'Xe', 'At', 'Rn'
]

class _ProteinProcessSelect(PDB.Select):
    def __init__(self, ion: bool, ligand: bool, water: bool, amino: bool=True):
        self.ion = ion
        self.ligand = ligand
        self.water = water
        self.amino = amino
    def accept_residue(self, residue: Residue):
        id0 = residue.get_id()[0]
        if id0 == ' ': # amino acid
            return self.amino
        elif id0 == 'W':
            return self.water
        elif id0[:2] == 'H_':
            if len(id0[2:]) <= 2:
                return self.ion
            else:
                return self.ligand
        else:
            raise ValueError


class SelectDataset(WrapDataset[OBMol|Chem.Mol]):
    def __init__(self, mol_data: Dataset[OBMol|Chem.Mol], ion: bool, ligand: bool, water: bool):
        super().__init__(mol_data)
        self.select = _ProteinProcessSelect(ion, ligand, water)
        self.ion = ion
        self.ligand = ligand
        self.water = water

    def __getitem__(self, idx: int):
        mol = self.dataset[idx]
        if isinstance(mol, OBMol):
            pdb = obmol2pdb(mol)
            parser = PDB.PDBParser(QUIET=True)
            mol = parser.get_structure('a', io.StringIO(pdb))
            pdbio = PDB.PDBIO()
            string_io = io.StringIO()
            pdbio.set_structure(mol)
            pdbio.save(string_io, self.select)
            pdb = string_io.getvalue()
            mol = pdb2obmol(pdb)
        else:
            rw_mol = Chem.RWMol(mol)
            remove_atom_idxss = []
            for frag_idxs in Chem.GetMolFrags(mol):
                if len(frag_idxs) == 1:
                    atom = mol.GetAtomWithIdx(frag_idxs[0])
                    if atom.GetSymbol() == 'O':
                        remain = self.water
                    elif atom.GetSymbol() in non_metals: # NH3, CH4, etc.
                        remain = self.ligand
                    else: # Fe, Mg, etc.
                        remain = self.ion
                else:
                    if all(mol.GetAtomWithIdx(idx).GetPDBResidueInfo().GetIsHeteroAtom() for idx in frag_idxs):
                        remain = self.ligand
                    else:
                        remain = True
                if not remain:
                    remove_atom_idxss.append(frag_idxs)
            for idx in sorted(itr.chain(*remove_atom_idxss), reverse=True):
                rw_mol.RemoveAtom(idx)
            mol = rw_mol.GetMol()
        return mol
