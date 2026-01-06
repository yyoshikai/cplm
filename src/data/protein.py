from dataclasses import dataclass
from ctypes import c_double
from typing import Literal
import numpy as np
from openbabel.openbabel import OBMol, OBMolAtomIter
from torch.utils.data import Dataset
from .data import WrapDataset
from .tokenizer import FloatTokenizer, ProteinAtomTokenizer
from ..utils import slice_str


@dataclass
class Pocket:
    atoms: np.ndarray
    coord: np.ndarray

    def __post_init__(self):
        assert len(self.atoms) == len(self.coord)
        assert self.coord.ndim == 2 and self.coord.shape[1] == 3

def pocket2pdb(pocket: Pocket, out_path: str):
    with open(out_path, 'w') as f:
        for ia in range(len(pocket.atoms)):
            atom = pocket.atoms[ia][0]
            coord = pocket.coord[ia]
            if atom == 'H': continue
            f.write(f"ATOM  {ia:5}  {atom:<3} UNK A   1    {coord[0]:8.03f}{coord[1]:8.03f}{coord[1]:8.03f}  1.00 40.00           {atom[0]}  \n")


def get_coord_from_mol(mol: OBMol) -> np.ndarray:
    coord = mol.GetCoordinates()
    return np.array((c_double * (mol.NumAtoms()*3)).from_address(int(coord))).reshape(-1, 3)

class ProteinTokenizer:
    def __init__(self, *, heavy_atom, heavy_coord, h_atom, h_coord, coord_follow_atom, coord_range, atom_order):
        self.heavy_atom = heavy_atom
        self.h_atom = h_atom
        self.heavy_coord = heavy_coord
        self.h_coord = h_coord
        self.atom_order = atom_order

        self.coord_follow_atom = coord_follow_atom
        self.atom_tokenizer = ProteinAtomTokenizer()
        self.coord_tokenizer = FloatTokenizer('protein', -coord_range, coord_range)
        assert not (self.heavy_coord and not self.heavy_atom)
        assert not (self.h_coord and not self.h_atom)
        assert not (self.coord_follow_atom and self.atom_order)

    def __call__(self, atoms: np.ndarray, coords: np.ndarray):
        # calc mask
        is_ca = atoms == 'CA'
        is_h = slice_str(atoms, 1) == 'H'
        is_heavy = (~is_ca)&(~is_h)

        # atoms 
        atom_mask = is_ca | (is_heavy if self.heavy_atom else False) | (is_h if self.h_atom else False)

        # coord
        coord_mask = is_ca | (is_heavy if self.heavy_coord else False) | (is_h if self.h_coord else False)
                    

        if self.coord_follow_atom:
            tokens = []
            for i in range(len(atoms)):
                if atom_mask[i]: 
                    tokens += self.atom_tokenizer.tokenize([atoms[i]])
                if coord_mask[i] and self.coord:
                    tokens += self.coord_tokenizer.tokenize_array(coords[i])
            order = list(range(len(tokens)))
        elif self.atom_order:
            atom_tokens = []
            coord_tokens = []
            atom_order = []
            coord_order = []
            x = 0
            for i in range(len(atoms)):
                if atom_mask[i]:
                    atom_token = self.atom_tokenizer.tokenize([atoms[i]])
                    atom_tokens += atom_token
                    atom_order += list(range(x, x+len(atom_token)))
                    x += len(atom_token)
                if coord_mask[i]:
                    coord_token = self.coord_tokenizer.tokenize_array(coords[i])
                    coord_tokens += coord_token
                    coord_order += list(range(x, x+len(coord_token)))
                    x += len(coord_token)
            tokens = atom_tokens+['[XYZ]']+coord_tokens
            order = atom_order+[x]+coord_order
        else:
            tokens = self.atom_tokenizer.tokenize(atoms[atom_mask]) \
                +['[XYZ]']+self.coord_tokenizer.tokenize_array(coords[coord_mask].ravel())
            order = list(range(len(tokens)))
        return tokens, order
    
    def vocs(self) -> set[str]:
        return self.atom_tokenizer.vocs() | self.coord_tokenizer.vocs() \
                | ({'[XYZ]'} if not self.coord_follow_atom else set())

# 水素は含んでいても含んでいなくてもよいが, atomとcoordでそろえること。
class PocketTokenizeDataset(WrapDataset[tuple[list[str], list[int]]]):
    def __init__(self, pocket_data: Dataset[Pocket], *,
            heavy_atom: bool, heavy_coord: bool, 
            h_atom: bool, h_coord: bool, 
            coord_follow_atom: bool, atom_order: bool, coord_range: int):
        super().__init__(pocket_data)
        self.pocket_data = pocket_data
        self.protein_tokenizer = ProteinTokenizer(heavy_atom=heavy_atom, heavy_coord=heavy_coord, h_atom=h_atom, h_coord=h_coord, coord_follow_atom=coord_follow_atom, coord_range=coord_range, atom_order=atom_order)
        assert not h_atom, f"h_atom is not supported for Pocket"
        assert not h_coord, f"h_coord is not supported for Pocket"

    def __getitem__(self, idx: int):
        protein = self.pocket_data[idx]
        assert len(protein.atoms) == len(protein.coord)
        return self.protein_tokenizer(protein.atoms, protein.coord)
    
    def vocs(self) -> set[str]:
        return self.protein_tokenizer.vocs()

class ProteinTokenizeDataset(WrapDataset[tuple[list[str], list[int]]]):
    def __init__(self, protein_data: Dataset[OBMol], *,
            heavy_atom: bool, heavy_coord: bool, 
            h_atom: bool, h_coord: bool, 
            coord_follow_atom: bool, atom_order: bool, coord_range: int):
        super().__init__(protein_data)
        self.protein_data = protein_data
        self.protein_tokenizer = ProteinTokenizer(heavy_atom=heavy_atom, heavy_coord=heavy_coord, h_atom=h_atom, h_coord=h_coord, coord_follow_atom=coord_follow_atom, coord_range=coord_range, atom_order=atom_order)
        self.h_atom = h_atom

    def __getitem__(self, idx: int):
        protein = self.protein_data[idx]

        # add/remove hydrogen
        if self.h_atom:
            success = protein.AddHydrogens()
        else:
            success = protein.DeleteHydrogens()
        assert success
        # Order atoms
        atoms = np.array([atom.GetResidue().GetAtomID(atom).strip() for atom in OBMolAtomIter(protein)])
        residue_idxs = np.array([atom.GetResidue().GetIdx() for atom in OBMolAtomIter(protein)])
        coords = get_coord_from_mol(protein)
        orders = np.argsort(residue_idxs, kind='stable')
        atoms = atoms[orders]
        coords = coords[orders]

        # tokenize
        return self.protein_tokenizer(atoms, coords)

    def vocs(self) -> set[str]:
        return self.protein_tokenizer.vocs()