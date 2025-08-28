from dataclasses import dataclass
import numpy as np
from torch.utils.data import Dataset
from .data import WrapDataset
from .tokenizer import TokenizeDataset, ArrayTokenizeDataset
from ..utils import slice_str


@dataclass
class Protein:
    atoms: np.ndarray
    coord: np.ndarray

# 水素は含んでいても含んでいなくてもよいが, atomとcoordでそろえること。
class ProteinProcessDataset(WrapDataset[tuple[list[str], np.ndarray]]):
    def __init__(self, protein_data: Dataset[Protein],
            heavy_atom: bool=True, h_atom: bool=False,
            heavy_coord: bool=False, h_coord: bool=False):
        super().__init__(protein_data)
        self.protein_data = protein_data
        self.heavy_atom = heavy_atom
        self.h_atom = h_atom
        self.heavy_coord = heavy_coord
        self.h_coord = h_coord
        assert not (self.heavy_coord and not self.heavy_atom)
        assert not (self.h_coord and not self.h_atom)

    def __getitem__(self, idx: int):
        protein = self.protein_data[idx]
        atoms = protein.atoms
        coord = protein.coord

        # calc mask
        is_ca = atoms == 'CA'
        is_h = slice_str(atoms, 1) == 'H'
        is_heavy = (~is_ca)&(~is_h)

        # atoms 
        atom_mask = is_ca.copy()
        if self.heavy_atom: atom_mask |= is_heavy
        if self.h_atom: atom_mask |= is_h
        atoms = atoms[atom_mask]

        # coord
        coord_mask = is_ca.copy()
        if self.heavy_coord: coord_mask |= is_heavy
        if self.h_coord: coord_mask |= is_h
        coord = coord[coord_mask]
        coord_position = np.where(coord_mask[atom_mask])[0]

        return atoms, coord, coord_position

class CoordFollowDataset(Dataset):
    def __init__(self, pocket_atom: TokenizeDataset, pocket_coord: ArrayTokenizeDataset, coord_position: Dataset[np.ndarray]):
        self.pocket_atom = pocket_atom
        self.pocket_coord = pocket_coord
        self.coord_position = coord_position
        assert len(self.pocket_atom) == len(self.pocket_coord)
    
    def __getitem__(self, idx: int):
        pocket_atom = self.pocket_atom[idx]
        pocket_coord = self.pocket_coord[idx]
        coord_position = self.coord_position[idx]
        assert len(coord_position)*6 == len(pocket_coord)
        output = []
        i_coord = 0
        for i_atom in range(len(pocket_atom)):
            output.append(pocket_atom[i_atom])
            if coord_position[i_coord] == i_atom:
                output += pocket_coord[i_coord*6:(i_coord+1)*6]
                i_coord += 1
        return output

    def __len__(self):
        return len(self.pocket_atom)
    
    def vocs(self):
        return self.pocket_atom.vocs()|self.pocket_coord.vocs()
