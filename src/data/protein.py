from dataclasses import dataclass
import numpy as np
from torch.utils.data import Dataset
from .data import WrapDataset
from .tokenizer import FloatTokenizer, ProteinAtomTokenizer
from ..utils import slice_str


@dataclass
class Protein:
    atoms: np.ndarray
    coord: np.ndarray

    def __post_init__(self):
        assert len(self.atoms) == len(self.coord)
        assert self.coord.ndim == 2 and self.coord.shape[1] == 3

def protein2pdb(protein: Protein, out_path: str):
    with open(out_path, 'w') as f:
        for ia in range(len(protein.atoms)):
            atom = protein.atoms[ia][0]
            coord = protein.coord[ia]
            if atom == 'H': continue
            f.write(f"ATOM  {ia:5}  {atom:<3} UNK A   1    {coord[0]:8.03f}{coord[1]:8.03f}{coord[1]:8.03f}  1.00 40.00           {atom[0]}  \n")

# 水素は含んでいても含んでいなくてもよいが, atomとcoordでそろえること。
class ProteinTokenizeDataset(WrapDataset[list[str]]):
    def __init__(self, protein_data: Dataset[Protein],
            heavy_atom: bool, h_atom: bool,
            heavy_coord: bool, h_coord: bool, 
            coord_follow_atom: bool, coord_range: int):
        super().__init__(protein_data)
        self.protein_data = protein_data
        self.heavy_atom = heavy_atom
        self.h_atom = h_atom
        self.heavy_coord = heavy_coord
        self.h_coord = h_coord
        self.coord_follow_atom = coord_follow_atom
        self.atom_tokenizer = ProteinAtomTokenizer()
        self.coord_tokenizer = FloatTokenizer('protein', -coord_range, coord_range)
        assert not (self.heavy_coord and not self.heavy_atom)
        assert not (self.h_coord and not self.h_atom)

    def __getitem__(self, idx: int):
        protein = self.protein_data[idx]
        atoms = protein.atoms
        coords = protein.coord
        assert len(atoms) == len(coords)

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
                if coord_mask[i]:
                    tokens += self.coord_tokenizer.tokenize_array(coords[i])
        else:
            tokens = self.atom_tokenizer.tokenize(atoms[atom_mask]) \
                    +['[XYZ]']+self.coord_tokenizer.tokenize_array(coords[coord_mask].ravel())
        return tokens
    
    def vocs(self) -> set[str]:
        return self.atom_tokenizer.vocs()|self.coord_tokenizer.vocs()| \
                ({'[XYZ]'} if not self.coord_follow_atom else set())
