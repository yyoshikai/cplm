from dataclasses import dataclass
from typing import Literal
import numpy as np

from openbabel.openbabel import OBMol, OBMolAtomIter, OBConversion
from torch.utils.data import Dataset
from ..utils import slice_str
from ..chem import get_coord_from_mol, obmol2rdmol, set_atom_order
from .data import WrapDataset, get_rng
from .tokenizer import FloatTokenizer, ProteinAtomTokenizer

AtomRepr = Literal['none', 'atom', 'all']

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


class ProteinTokenizer:
    def __init__(self, *, heavy: AtomRepr, h: AtomRepr, format, coord_range):
        self.heavy = heavy
        self.h = h
        self.format = format
        self.atom_tokenizer = ProteinAtomTokenizer()
        self.coord_tokenizer = FloatTokenizer('protein', -coord_range, coord_range)
        assert self.heavy in ['all', 'atom', 'none']
        assert self.h in ['all', 'atom', 'none']

    def __call__(self, atoms: np.ndarray, coords: np.ndarray):
        # calc mask
        is_ca = atoms == 'CA'
        is_h = slice_str(atoms, 1) == 'H'
        is_heavy = (~is_ca)&(~is_h)

        # atoms 
        atom_mask = is_ca | (is_heavy if self.heavy in ['all', 'atom'] else False) | (is_h if self.h in ['all', 'atom'] else False)

        # coord
        coord_mask = is_ca | (is_heavy if self.heavy == 'all' else False) | (is_h if self.h == 'all' else False)
                    

        if self.format == 'atom_coords':
            tokens = []
            for i in range(len(atoms)):
                if atom_mask[i]: 
                    tokens += self.atom_tokenizer.tokenize([atoms[i]])
                if coord_mask[i]:
                    tokens += self.coord_tokenizer.tokenize_array(coords[i])
            order = list(range(len(tokens)))
        elif self.format == 'ordered_atoms_coords':
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
        elif self.format == 'atoms_coords':
            tokens = self.atom_tokenizer.tokenize(atoms[atom_mask]) \
                +['[XYZ]']+self.coord_tokenizer.tokenize_array(coords[coord_mask].ravel())
            order = list(range(len(tokens)))
        else:
            raise ValueError(f"Unknown {self.format=}")
        return tokens, order
    
    def vocs(self) -> set[str]:
        return self.atom_tokenizer.vocs() | self.coord_tokenizer.vocs() \
                | (set() if self.format == 'atom_coords' else {'[XYZ]'})

class Protein2PDBDataset(WrapDataset[str]):
    def __init__(self, dataset: Dataset[OBMol]):
        super().__init__(dataset)
        self.obc = OBConversion()
        self.obc.SetOutFormat('pdb')
    def __getitem__(self, idx: int):
        protein = self.dataset[idx]
        pdb = self.obc.WriteString(protein)
        return pdb


# 水素は含んでいても含んでいなくてもよいが, atomとcoordでそろえること。
class PocketTokenizeDataset(WrapDataset[tuple[list[str], list[int]]]):
    def __init__(self, pocket_data: Dataset[Pocket], *,
            heavy: AtomRepr, h: AtomRepr, 
            format: Literal['atoms_coords', 'atom_coords', 'ordered_atoms_coords'], coord_range: int):
        super().__init__(pocket_data)
        self.pocket_data = pocket_data
        self.protein_tokenizer = ProteinTokenizer(heavy=heavy, h=h, format=format, coord_range=coord_range)
        assert h == 'none', f"h must be none for pocket."

    def __getitem__(self, idx: int):
        protein = self.pocket_data[idx]
        assert len(protein.atoms) == len(protein.coord)
        return self.protein_tokenizer(protein.atoms, protein.coord)
    
    def vocs(self) -> set[str]:
        return self.protein_tokenizer.vocs()

from .molecule import MolTokenizer
class ProteinTokenizeDataset(WrapDataset[tuple[list[str], list[int]]]):
    def __init__(self, protein_data: Dataset[OBMol], *,
            heavy: AtomRepr, h: AtomRepr, format, coord_range: int, order: Literal['residue', 'can', 'ran'], base_seed: int):
        super().__init__(protein_data)
        self.protein_data = protein_data

        self.order = order
        if order == 'residue':
            self.tokenizer = ProteinTokenizer(heavy=heavy, h=h, format=format, coord_range=coord_range)
        else:
            self.tokenizer = MolTokenizer(format, h_coord=h == 'all', coord_range=coord_range)

        self.h = h
        self.base_seed = base_seed

    def __getitem__(self, idx: int):
        protein = self.protein_data[idx]

        # add/remove hydrogen
        if self.h == 'none':
            success = protein.DeleteHydrogens()
        else:
            success = protein.AddHydrogens()
        assert success

        if self.order == 'residue':
            # Order atoms
            atoms = np.array([atom.GetResidue().GetAtomID(atom).strip() for atom in OBMolAtomIter(protein)])
            residue_idxs = np.array([atom.GetResidue().GetIdx() for atom in OBMolAtomIter(protein)])
            coords = get_coord_from_mol(protein)
            orders = np.argsort(residue_idxs, kind='stable')
            atoms = atoms[orders]
            coords = coords[orders]
            # tokenize
            tokens, orders = self.tokenizer(atoms, coords)
        else:
            protein = obmol2rdmol(protein, sanitize=False) # sanitize=True raises errors but not needed for following processes
            protein = set_atom_order(protein, self.order == 'ran', get_rng(self.base_seed, idx))
            tokens, orders = self.tokenizer.tokenize(protein)
        return tokens, orders
    def vocs(self) -> set[str]:
        return self.tokenizer.vocs()

        