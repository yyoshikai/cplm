import io
import itertools as itr
from dataclasses import dataclass
from typing import Literal
import numpy as np
from openbabel.openbabel import OBMol, OBMolAtomIter, OBConversion
from rdkit import Chem
from Bio import PDB
from Bio.PDB.Residue import Residue
from torch.utils.data import Dataset
from ..utils import slice_str
from ..chem import get_coord_from_mol, obmol2rdmol, set_atom_order, obmol2pdb, pdb2obmol
from .data import WrapDataset, get_rng
from .tokenizer import FloatTokenizer, ProteinAtomTokenizer

AtomRepr = Literal['none', 'atom', 'all']

non_metals = [
    'H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Si', 'P', 'S', 'Cl', 'Ar', 'As', 'Se', 'Br', 'Kr',
    'Te', 'I', 'Xe', 'At', 'Rn'
]

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

class _ProteinProcessSelect(PDB.Select):
    def __init__(self, ion: bool, ligand: bool, water: bool):
        self.ion = ion
        self.ligand = ligand
        self.water = water
    def accept_residue(self, residue: Residue):
        id0 = residue.get_id()[0]
        if id0 == ' ': # amino acid
            return True
        elif id0 == 'W':
            return self.water
        elif id0[:2] == 'H_':
            if len(id0[2:]) <= 2:
                return self.ion
            else:
                return self.ligand
        else:
            raise ValueError


class ProteinProcessDataset(WrapDataset[OBMol|Chem.Mol]):
    def __init__(self, protein_data: Dataset[OBMol|Chem.Mol], ion: bool, ligand: bool, water: bool):
        super().__init__(protein_data)
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
    def __init__(self, dataset: Dataset[OBMol|Chem.Mol]):
        super().__init__(dataset)
        self.obc = OBConversion()
        self.obc.SetOutFormat('pdb')
    def __getitem__(self, idx: int):
        protein = self.dataset[idx]
        if isinstance(protein, OBMol):
            pdb = self.obc.WriteString(protein)
        else:
            pdb = Chem.MolToPDBBlock(protein)
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
            heavy: AtomRepr, h: AtomRepr, format, coord_range: int, smiles_voc_dir, order: Literal['residue', 'can', 'ran'], base_seed: int):
        super().__init__(protein_data)
        self.protein_data = protein_data

        self.order = order
        if order == 'residue':
            self.tokenizer = ProteinTokenizer(heavy=heavy, h=h, format=format, coord_range=coord_range)
        else:
            self.tokenizer = MolTokenizer(format, h_coord=h == 'all', coord_range=coord_range, smiles_voc_dir=smiles_voc_dir)

        self.h = h
        self.base_seed = base_seed

    def __getitem__(self, idx: int):
        protein = self.protein_data[idx]

        if isinstance(protein, OBMol):
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
        else:

            if self.order == 'residue':
                chain_id_is =[]
                chain_id2i = {} # 出現した順に並べる
                serial_numbers = []
                for atom in protein.GetAtoms():
                    rinfo = atom.GetPDBResidueInfo()
                    if rinfo is None:
                        rinfo = atom.GetNeighbors()[0].GetPDBResidueInfo()
                    chain_id = rinfo.GetChainId()
                    serial_numbers.append(rinfo.GetSerialNumber())
                    if chain_id not in chain_id2i:
                        chain_id2i[chain_id] = len(chain_id2i)
                    chain_id_is.append(chain_id2i[chain_id])
                orders = np.lexsort([serial_numbers, chain_id_is]) # これがopenbabelと同じ順になる

                atoms = np.array([atom.GetPDBResidueInfo().GetName().strip() for atom in protein.GetAtoms()])
                coords = protein.GetConformer().GetPositions()

                atoms = atoms[orders]
                coords = coords[orders]
                tokens, orders = self.tokenizer(atoms, coords)
            else:
                protein = set_atom_order(protein, self.order == 'ran', get_rng(self.base_seed, idx))
                tokens, orders = self.tokenizer.tokenize(protein)
        return tokens, orders
    def vocs(self) -> set[str]:
        return self.tokenizer.vocs()

        