import re
from typing import Literal
from logging import getLogger
import numpy as np
from torch.utils.data import Dataset
from rdkit import Chem
from openbabel import openbabel as ob
from ..chem import set_atom_order, randomize_smiles, get_coord_from_mol, ELEMENT_SYMBOLS
from ..utils.path import WORKDIR
from .data import WrapDataset, WrapTupleDataset, get_rng
from .tokenizer import StringTokenizer2, FloatTokenizer
from .protein import AtomRepr, Pocket

class MolProcessDataset(WrapDataset[Chem.Mol]):
    def __init__(self, mol_data: Dataset[Chem.Mol], base_seed: int, random: bool):
        super().__init__(mol_data)
        self.base_seed = base_seed
        self.random = random
    
    def __getitem__(self, idx: int):
        mol = self.dataset[idx]

        # MolTokenizeDatasetに移しても良いと思ったが、get_finetune_dataでRenumberAtoms後のmolが必要なのでこうしているっぽい。
        rng = get_rng(self.base_seed, idx)
        mol = set_atom_order(mol, self.random, rng)
        return mol

MAX_VALENCE = 10

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

class SmilesOrderDataset(WrapTupleDataset[tuple[str, np.ndarray]]):
    def __init__(self, mol_data: Dataset[ob.OBMol|Chem.Mol], order: Literal['residue', 'can', 'ran'], base_seed: int):
        super().__init__(mol_data, 2)
        self.order = order
        self.base_seed = base_seed
        # For OBMol
        if order != 'residue':
            self.obc = ob.OBConversion()
            self.obc.SetOutFormat('smi')
            if self.order == 'ran':
                self.obc.AddOption("C", self.obc.OUTOPTIONS)
            self.obc.AddOption("O", self.obc.OUTOPTIONS)

    def __getitem__(self, idx: int):
        mol = self.dataset[idx]
        if isinstance(mol, ob.OBMol):
            if self.order == 'residue':
                residue_idxs = np.array([atom.GetResidue().GetIdx() for atom in ob.OBMolAtomIter(mol)])
                smi = None
                orders = np.argsort(residue_idxs, kind='stable')
            else:
                smi = self.obc.WriteString(mol)
                orders = ob.toPairData(mol.GetData('SMILES Atom Order')).GetValue()
                orders = np.array([int(o) for o in orders.split(' ')])
        elif isinstance(mol, Chem.Mol):
            if self.order == 'residue':
                chain_id_is =[]
                chain_id2i = {} # 出現した順に並べる
                serial_numbers = []
                for atom in mol.GetAtoms():
                    rinfo = atom.GetPDBResidueInfo()
                    if rinfo is None:
                        rinfo = atom.GetNeighbors()[0].GetPDBResidueInfo()
                    chain_id = rinfo.GetChainId()
                    serial_numbers.append(rinfo.GetSerialNumber())
                    if chain_id not in chain_id2i:
                        chain_id2i[chain_id] = len(chain_id2i)
                    chain_id_is.append(chain_id2i[chain_id])
                orders = np.lexsort([serial_numbers, chain_id_is]) # これがopenbabelと同じ順になる
                smi = None
            else:
                if self.order == 'ran':
                    smi = randomize_smiles(mol, get_rng(self.base_seed, idx))
                else:
                    smi = Chem.MolToSmiles(mol, canonical=True)
                orders = eval(mol.GetProp('_smilesAtomOutputOrder'))
        else:
            raise ValueError(f"Unknown {type(mol)=}")
        return smi, orders

class AtomsDataset(WrapDataset[list[str]]):
    """
    Calpha炭素はCA
    それ以外は元素記号 (2文字目は小文字 Mg等)
    
    """
    def __init__(self, mol_data: Dataset[ob.OBMol|Chem.Mol]):
        super().__init__(mol_data)
    
    def __getitem__(self, idx):
        mol = self.dataset[idx]
        if isinstance(mol, ob.OBMol):
            atoms = [
                'CA' if atom.GetResidue().GetAtomID(atom) == " CA " \
                else ELEMENT_SYMBOLS[atom.GetAtomicNum()-1] 
                for atom in ob.OBMolAtomIter(mol)
            ]
        elif isinstance(mol, Chem.Mol):
            atoms = ['CA' if atom.GetPDBResidueInfo().GetName() == ' CA ' else atom.GetSymbol() for atom in mol.GetAtoms()]
        elif isinstance(mol, Pocket):
            atoms = mol.atoms
        else:
        return atoms


class CoordsDataset(WrapDataset[np.ndarray]):
    def __init__(self, mol_data: Dataset[ob.OBMol|Chem.Mol]):
        super().__init__(mol_data)
    
    def __getitem__(self, idx):
        mol = self.dataset[idx]
        if isinstance(mol, ob.OBMol):
            return get_coord_from_mol(mol)
        else:
            return mol.GetConformer().GetPositions()

class RemainValencesDataset(Dataset[list[int]]):
    """
    こちらは既にorder順に並べ替えられたものを必要とする。
    
    """
    def __init__(self, mol_data: Dataset[ob.OBMol|Chem.Mol], orders_data: Dataset[np.ndarray]):
        self.mol_data = mol_data
        self.orders_data = orders_data
    def __getitem__(self, idx: int):
        mol = self.mol_data[idx]
        orders = self.orders_data[idx]

        if isinstance(mol, ob.OBMol):
            remain_valences = []
            added_idxs = set()
            for idx in orders:
                atom = mol.GetAtomById(idx)
                remain_valence = 0
                for natom in ob.OBAtomAtomIter(atom):
                    if natom.GetId() not in added_idxs:
                        remain_valence += 1
                remain_valences.append(remain_valence)
                added_idxs.add(idx)
        else:
            remain_valences = []
            added_idxs = set()
            for o in orders:
                remain_valence = 0
                atom = mol.GetAtomWithIdx(o)
                remain_valence = 0
                for natom in atom.GetNeighbors():
                    if natom.GetIdx() not in added_idxs:
                        remain_valence += 1
                remain_valences.append(remain_valence)
                added_idxs.add(o)
        return remain_valences




class MolTokenizeDataset(Dataset[list[str]]):
    def __init__(self, 
            atoms_data: Dataset[list[str]],
            coords_data: Dataset[np.ndarray], 
            smi_data: Dataset[str],
            orders_data: Dataset[np.ndarray],
            remain_valences_data: Dataset[list[int]],
            format: Literal['atoms_coords', 'atom_coords', ], 
            coord_range: float,
            smiles_voc_dir: str,
            heavy: AtomRepr, h: AtomRepr
    ):
        self.atoms = atoms_data
        self.coords = coords_data
        self.smi = smi_data
        self.orders = orders_data
        self.remain_valences = remain_valences_data
        self.format = format
        self.smiles_voc_dir = smiles_voc_dir
        self.heavy = heavy
        self.h = h

        self.coord_tokenizer = FloatTokenizer("mol coord", -coord_range, coord_range)
        if self.format in ['smiles_coords', 'smile_corods']:
            self.smi_tokenizer = StringTokenizer2(smiles_voc_dir)
        if self.format == 'smile_coords':
            with open(f"{WORKDIR}/cplm/src/data/vocs/{smiles_voc_dir}/non_atom_tokens.txt") as f:
                self.non_atom_tokens = f.read().splitlines()


    def __getitem__(self, idx: int) -> tuple[list[str], list[int]]:
        atoms = self.atoms[idx]
        coords = self.coords[idx]
        orders = self.orders[idx]
        atom2repr = {'H': self.h, 'CA': 'all'} # others: self.heavy
        reprs = [atom2repr.get(atom, self.heavy) for atom in atoms]
        if self.format in ['atom_coords']:
            tokens = []
            for o in orders:
                if reprs[o] != 'none':
                    tokens.append(atoms[o])
                    if reprs[o] == 'all':
                        tokens += self.coord_tokenizer.tokenize_array(coords[o])
            poss = list(range(len(tokens)))
        elif self.format == 'atom_valence_coords':
            remain_valences = self.remain_valences[idx]
            tokens = []
            for o in orders:
                if reprs[o] != 'none':
                    tokens += [atoms[o], remain_valences[o]]
                    if reprs[o] == 'all':
                        tokens += self.coord_tokenizer.tokenize_array(coords[o])
            poss = list(range(len(tokens)))
        elif self.format == 'ordered_atoms_coords':
            atom_tokens = []
            coord_tokens = []
            atom_poss = []
            coord_poss = []
            pos = 0
            for o in orders:
                if reprs[o] != 'none':
                    atom_tokens.append(atoms[o])
                    atom_poss.append(pos)
                    pos += 1
                    if reprs[o] == 'all':
                        coord_tokens += self.coord_tokenizer.tokenize_array(coords[o])
                        coord_poss.append(list(range(pos, pos+6)))
                        pos += 6
            tokens = atom_tokens + ['[XYZ]'] + coord_tokens
            poss = atom_poss + [pos] + coord_poss

        elif self.format == 'atoms_coords':
            tokens = [atoms[o] for o in orders if reprs[o] != 'none']+['[XYZ]']
            coord_orders = [o for o in orders if reprs[o] == 'all']
            tokens += self.coord_tokenizer.tokenize_array(coords[coord_orders].ravel())
            order = list(range(len(tokens)))
        elif self.format == 'smiles_coords':
            smi = self.smi[idx]
            tokens = self.smi_tokenizer.tokenize(smi)+['[XYZ]']
            coord_orders = [o for o in orders if reprs[o] == 'all']
            tokens += ['[XYZ]']+self.coord_tokenizer.tokenize_array(coords[coord_orders].ravel())
            order = list(range(len(tokens)))
        elif self.format == 'smile_coords':
            smi = self.smi[idx]
            smi_tokens = self.smi_tokenizer.tokenize(smi)            
            tokens = []
            i = 0
            for smi_token in smi_tokens:
                tokens.append(smi_token)
                if smi_token not in self.non_atom_tokens:
                    o = orders[i]
                    if reprs[o] == 'all':
                        tokens += self.coord_tokenizer.tokenize_array(coords[o])
                    i += 1
            assert i == len(atoms)
            order = list(range(len(tokens)))
        else:
            raise ValueError(f"Unknown format={self.format}")
        return tokens, order

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
