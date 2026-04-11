import re
from typing import Literal
from logging import getLogger
import numpy as np
from torch.utils.data import Dataset
from rdkit import Chem
from openbabel import openbabel as ob
from ..chem import set_atom_order
from .data import WrapDataset, get_rng
from .tokenizer import SmilesTokenizer, FloatTokenizer
from .protein import AtomRepr

def element_symbols() -> list[str]:
    table = Chem.GetPeriodicTable()
    return [table.GetElementSymbol(i) for i in range(1, 119)]


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
                Chem.AddHs(mol, addCoords=True)
            else:
                Chem.RemoveHs(mol)
        return mol

class MolTokenizer:
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, format: AtomRepr, h_coord: bool, coord_range: float, smiles_voc_file: str):        
        self.h_coord = h_coord
        self.format = format
        if self.format == 'smiles_coords':
            self.smi_tokenizer = SmilesTokenizer(smiles_voc_file)
        if self.format == 'smile_coords':
            self.atom_pattern = re.compile(r"Cl|Br|[bcnopsHBCNOFPSI]|\[se|\[as|\[si|\[te|\[H\]|\[He|\[Li|\[Be|\[Ne|\[Na|\[Mg|\[Al|\[Si|\[Ar|\[Ca|\[Sc|\[Ti|\[Cr|\[Mn|\[Fe|\[Co|\[Ni|\[Cu|\[Zn|\[Ga|\[Ge|\[As|\[Se|\[Kr|\[Rb|\[Sr|\[Zr|\[Nb|\[Mo|\[Tc|\[Ru|\[Rh|\[Pd|\[Ag|\[Cd|\[In|\[Sn|\[Sb|\[Te|\[Xe|\[Cs|\[Ba|\[La|\[Ce|\[Pr|\[Nd|\[Pm|\[Sm|\[Eu|\[Gd|\[Tb|\[Dy|\[Ho|\[Er|\[Tm|\[Yb|\[Lu|\[Hf|\[Ta|\[Re|\[Os|\[Ir|\[Pt|\[Au|\[Hg|\[Tl|\[Pb|\[Bi|\[Po|\[At|\[Rn|\[Fr|\[Ra|\[Ac|\[Th|\[Pa|\[Np|\[Pu|\[Am|\[Cm|\[Bk|\[Cf|\[Es|\[Fm|\[Md|\[No|\[Lr|\[Rf|\[Db|\[Sg|\[Bh|\[Hs|\[Mt|\[Ds|\[Rg|\[Cn|\[Nh|\[Fl|\[Mc|\[Lv|\[Ts|\[Og|\[K|\[V|\[Y|\[W|\[U")
        self.coord_tokenizer = FloatTokenizer("mol coord", -coord_range, coord_range)

        self.logged = False

    def tokenize(self, mol: Chem.Mol):
        
        smi = Chem.MolToSmiles(mol, canonical=False)
        if not self.logged:
            self.logger.debug(f"{smi=}")
            self.logged = True
        atom_idxs = eval(mol.GetProp('_smilesAtomOutputOrder'))
        
        coords = mol.GetConformer().GetPositions()
        atoms = list(mol.GetAtoms())
        symbols = [atom.GetSymbol() for atom in atoms]
        tokens = []
        if self.format == 'atom_coords':
            for i in range(mol.GetNumAtoms()):
                ai = atom_idxs[i]
                tokens.append(symbols[ai])
                if self.h_coord or symbols[ai] != 'H':
                    tokens += self.coord_tokenizer.tokenize_array(coords[ai])
            order = list(range(len(tokens)))
        elif self.format == 'atom_valence_coords':
            tokenized_atom_idxs = []
            for ai in atom_idxs:
                tokens.append(symbols[ai])
                remain_valence = 0
                for bond in atoms[ai].GetBonds():
                    other_idx = bond.GetBeginAtomIdx()
                    if other_idx == ai:
                        other_idx = bond.GetEndAtomIdx()
                    if other_idx not in tokenized_atom_idxs:
                        remain_valence += 1
                tokens.append(str(remain_valence))
                if self.h_coord or symbols[ai] != 'H':
                    tokens += self.coord_tokenizer.tokenize_array(coords[ai])
                tokenized_atom_idxs.append(ai)
            order = list(range(len(tokens)))
        elif self.format == 'ordered_atoms_coords':
            atom_tokens = []
            coord_tokens = []
            atom_order = []
            coord_order = []
            x = 0
            for i in range(len(symbols)):
                atom_tokens.append(symbols[i])
                atom_order.append(x)
                x += 1
                if self.h_coord or symbols[i] != 'H':
                    coord_token = self.coord_tokenizer.tokenize_array(coords[i])
                    coord_tokens += coord_token
                    coord_order += list(range(x, x+len(coord_token)))
                    x += len(coord_token)
            tokens = atom_tokens+['[XYZ]']+coord_tokens
            order = atom_order+[x]+coord_order
        elif self.format == 'atoms_coords':
            coord_atom_idxs = [ai for ai in atom_idxs if (self.h_coord or symbols[ai] != 'H')]
            tokens = [symbols[ai] for ai in atom_idxs] + ['[XYZ]'] \
                    + self.coord_tokenizer.tokenize_array(coords[coord_atom_idxs].ravel())
            order = list(range(len(tokens)))
        elif self.format == 'smiles_coords':
            tokens = self.smi_tokenizer.tokenize(smi)
            shown_coords = np.concatenate([coords[ai] 
                    for ai in atom_idxs if (symbols[ai] != 'H' or self.h_coord)])
            tokens += ['[XYZ]']+self.coord_tokenizer.tokenize_array(shown_coords)
            order = list(range(len(tokens)))
        elif self.format == 'smile_coords':
            atom_matches = list(re.finditer(self.atom_pattern, smi))
            assert len(atom_matches) == len(atom_idxs)
            smi_tokens = self.smi_tokenizer.tokenize(smi)
            tokens = []
            cur_smi_size = 0
            cur_match_idx = 0
            for smi_token in smi_tokens:
                tokens.append(smi_token)
                cur_smi_size += len(smi_token)
                if cur_smi_size >= atom_matches[cur_match_idx]+1:
                    ai = atom_idxs[cur_match_idx]
                    if symbols[ai] != 'H' or self.h_coord:
                        tokens += self.coord_tokenizer.tokenize_array(coords[atom_idxs[cur_match_idx]])
                    cur_match_idx += 1
            order = list(range(len(tokens)))
        else:
            raise ValueError(f"Unknown format={self.format}")
        return tokens, order
    def vocs(self):
        if self.format == 'atom_coords':
            return set(element_symbols()) | self.coord_tokenizer.vocs()
        elif self.format == 'atom_valence_coords':
            return set(element_symbols()) | self.coord_tokenizer.vocs() | {str(i) for i in range(MAX_VALENCE)}
        elif self.format in ['ordered_atoms_coords', 'atoms_coords']:
            return set(element_symbols()) | self.coord_tokenizer.vocs() | {'[XYZ]'}
        elif self.format == 'smiles_coords':
            return self.smi_tokenizer.vocs() | self.coord_tokenizer.vocs() | {'[XYZ]'}



class MolTokenizeDataset(WrapDataset[tuple[list[str], list[int]]]):
    def __init__(self, mol_data: Dataset[Chem.Mol], *, format: Literal['smiles_coords', 'atoms_coords', 'atom_coords', 'ordered_atoms_coords'], coord_range: float, smiles_voc_file: str, h_coord: bool=True):
        super().__init__(mol_data)
        self.mol_data = mol_data
        self.mol_tokenizer = MolTokenizer(format, h_coord, coord_range, smiles_voc_file)
        
    def __getitem__(self, idx: int):
        return self.mol_tokenizer.tokenize(self.mol_data[idx])

    def vocs(self) -> set[str]:
        return self.mol_tokenizer.vocs()

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