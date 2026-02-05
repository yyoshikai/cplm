import numpy as np
from torch.utils.data import Dataset
from rdkit import Chem
from .data import WrapDataset, get_rng
from .tokenizer import SmilesTokenizer, FloatTokenizer

def element_symbols() -> list[str]:
    table = Chem.GetPeriodicTable()
    return [table.GetElementSymbol(i) for i in range(1, 119)]

class MolProcessDataset(WrapDataset[Chem.Mol]):
    def __init__(self, mol_data: Dataset[Chem.Mol], base_seed: int, h: bool, random: bool):
        super().__init__(mol_data)
        self.base_seed = base_seed
        self.h = h
        self.random = random
    
    def __getitem__(self, idx: int):
        mol = self.dataset[idx]

        # remove/add hydrogen
        if self.h:
            mol = Chem.AddHs(mol, addCoords=True)
        else:
            mol = Chem.RemoveHs(mol)

        # randomize/canonicalize
        # refer /workspace/cplm/experiments/tests/source.ipynb "260204 canonical"
        if self.random:
            rng = get_rng(self.base_seed, idx)
            idxs = np.arange(mol.GetNumAtoms(), dtype=int)
            rng.shuffle(idxs)
            ran = Chem.MolToSmiles(mol, canonical=False)
        else:
            can = Chem.MolToSmiles(mol, canonical=True)
        mol = Chem.RenumberAtoms(mol, eval(mol.GetProp('_smilesAtomOutputOrder')))
        return mol


class MolTokenizeDataset(WrapDataset[tuple[list[str], list[int]]]):
    def __init__(self, mol_data: Dataset[Chem.Mol], *, coord_follow_atom: bool, atoms: bool, atom_order: bool, coord_range: float, no_h_coord: bool=False):
        super().__init__(mol_data)
        self.mol_data = mol_data
        self.h_coord = not no_h_coord
        self.coord_follow_atom = coord_follow_atom
        self.atoms = atoms
        self.atom_order = atom_order
        assert not ((not self.atoms) and self.coord_follow_atom), 'Not Implemented'
        if self.atom_order:
            assert self.atoms and not self.coord_follow_atom
        if not self.atoms:
            self.smi_tokenizer = SmilesTokenizer()
        self.coord_tokenizer = FloatTokenizer("mol coord", -coord_range, coord_range)
        
    def __getitem__(self, idx: int):
        mol = self.mol_data[idx]

        smi = Chem.MolToSmiles(mol, canonical=False)
        atom_idxs = eval(mol.GetProp('_smilesAtomOutputOrder'))
        
        coords = mol.GetConformer().GetPositions()
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        tokens = []
        if self.atoms:
            if self.coord_follow_atom:
                for i in range(mol.GetNumAtoms()):
                    ai = atom_idxs[i]
                    tokens.append(symbols[ai])
                    if self.h_coord or symbols[ai] != 'H':
                        tokens += self.coord_tokenizer.tokenize_array(coords[ai])
                order = list(range(len(tokens)))
            elif self.atom_order:
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
            else:
                coord_atom_idxs = [ai for ai in atom_idxs if (self.h_coord or symbols[ai] != 'H')]
                tokens = [symbols[ai] for ai in atom_idxs] + ['[XYZ]'] \
                        + self.coord_tokenizer.tokenize_array(coords[coord_atom_idxs].ravel())
                order = list(range(len(tokens)))
        else: # smiles
            tokens = self.smi_tokenizer.tokenize(smi)
            shown_coords = np.concatenate([coords[ai] 
                    for ai in atom_idxs if (symbols[ai] != 'H' or self.h_coord)])
            tokens += ['[XYZ]']+self.coord_tokenizer.tokenize_array(shown_coords)
            order = list(range(len(tokens)))
        return tokens, order

    def vocs(self) -> set[str]:
        if self.atoms:
            return set(element_symbols())
        else:
            vocs = self.smi_tokenizer.vocs()
        vocs |= self.coord_tokenizer.vocs()
        if not self.coord_follow_atom:
            vocs.add('[XYZ]')
        return vocs

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