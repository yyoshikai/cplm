import numpy as np
from torch.utils.data import Dataset
from rdkit import Chem
from .data import WrapDataset, get_rng
from .tokenizer import SmilesTokenizer, FloatTokenizer

class MolTokenizeDataset(WrapDataset[list[str]]):
    def __init__(self, mol_data: Dataset[Chem.Mol], base_seed: int, h_atom: bool, h_coord: bool, randomize: bool, coord_range: float, coord_follow_atom: bool, atoms: bool):
        super().__init__(mol_data)
        self.mol_data = mol_data
        self.h_atom = h_atom
        self.h_coord = h_coord
        self.coord_follow_atom = coord_follow_atom
        self.atoms = atoms
        assert not ((not self.h_atom) and self.h_coord), 'Not Implemented.'
        assert not ((not self.atoms) and self.coord_follow_atom), 'Not Implemented'
        self.randomize = randomize
        self.seed = base_seed
        if not self.atoms:
            self.smi_tokenizer = SmilesTokenizer()
        self.coord_tokenizer = FloatTokenizer("mol coord", -coord_range, coord_range)
        
    def __getitem__(self, idx: int):
        mol = self.mol_data[idx]
        rng = get_rng(self.seed, idx)

        # remove/add hydrogen
        if self.h_atom:
            mol = Chem.AddHs(mol, addCoords=True)
        else:
            mol = Chem.RemoveHs(mol)
        
        # randomize
        if self.randomize:
            idxs = np.arange(mol.GetNumAtoms(), dtype=int)
            rng.shuffle(idxs)
            mol = Chem.RenumberAtoms(mol, idxs.tolist())
        smi = Chem.MolToSmiles(mol, canonical=not self.randomize)
        try:
            atom_idxs = eval(mol.GetProp('_smilesAtomOutputOrder'))
        except Exception as e:
            print(f"{mol=}", flush=True)
            raise e
        
        coords = mol.GetConformer().GetPositions()
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        tokens = []
        if self.atoms and self.coord_follow_atom:
            for i in range(mol.GetNumAtoms()):
                ai = atom_idxs[i]
                tokens.append(symbols[ai])
                if self.h_coord or symbols[ai] != 'H':
                    tokens += self.coord_tokenizer.tokenize_array(coords[ai])
        elif self.atoms:
            coord_atom_idxs = [ai for ai in atom_idxs if (self.h_coord or symbols[ai] != 'H')]
            tokens = [symbols[ai] for ai in atom_idxs] + ['[XYZ]'] \
                    + self.coord_tokenizer.tokenize_array(coords[coord_atom_idxs].ravel())
        else: # smiles
            tokens = self.smi_tokenizer.tokenize(smi)
            shown_coords = np.concatenate([coords[ai] 
                    for ai in atom_idxs if (symbols[ai] != 'H' or self.h_coord)])
            tokens += ['[XYZ]']+self.coord_tokenizer.tokenize_array(shown_coords)
        
        return tokens

    def vocs(self) -> set[str]:
        if self.atoms:
            table = Chem.GetPeriodicTable()
            vocs = {table.GetElementSymbol(i) for i in range(1, 119)}
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