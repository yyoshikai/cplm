
from logging import getLogger
import numpy as np
from torch.utils.data import Dataset
 
from ..tokenizer import ProteinAtomTokenizer, FloatTokenizer, TokenizeDataset, ArrayTokenizeDataset
from ..coord_transform import CoordTransform
from ..lmdb import PickleLMDBDataset
from ...utils import slice_str
from ..finetune2 import Protein

# net_datasetは {'atoms': list, 'coordinate': np.ndarray} を出力すればよい。
# 水素は含んでいても含んでいなくてもよいが, atomとcoordでそろえること。
class ProteinDataset(Dataset):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, net_dataset: Dataset[Protein], atom_tokenizer: ProteinAtomTokenizer, 
            coord_tokenizer: FloatTokenizer,
            coord_transform: CoordTransform, 
            atom_heavy: bool = True, atom_h: bool = False,
            coord_heavy: bool=False, coord_h: bool = False, 
            coord_follow_atom: bool=False):
        """
        Pretrain用のポケットデータを生成する。
        coord_heavy: ca, heavy, h のうちheavyのcoordを抽出するかどうか。
        """
        self.net_dataset = net_dataset
        self.atom_tokenizer = atom_tokenizer
        self.coord_tokenizer = coord_tokenizer
        self.coord_transform = coord_transform
        self.atom_heavy = atom_heavy
        self.atom_h = atom_h
        self.coord_heavy = coord_heavy
        self.coord_h = coord_h
        self.coord_follow_atom = coord_follow_atom
        assert not (self.coord_heavy and not self.atom_heavy)
        assert not (self.coord_h and not self.atom_h)

    def __getitem__(self, idx):
        data = self.net_dataset[idx]
        atoms = data.atoms
        coords = data.coord

        assert len(atoms) == len(coords)

        # calc mask
        is_ca = atoms == 'CA'
        is_h = slice_str(atoms, 1) == 'H'
        is_heavy = (~is_ca)&(~is_h)

        atom_mask = is_ca.copy()
        if self.atom_heavy: atom_mask |= is_heavy
        if self.atom_h: atom_mask |= is_h
        atoms = atoms[atom_mask]
        coord_mask = is_ca.copy()
        if self.coord_heavy: coord_mask |= is_heavy
        if self.coord_h: coord_mask |= is_h
        coords = coords[coord_mask]
        
        coords = self.coord_transform(coords)
        i_coord2i_atom = np.where(coord_mask[atom_mask])[0]

        return atoms, coords, i_coord2i_atom

        # tokenize
        atom_tokens = self.atom_tokenizer.tokenize(atoms)
        coord_tokens = self.coord_tokenizer.tokenize_array(coords.ravel())

        if self.coord_follow_atom:
            coord_mask = coord_mask[atom_mask]
            tokens = ['[POCKET]']
            for atom_token, has_coord in zip(atom_tokens, coord_mask):
                tokens.append(atom_token)
                if has_coord:
                    tokens += coord_tokens[:6]
                    coord_tokens = coord_tokens[6:]
            assert len(coord_tokens) == 0
            tokens.append('[END]')
            return tokens
        else:
            return ['[POCKET]']+atom_tokens+['[XYZ]']+coord_tokens+['[END]']

    def vocs(self) -> set[str]:
        return self.atom_tokenizer.vocs()|self.coord_tokenizer.vocs()|{'[POCKET]', '[XYZ]', '[END]'}

    def __len__(self):
        return len(self.net_dataset)

class UniMolPocketDataset(Dataset[Protein]):
    def __init__(self, lmdb_path, **kwargs):
        self.dataset = PickleLMDBDataset(lmdb_path, **kwargs)
    
    def __getitem__(self, idx) -> Protein:
        data = self.dataset[idx]
        atoms = np.array(data['atoms'])
        coord =  data.pop('coordinates')[0] # * np.array([0, 1, 2])
        return Protein(atoms=atoms, coord=coord)

    def __len__(self):
        return len(self.dataset)

class CoordFollowDataset(Dataset):
    def __init__(self, pocket_atom: TokenizeDataset, pocket_coord: ArrayTokenizeDataset, i_coord2i_atom: Dataset[np.ndarray]):
        self.pocket_atom = pocket_atom
        self.pocket_coord = pocket_coord
        self.i_coord2i_atom = i_coord2i_atom
        assert len(self.pocket_atom) == len(self.pocket_coord)
    
    def __getitem__(self, idx: int):
        pocket_atom = self.pocket_atom[idx]
        pocket_coord = self.pocket_coord[idx]
        i_coord2i_atom = self.i_coord2i_atom[idx]
        assert len(i_coord2i_atom)*6 == len(pocket_coord)
        output = []
        i_coord = 0
        for i_atom in range(len(pocket_atom)):
            output.append(pocket_atom[i_atom])
            if i_coord2i_atom[i_coord] == i_atom:
                output += pocket_coord[i_coord*6:(i_coord+1)*6]
                i_coord += 1
        return output

    def __len__(self):
        return len(self.pocket_atom)
    
    def vocs(self):
        return self.pocket_atom.vocs()|self.pocket_coord.vocs()