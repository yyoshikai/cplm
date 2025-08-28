
from logging import getLogger
import numpy as np
from torch.utils.data import Dataset
 
from ..tokenizer import ProteinAtomTokenizer, FloatTokenizer, TokenizeDataset, ArrayTokenizeDataset
from ..coord_transform import CoordTransform
from ..lmdb import PickleLMDBDataset
from ...utils import slice_str
from ..finetune2 import Protein

# 水素は含んでいても含んでいなくてもよいが, atomとcoordでそろえること。
class ProteinDataset(Dataset):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, net_dataset: Dataset[Protein], 
            coord_transform: CoordTransform, 
            atom_heavy: bool = True, atom_h: bool = False,
            coord_heavy: bool=False, coord_h: bool = False):
        """
        Pretrain用のポケットデータを生成する。
        coord_heavy: ca, heavy, h のうちheavyのcoordを抽出するかどうか。
        """
        self.net_dataset = net_dataset
        self.coord_transform = coord_transform
        self.atom_heavy = atom_heavy
        self.atom_h = atom_h
        self.coord_heavy = coord_heavy
        self.coord_h = coord_h
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
        coord_position = np.where(coord_mask[atom_mask])[0]

        return atoms, coords, coord_position

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