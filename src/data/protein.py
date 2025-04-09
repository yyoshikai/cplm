
from logging import getLogger
import numpy as np
from torch.utils.data import Dataset
 
from .tokenizer import ProteinAtomTokenizer, FloatTokenizer
from .data import CoordTransform, PickleLMDBDataset
from ..utils import logtime, slice_str

# net_datasetは {'atoms': list, 'coordinate': np.ndarray} を出力すればよい。
# 水素は含んでいても含んでいなくてもよいが, atomとcoordでそろえること。
class ProteinDataset(Dataset):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, net_dataset: Dataset, atom_tokenizer: ProteinAtomTokenizer, 
            coord_tokenizer: FloatTokenizer,
            coord_transform: CoordTransform, 
            atom_heavy: bool = True, atom_h: bool = False,
            coord_heavy: bool=False, coord_h: bool = False):
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

    def __getitem__(self, idx):
        data = self.net_dataset[idx]
        with logtime(self.logger, f"[{idx}]"):
            atoms = np.array(data['atoms'])
            coords = data['coordinate']
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
            return ['[POCKET]']+self.atom_tokenizer.tokenize(atoms) \
                +['[XYZ]']+self.coord_tokenizer.tokenize_array(coords.ravel())+['[END]']

    def vocs(self) -> set[str]:
        return self.atom_tokenizer.vocs()|self.coord_tokenizer.vocs()|{'[POCKET]', '[XYZ]', '[END]'}

    def __len__(self):
        return len(self.net_dataset)

class UniMolPocketDataset(Dataset):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, lmdb_path, **kwargs):
        self.dataset = PickleLMDBDataset(lmdb_path, **kwargs)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        with logtime(self.logger, f"[{idx}]"):
            data['coordinate'] = data.pop('coordinates')[0]
            return data

    def __len__(self):
        return len(self.dataset)
