from logging import getLogger
from functools import lru_cache
import torch
from torch.utils.data import Dataset
from .data import LMDBDataset, CoordTransform
from ..tokenizer import MoleculeProteinTokenizer

# Pretrain時, 分子の処理用のデータセット
class MoleculeDataset(Dataset):
    def __init__(self, net_dataset: Dataset, 
            coord_transform: CoordTransform, 
            tokenizer: MoleculeProteinTokenizer):
        self.net_dataset = net_dataset
        self.coord_transform = coord_transform
        self.tokenizer = tokenizer

    def __geittem__(self, idx):
        data = self.net_dataset[idx]
        smi = data['smi']
        coord = data['coordinate']
        coord = self.coord_transform(coord)
        tokens = self.tokenizer.tokenize_smi(smi)+self.tokenizer.tokenize_coord(coord)
        return torch.tensor(tokens, dtype=torch.long)
        
    def __len__(self):
        return len(self.net_dataset)

class UniMolLigandDataset(Dataset):
    logger = getLogger(f'{__module__}.{__qualname__}')
    def __init__(self, lmdb_path, n_conformer, **kwargs):
        self.net_dataset = LMDBDataset(lmdb_path, key_is_indexed=True, **kwargs)
        self.n_conformer = n_conformer

    @lru_cache(maxsize=1)    
    def __getitem__(self, idx):
        self.logger.debug(f"__getitem__({idx})")
        mol_idx, conformer_idx = divmod(idx, self.n_conformer)
        data = self.net_dataset[mol_idx]
        return {'smi': data['smi'], 'coordinate': data['coordinates'][conformer_idx]}
    
    def __len__(self):
        return len(self.net_dataset) * self.n_conformer
