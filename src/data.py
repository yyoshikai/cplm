import pickle
from logging import getLogger

from torch.utils.data import Dataset
import lmdb

from .tokenizer import MoleculeTokenizer
from .lmdb import load_lmdb

class LMDBDataset(Dataset):
    logger = getLogger(f"{__module__}.{__qualname__}")

    def __init__(self, lmdb_path, key_is_indexed=False):
        self.env, self.txn = load_lmdb(lmdb_path) 
        self.key_is_indexed = key_is_indexed
        self._lazy_keys = None
    
    def __getitem__(self, idx):
        return pickle.loads(self.txn.get(self.key(idx)))
    
    def key(self, idx):
        if self.key_is_indexed:
            return str(idx).encode('ascii')
        else:
            if self._lazy_keys is None:
                self.logger.info("Getting all key list...")
                self._lazy_keys = list(self.txn.cursor().iternext(values=False))
                self.logger.info("Done.")
            return self._lazy_keys[idx]

    def __len__(self):
        return self.env.stat()['entries']


class ChemDataset(Dataset):
    def __init__(self, split):
        self.lmdbdata = LMDBDataset(f"../cheminfodata/unimol/pockets/{split}.lmdb", 
                key_is_indexed=True)
        self.tokenizer = MoleculeTokenizer()
    
    def __getitem__(self, idx):
        data = self.lmdbdata[idx]
        return data

