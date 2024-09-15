import pickle

from torch.utils.data import Dataset
import lmdb

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, key_is_indexed=False):
        self.env = lmdb.open(lmdb_path, subdir=False, readonly=True,
            lock=False, readahead=False, meminit=False, max_readers=256)
        self.txn = self.env.begin()
        self.key_is_indexed = key_is_indexed
        if not self.key_is_indexed:
            self.keys = list(self.txn.cursor().iternext(values=False))
        else:
            self.keys = None
    
    def __getitem__(self, idx):
        if self.key_is_indexed:
            key = str(idx).encode('ascii')
        else:
            key = self.keys[idx]
        data = pickle.loads(self.txn.get(key))
        return data

    def __len__(self):
        return self.env.stat()['entries']

class MoleculeTokenizer:
    def __init__(self):
        pass

    def tokenize(self) -> list[int]:
        pass

    def detokenize(self) -> list[int]:
        pass



