import itertools, math
from collections.abc import Iterable
from collections import defaultdict
from logging import getLogger
import numpy as np
import torch
from torch.utils.data import Dataset
from contextlib import nullcontext
from ..utils import logtime, LOGTIME

class VocEncoder:
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, vocs: Iterable[str]):
        assert '[PAD]' not in vocs
        self.i2voc = ['[PAD]']+sorted(set(vocs))
        self.pad_token = 0
        self.voc2i = {voc: i for i, voc in enumerate(self.i2voc)}
    
    def encode(self, words: str):
        # try:
            return [self.voc2i[t] for t in words]
        # except KeyError as e:
        #     self.logger.error(f"KeyError in {words}")
        #     raise e
            
    def decode(self, tokens: Iterable[int]):
        return [self.i2voc[t] for t in 
            itertools.takewhile(lambda x: x!= self.pad_token, tokens)]

    @property
    def voc_size(self) -> int:
        return len(self.i2voc)

class TokenEncodeDataset(Dataset):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, dataset, encoder: VocEncoder):
        self.dataset = dataset
        self.encoder = encoder
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        with logtime(self.logger, f"[{idx}]"):
            data = torch.tensor(self.encoder.encode(data), dtype=torch.long)
        return data

    def __len__(self):
        return len(self.dataset)

class ProteinAtomTokenizer:
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, atom_vocs: list=['CA', 'C', 'H', 'O', 'N', 'S', 'P', 'F', 'ZN', 'BR', 'MG'], unk_voc='[UNK]', log_interval: int=1000000):
        self.atom_vocs = sorted(atom_vocs, key=len, reverse=True)
        self.unk_voc = unk_voc
        self.n_tokenized = 0
        self.unk_count = defaultdict(int)
        self.log_interval = log_interval
    def tokenize(self, atoms: list[str]):
        tokens = []
        for atom in atoms:
            for voc in self.atom_vocs:
                if atom[:len(voc)] == voc:
                    tokens.append(voc)
                    break
            else:
                self.unk_count[atom] += 1
                tokens.append(self.unk_voc)
        self.n_tokenized += len(atoms)
        if (self.n_tokenized%self.log_interval) < len(atoms):
            if len(self.unk_count) == 0:
                self.logger.info(f"No unknown atoms in {self.n_tokenized} atoms.")
            else:
                self.logger.info(f"Unknown atoms in {self.n_tokenized} atoms:")
                for atom, n in sorted(self.unk_count.items(), key=lambda x: x[1], reverse=True):
                    self.logger.info(f"  {atom}: {n}")
        
        return tokens

    def vocs(self):
        return set(self.atom_vocs+[self.unk_voc])
                
class StringTokenizer:
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, vocs: list, unk_voc='[UNK]'):
        self.vocs_ = sorted(vocs, key=len, reverse=True)
        self.unk_voc = unk_voc
    def tokenize(self, string: str):
        org_string = string
        tokens = []
        while len(string) > 0:
            for voc in self.vocs_:
                if string[:len(voc)] == voc:
                    tokens.append(voc)
                    string = string[len(voc):]
                    break
            else:
                self.logger.info(f"Unknown word {string} in {org_string}")
                tokens.append(self.unk_voc)
                string = string[1:]
        return tokens
    def vocs(self):
        return set(self.vocs_+[self.unk_voc])

class FloatTokenizer:
    logger = getLogger(f"{__module__}.{__qualname__}")

    def __init__(self, vmin: float, vmax: float, decimal: int=3, log_interval: int=1000000):
        self.decimal = decimal
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.n_tokenized = 0
        self.n_over = 0
        self.n_under = 0
        self.log_interval = log_interval
        self.float_format = "{:.0"+str(self.decimal)+"f}"

    def tokenize(self, x: float):
        x = float(x)
        if x < self.vmin:
            self.n_over += 1
            x = self.vmin
        if x > self.vmax:
            self.n_under += 1
            x = self.vmax
        self.n_tokenized += 1
        if self.n_tokenized % self.log_interval == 0:
            self.logger.info(f"{self.n_over}/{self.n_tokenized} are over vmax, {self.n_under}/{self.n_tokenized} are under vmin")
        x = self.float_format.format(x)
        xi = x[:-4]
        xf = x[-4:]

        return [xi, xf]

    def tokenize_array(self, x: Iterable[float]):
        return list(itertools.chain.from_iterable(self.tokenize(x) for x in x))
    
    def vocs(self):
        ivmin = int(self.float_format.format(self.vmin).split('.')[0])
        ivmax = int(self.float_format.format(self.vmax).split('.')[0])
        vocs = {str(i) for i in range(ivmin, ivmax+1)}
        if self.vmin < 0: vocs.add('-0')
        vocs |= {'.'+str(i).zfill(self.decimal) for i in range(10**self.decimal)}
        return vocs
