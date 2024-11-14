import itertools, math
from collections.abc import Iterable
from logging import getLogger
import numpy as np
import torch
from torch.utils.data import Dataset
from ..utils import logtime

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
    def __init__(self, atom_vocs: list=['CA', 'C', 'H', 'O', 'N', 'S', 'P', 'F', 'ZN'], unk_voc='[UNK]'):
        self.atom_vocs = sorted(atom_vocs, key=len, reverse=True)
        self.unk_voc = unk_voc
    def tokenize(self, atoms: list[str]):
        tokens = []
        for atom in atoms:
            for voc in self.atom_vocs:
                if atom[:len(voc)] == voc:
                    tokens.append(voc)
                    break
            else:
                self.logger.warning(f"Unknown atom {atom}")
                tokens.append(self.unk_voc)
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
                self.logger.warning(f"Unknown word {string} in {org_string}")
                tokens.append(self.unk_voc)
                string = string[1:]
        return tokens
    def vocs(self):
        return set(self.vocs_+[self.unk_voc])

class FloatTokenizer:
    logger = getLogger(f"{__module__}.{__qualname__}")

    def __init__(self, vmin: float, vmax: float, decimal: int=3):
        self.decimal = decimal
        self.vmin = float(vmin)
        self.vmax = float(vmax)

    def tokenize(self, x: float):
        x = float(x)
        self.logger.warning(f"Tokenizing {x}")
        if x < self.vmin:
            self.logger.warning(f"float value out of range: {x}")
            x = self.vmin
        if x > self.vmax:
            self.logger.warning(f"float value out of range: {x}")
            x = self.vmax
        xi, xf = str(x).split('.')
        xf = '.'+xf[:self.decimal].ljust(self.decimal, '0')

        return [xi, xf]

    def tokenize_array(self, x: Iterable[float]):
        return list(itertools.chain.from_iterable(self.tokenize(x) for x in x))
    
    def vocs(self):
        return {str(i) for i in range(max(0, math.floor(self.vmin)), math.floor(self.vmax)+1)}\
            |{'-'+str(i) for i in range(max(0, math.floor(-self.vmax)), math.floor(-self.vmin)+1)}\
            |{'.'+str(i).zfill(self.decimal) for i in range(10**self.decimal)}
