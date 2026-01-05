import itertools, re, math
from collections.abc import Iterable
from collections import defaultdict
from logging import getLogger
from typing import TypeVar

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .data import WorkerAggregator, is_main_worker, WrapDataset, TupleDataset
from ..utils import should_show
from ..utils.path import WORKDIR


T_co = TypeVar('T_co', covariant=True)

class VocEncoder:
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, vocs: Iterable[str]):
        assert '[PAD]' not in vocs
        self.i2voc = ['[PAD]']+sorted(set(vocs))
        self.pad_token = 0
        self.voc2i = {voc: i for i, voc in enumerate(self.i2voc)}
    
    def encode(self, words: str):
        try:
            return [self.voc2i[t] for t in words]
        except KeyError as e:
            self.logger.error(f"KeyError in {words}")
            raise e
            
    def decode(self, tokens: Iterable[int]):
        return [self.i2voc[t] for t in 
            itertools.takewhile(lambda x: x!= self.pad_token, tokens)]

    @property
    def voc_size(self) -> int:
        return len(self.i2voc)
    
    @classmethod
    def from_i2voc(cls, vocs: list[str]) -> 'VocEncoder':
        assert vocs[0] == '[PAD]'
        return VocEncoder(vocs[1:])

class TokenEncodeDataset(Tensor):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, dataset, encoder: VocEncoder):
        self.dataset = dataset
        self.encoder = encoder
    
    def __getitem__(self, idx):
        token = self.encoder.encode(self.dataset[idx])
        token = torch.tensor(token, dtype=torch.long)
        return token

    def __len__(self):
        return len(self.dataset)

class Tokenizer:
    def __init__(self):
        pass

    def tokenize(self, item) -> list[str]:
        raise NotImplementedError
    
    def vocs(self) -> set[str]:
        raise NotImplementedError

class ProteinAtomTokenizer(Tokenizer):
    unk_logger = getLogger(f"unk.{__module__}.{__qualname__}")
    def __init__(self, atom_vocs: list=['CA', 'C', 'H', 'O', 'N', 'S', 'P', 'F', 'ZN', 'BR', 'MG'], unk_voc='[UNK]'):
        self.atom_vocs = sorted(atom_vocs, key=len, reverse=True)
        self.unk_voc = unk_voc

        # Process-specific attributes
        self.n_tokenized = 0
        self.unk_count = defaultdict(int)
        def agg_count(x: defaultdict[str, int], y: defaultdict[str, int]):
            for k, v in y.items():
                x[k] += v
            return x
        self.unk_count_agg = WorkerAggregator(defaultdict(int), agg_count)
        self.n_tokenized_agg = WorkerAggregator(0, lambda x, y: x+y)

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

        # Log unknown tokens
        if (self.n_tokenized-len(atoms)).bit_length() < self.n_tokenized.bit_length():
            self.unk_count_agg.add(self.unk_count)
            self.n_tokenized_agg.add(self.n_tokenized)
            self.unk_count.clear()
            self.n_tokenized = 0

            if is_main_worker():
                unk_count = self.unk_count_agg.get()
                n_tokenized = self.n_tokenized_agg.get()
                if len(unk_count) == 0:
                    self.unk_logger.info(f"No unknown atoms in {n_tokenized} atoms.")
                else:
                    self.unk_logger.info(f"Unknown atoms in {n_tokenized} atoms:")
                    for atom, n in sorted(unk_count.items(), key=lambda x: x[1], reverse=True):
                        self.unk_logger.info(f"  {atom}: {n}")
        
        return tokens

    def vocs(self):
        return set(self.atom_vocs+[self.unk_voc])
                
class StringTokenizer(Tokenizer):
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

class StringTokenizer2(Tokenizer):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, voc_dir: str, unk_voc: str='[UNK]'):
        with open(f"{voc_dir}/re.txt") as f:
            re_str = f.readline().strip()
        self.pattern = re.compile(re_str)
        self.unk_voc = unk_voc
        with open(f"{voc_dir}/tokens.txt") as f:
            vocs = f.read().splitlines()
        self.tokenizer = StringTokenizer(vocs, unk_voc)

    def tokenize(self, string: str):
        tokens = self.pattern.findall(string)
        if len(''.join(tokens)) != len(string):
            self.logger.info(f"Unknown token; fall back to StringTokenizer: {string}")
            return self.tokenizer.tokenize(string)
        return tokens

    def vocs(self):
        return self.tokenizer.vocs()
    
class SmilesTokenizer(StringTokenizer):
    def __init__(self):
        super().__init__(open(f"{WORKDIR}/cplm/src/data/smiles_tokens.txt").read().splitlines())

class FloatTokenizer(Tokenizer):
    

    def __init__(self, name, vmin: float, vmax: float, decimal: int=3):
        self.name = name
        self.decimal = decimal
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.n_tokenized_total = 0
        self.n_tokenized = 0
        self.n_over = 0
        self.n_under = 0
        self.n_agg = WorkerAggregator((0, 0, 0), 
                lambda x, y: tuple(x0+y0 for x0, y0 in zip(x, y)))

        self.float_format = "{:.0"+str(self.decimal)+"f}"
        self.unk_logger = getLogger(f"unk.src.data.tokenizer.FloatTokenizer.{self.name}")

    def tokenize(self, x: float):
        x = float(x)
        if x < self.vmin:
            self.n_over += 1
            x = self.vmin
        if x > self.vmax:
            self.n_under += 1
            x = self.vmax
        self.n_tokenized += 1
        self.n_tokenized_total += 1
        if should_show(self.n_tokenized_total, math.inf):
            self.n_agg.add((self.n_tokenized, self.n_over, self.n_under))
            self.n_tokenized = self.n_over = self.n_under = 0

            n = self.n_agg.get()
            if n is not None:
                n_tokenized, n_over, n_under = n
                self.unk_logger.info(f"{n_over}/{n_tokenized} are over vmax, {n_under}/{n_tokenized} are under vmin")
        x = self.float_format.format(x)
        xi = x[:-4]
        xf = x[-4:]

        return [xi, xf]

    def tokenize_array(self, x: Iterable[float]):
        return list(itertools.chain.from_iterable(self.tokenize(x) for x in x))
    
    def int_vocs(self):
        ivmin = int(self.float_format.format(self.vmin).split('.')[0])
        ivmax = int(self.float_format.format(self.vmax).split('.')[0])
        vocs = {str(i) for i in range(ivmin, ivmax+1)}
        if self.vmin < 0: vocs.add('-0')
        return vocs

    def frac_vocs(self):
        return {'.'+str(i).zfill(self.decimal) for i in range(10**self.decimal)}

    def vocs(self):
        return self.int_vocs() | self.frac_vocs()
    
class BinaryClassTokenizer(Tokenizer):
    def __init__(self):
        pass
    def tokenize(self, x: int):
        return ['True'] if bool(x) else ['False']
    def vocs(self):
        return {'True', 'False'}

class TokenizeDataset(WrapDataset[list[str]]):
    def __init__(self, dataset: Dataset, tokenizer: Tokenizer):
        super().__init__(dataset)
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx: int):
        return self.tokenizer.tokenize(self.dataset[idx])
    
    def __len__(self):
        return len(self.dataset)

    def vocs(self):
        return self.tokenizer.vocs()

class ArrayTokenizeDataset(TokenizeDataset):
    def __init__(self, dataset: Dataset[np.ndarray], tokenizer: FloatTokenizer):
        super().__init__(dataset, tokenizer)

    def __getitem__(self, idx: int):
        return self.tokenizer.tokenize_array(self.dataset[idx].ravel())

class SentenceDataset(TupleDataset[tuple[list[str], list[int]]]):
    def __init__(self, *sentence: list[str|Dataset[tuple[list[str], list[int]]]]):
        super().__init__(2)
        self.sentence = sentence

        # check length
        self.len = None
        for word in self.sentence:
            if isinstance(word, Dataset):
                if self.len is None:
                    self.len = len(word)
                else:
                    assert self.len == len(word)
        if self.len is None:
            raise ValueError(f"No dataset in sentence.")

    def __getitem__(self, idx: int) -> list[str]:
        words = []
        positions = []
        pos_offset = 0
        for word in self.sentence:
            if isinstance(word, str):
                words.append(word)
                positions.append(pos_offset)
                pos_offset += 1
            else:
                words0, positions0 = word[idx]
                words += words0
                positions += [pos+pos_offset for pos in positions0]
                pos_offset += len(words)
        return words, positions
    
    def __len__(self):
        return self.len
    
    def vocs(self) -> set[str]:
        vocs = set()
        for word in self.sentence:
            if isinstance(word, str):
                vocs.add(word)
            else:
                    vocs |= word.vocs()
        return vocs
    
    def __str__(self):
        return f"SentenceDataset([{', '.join(str(word) for word in self.sentence)}])"

class TokenWeightDataset(WrapDataset[Tensor]):
    def __init__(self, token_dataset: Dataset[list[str]], separates: set[str], separates2weight: dict[tuple[str], float]|list[float], by_n_separate: bool=False):
        super().__init__(token_dataset)
        self.separates = set(separates)
        self.separates2weight = separates2weight
        self.by_n_separate = by_n_separate

    def __getitem__(self, idx):
        tokens = self.dataset[idx]
        weights = []
        separates = 0 if self.by_n_separate else tuple()
        self.separates2weight[0] if self.by_n_separate else self.separates2weight.get(separates, None)
        for token in tokens:
            if token in self.separates:
                separates = separates+1 if self.by_n_separate else separates + (token,)
                cur_weight = self.separates2weight[separates]
            weights.append(cur_weight)
        return torch.tensor(weights, dtype=torch.float)

class RemoveLastDataset(WrapDataset[T_co]):
    def __getitem__(self, idx: int):
        return self.dataset[idx][:-1]
    def __len__(self):
        return len(self.dataset)