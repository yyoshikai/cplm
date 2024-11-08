import os
import itertools
from collections.abc import Iterable
import numpy as np

"""
modelsのTokenizerから, pad_token, start_token, end_tokenを除き, offsetを追加
"""

import numpy as np

class StringTokenizer():
    def __init__(self, vocs, offset=0):
        """
        Parameters
        ----------
        vocs: array-like of str
        """
        vocs = sorted(list(vocs), key=lambda x:len(x), reverse=True)
        self.voc_lens = np.sort(np.unique([len(voc) for voc in vocs]))[::-1]
        self.min_voc_len = self.voc_lens[-1]
        self.offset = offset
        self.voc2tok = {voc: tok+offset for tok, voc in enumerate(vocs)}
        self.tok2voc = {tok: voc for voc, tok in self.voc2tok.items()}
        self.net_tok2voc = np.array(vocs, dtype=object)

    def tokenize(self, string):
        toks = []
        string_left = string
        while len(string_left) > 0:
            for voc_len in self.voc_lens:
                if string_left[:voc_len] in self.voc2tok:
                    toks.append(self.voc2tok[string_left[:voc_len]])
                    string_left = string_left[voc_len:]
                    break
                if voc_len == self.min_voc_len:
                    raise KeyError(f"Unknown keyward '{string_left}' in {string}")
        return toks

    def detokenize(self, toks, end_token=None):
        """
        Parameters
        ----------
        toks: array_like of int

        Returns
        -------
        string: str
            detokenized string.
        """
        string = ""
        for tok in toks:
            if tok == end_token:
                break
            else:
                if 0 <= tok - self.offset < len(self.net_tok2voc):
                    string += self.net_tok2voc[tok-self.offset]
                else:
                    string += '[UNK]'
        return string

    @property
    def voc_size(self):
        return len(self.net_tok2voc)

def get_string_tokenizer(voc_file, offset=0) -> StringTokenizer: 
    with open(voc_file) as f:
        toker = StringTokenizer(f.read().splitlines(), offset)
    return toker

class MoleculeProteinTokenizer:
    def __init__(self, coord_min=-20, coord_sup=20):
        self.coord_min = coord_min
        self.coord_sup = coord_sup
        self.pad_token = 0
        self.coord_start_token = 1
        self.mol_start_token = 2
        self.prot_start_token = 3

        self.coord_i_offset = 4
        self.coord_f_offset = self.coord_i_offset + (self.coord_sup-self.coord_min)

        smi_offset = self.coord_f_offset + 1000
        self.smi_tokenizer = get_string_tokenizer(f"{os.path.dirname(__file__)}/smiles_tokens.txt", smi_offset)
        self.residue_offset = smi_offset + self.smi_tokenizer.voc_size
        self.residue_vocs = ['CA', 'C', 'N', 'O', 'S']
        self.residue_voc_size = len(self.residue_vocs)
        
        self.added_offset = self.residue_offset + self.residue_voc_size
        self.added_tok2voc = []
        self.added_voc2tok = {}
        pass

    def add_voc(self, voc):
        if voc in self.added_tok2voc: return
        self.added_tok2voc.append(voc)
        self.added_voc2tok[voc] = self.added_offset+len(self.added_tok2voc)-1

    def tokenize_float(self, value: float) -> list[int]:
        value = np.clip(value, self.coord_min, self.coord_sup-0.001)-self.coord_min
        coord_i = int(value)+self.coord_i_offset
        coord_f = int((value%1)*1000)+self.coord_f_offset
        return [coord_i, coord_f]


    def tokenize_coord(self, coord: np.ndarray) -> list[int]:
        coord = coord.ravel()
        coord = np.clip(coord, self.coord_min, self.coord_sup-0.001)-self.coord_min
        coord_i = coord.astype(int)+self.coord_i_offset
        coord_f = ((coord%1)*1000).astype(int)+self.coord_f_offset
        coord_tokens = np.stack([coord_i, coord_f], axis=1).ravel()

        return [self.coord_start_token]+coord_tokens.tolist()
    
    def detokenize_coord(self, tokens: Iterable[int], remove_start=True) -> np.ndarray:
        
        tokens = list(tokens)
        if remove_start:
            tokens = tokens[1:]

        if len(tokens)%6 != 0:
            tokens = tokens+[-1]*(6-len(tokens)%6)
        tokens = np.array(tokens, dtype=float).reshape(-1, 2)
        coord_i = tokens[:,0]-self.coord_i_offset
        coord_f = tokens[:,1]-self.coord_f_offset
        coord_i[(coord_i < 0)|(self.coord_sup-self.coord_min <= coord_i)] = np.nan
        coord_f[(coord_f < 0)|(1000 <= coord_f)] = np.nan
        coord = coord_i+coord_f*0.001-self.coord_min
        coord = coord.reshape(-1, 3)
        return coord


    def tokenize_smi(self, smi) -> list[int]:
        return [self.mol_start_token]+self.smi_tokenizer.tokenize(smi)
    
    def tokenize_mol(self, smi, coord: np.ndarray) -> list[int]:
        return self.tokenize_smi(smi)+self.tokenize_coord(coord)

    def detokenize_mol(self, tokens: Iterable[int]) -> tuple[str, np.ndarray]:
        tokens = iter(tokens)
        if next(tokens) != self.mol_start_token:
            return None, None
        smi_tokens = list(itertools.takewhile(lambda x: x != self.coord_start_token, tokens))
        coord_tokens = list(tokens)

        smi = self.smi_tokenizer.detokenize(smi_tokens)
        coord = self.detokenize_coord(coord_tokens, remove_start=False)
        return smi, coord

    def tokenize_protein(self, atoms, coord) -> list[int]:
        tokens = [self.prot_start_token]
        for ires, residue in enumerate(atoms):
            if residue[:2] == 'CA':
                tokens.append(self.residue_offset)
                continue

            for i in range(1, self.residue_voc_size):
                voc = self.residue_vocs[i]
                if residue[:len(voc)] == voc:
                    tokens.append(self.residue_offset+i)
                    break
        return tokens + self.tokenize_coord(coord)
    
    def detokenize_protein(self, tokens: Iterable[int]) -> tuple[list[str], np.ndarray]:
        tokens = iter(tokens)
        if next(tokens) != self.prot_start_token:
            return None, None
        prot_tokens = list(itertools.takewhile(lambda x: x != self.coord_start_token, tokens))
        coord_tokens = list(tokens)
        
        atoms = []
        for token in prot_tokens:
            if 0 <= token-self.residue_offset < self.residue_voc_size:
                atoms.append(self.residue_vocs[token-self.residue_offset])
            else:
                atoms.append('[UNK]')

        coord = self.detokenize_coord(coord_tokens, remove_start=False)
        
        return atoms, coord

    @property
    def voc_size(self):
        return self.added_offset+len(self.added_tok2voc)
