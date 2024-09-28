import os
import numpy as np

COORD_MIN = -20
COORD_SUP = 20

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
                string += self.net_tok2voc[tok-self.offset]
        return string

    @property
    def voc_size(self):
        return len(self.net_tok2voc)

def get_string_tokenizer(voc_file, offset=0) -> StringTokenizer: 
    with open(voc_file) as f:
        toker = StringTokenizer(f.read().splitlines(), offset)
    return toker

class MoleculeProteinTokenizer:
    def __init__(self):
        self.pad_token = 0
        self.coord_start_token = 1
        self.mol_start_token = 2
        self.prot_start_token = 3

        self.coord_min = -20
        self.coord_max = 20
        self.coord_i_offset = 4
        self.coord_f_offset = self.coord_i_offset + (COORD_SUP-COORD_MIN)

        smi_offset = self.coord_f_offset + 1000
        self.smi_tokenizer = get_string_tokenizer(f"{os.path.basename(__file__)}/smiles_tokens.txt", smi_offset)
        self.residue_offset = smi_offset + self.smi_tokenizer.voc_size
        self.residue_vocs = ['CA', 'C', 'N', 'O', 'S']
        self.residue_voc_size = len(self.residue_vocs)

        pass
    
    def tokenize_coord(self, coord: np.ndarray) -> list[int]:
        coord = coord.ravel()
        coord = np.clip(coord, COORD_MIN, COORD_SUP-0.001)-COORD_MIN
        coord_i = coord.astype(int)+self.coord_i_offset
        coord_f = ((coord%1)*1000).astype(int)+self.coord_f_offset
        coord_tokens = np.stack([coord_i, coord_f], axis=1).ravel()

        return [self.coord_start_token]+coord_tokens.tolist()

    def tokenize_smi(self, smi) -> list[int]:
        return self.smi_tokenizer.tokenize(smi)

    def tokenize_protein(self, residues, coord) -> list[int]:
        tokens = [self.prot_start_token]
        ca_idxs = []
        for ires, residue in enumerate(residues):
            if residue[:2] == 'CA':
                ca_idxs.append(ires)
                tokens.append(self.residue_offset)
                continue

            for i in range(1, self.n_residue_token):
                voc = self.residue_vocs[i]
                if residue[:len(voc)] == voc:
                    tokens.append(self.residue_offset+i)
                    break
        coord = coord[ca_idxs]
        return tokens + self.tokenize_coord(coord)

    @property
    def voc_size(self):
        raise NotImplementedError
