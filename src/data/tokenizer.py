from collections.abc import Iterable
from logging import getLogger
import numpy as np

class TokenEncoder:
    def __init__(self, tokens: Iterable[str]):
        self.i2tok = sorted(tokens)
        self.tok2i = {tok: i for i, tok in enumerate(tokens, 1)}
        self.pad_token = 0
    
    def encode(self, tokens: str):
        return [self.tok2i[t] for t in tokens]
    
    def decode(self, tokens: int):
        return [self.i2tok[t] for t in tokens]


class ProteinAtomTokenizer:
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, atom_vocs: list=['CA', 'C', 'H', 'O', 'N', 'S'], unk_voc='[UNK]'):
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
                self.logger.warning(f"Unknown atom {atom} in {atoms}")
                self.tokens.append(self.unk_voc)

    def vocs(self):
        return self.atom_vocs+self.unk_voc
                
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
    def vocs(self):
        return self.vocs_+self.unk_voc

class FloatTokenizer:
    logger = getLogger(f"{__qualname__}.{__module__}")

    def __init__(self, vmin: float, vmax: float, decimal: int=3):
        self.vmin = vmin
        self.vmax = vmax
        self.decimal = decimal


    def tokenize(self, x: float):
        if x < self.vmin: 
            self.logger.warning(f"float value out of range: {x}")
            x = self.vmin
        if x > self.vmax:
            self.logger.warning(f"float value out of range: {x}")
            x = self.vmax
        x = f"{x:.03f}"
        xi, xf = str(x).split('.')
        xf = '.'+xf

        return [xi, xf]

    def tokenize_array(self, x: np.ndarray):
        x = np.clip(x, self.vmin, self.vmax-10**(-self.decimal))
        xi = x.astype(int).map(str)
        xf = ((x%1)*1000).astype(int).map(lambda x: '.'+str(x))
        coord_tokens = np.stack([xi, xf], axis=1)

        return coord_tokens.ravel().tolist()
    

    def vocs(self):
        return [str(i) for i in range(int(self.vmin), int(self.vmax)+1)] + \
            [f".{f:.03f}" for f in range(0, 1, 10**-self.decimal)]
