from logging import getLogger
import numpy as np

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

    def __init__(self, vmin: float, vmax: float):
        self.vmin = vmin
        self.vmax = vmax

    def tokenize(self, x: float):
        if self.vmin 





