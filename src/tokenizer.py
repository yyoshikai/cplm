import numpy as np

COORD_MIN = -20
COORD_SUP = 20

class MoleculeProteinTokenizer:
    def __init__(self):
        self.pad_token = 0
        self.coord_start_token = 1
        self.mol_start_token = 2
        self.prot_start_token = 3

        self.coord_i_offset = 4
        self.coord_f_offset = self.coord_i_offset + (COORD_SUP - COORD_MIN)
        

        pass
    
    def tokenize_coord(self, coord: np.ndarray) -> list[int]:
        coord = np.clip(coord, COORD_MIN, COORD_SUP-0.001) - COORD_MIN
        coord_i = coord.astype(int) + self.coord_i_offset
        coord_f = ((coord % 1)*1000).astype(int) + self.coord_f_offset
        tokens = np.stack([coord])

    def tokenize_mol(self) -> list[int]:
        pass

    def tokenize_protein(self) -> list[int]:
        pass

    @property
    def voc_size(self):
        raise NotImplementedError
