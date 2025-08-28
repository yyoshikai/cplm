
import numpy as np
from torch.utils.data import Dataset
 
from ..tokenizer import TokenizeDataset, ArrayTokenizeDataset
from ..lmdb import PickleLMDBDataset
from ..finetune2 import Protein

class UniMolPocketDataset(Dataset[Protein]):
    def __init__(self, lmdb_path, **kwargs):
        self.dataset = PickleLMDBDataset(lmdb_path, **kwargs)
    
    def __getitem__(self, idx) -> Protein:
        data = self.dataset[idx]
        atoms = np.array(data['atoms'])
        coord =  data.pop('coordinates')[0] # * np.array([0, 1, 2])
        return Protein(atoms=atoms, coord=coord)

    def __len__(self):
        return len(self.dataset)

class CoordFollowDataset(Dataset):
    def __init__(self, pocket_atom: TokenizeDataset, pocket_coord: ArrayTokenizeDataset, coord_position: Dataset[np.ndarray]):
        self.pocket_atom = pocket_atom
        self.pocket_coord = pocket_coord
        self.coord_position = coord_position
        assert len(self.pocket_atom) == len(self.pocket_coord)
    
    def __getitem__(self, idx: int):
        pocket_atom = self.pocket_atom[idx]
        pocket_coord = self.pocket_coord[idx]
        coord_position = self.coord_position[idx]
        assert len(coord_position)*6 == len(pocket_coord)
        output = []
        i_coord = 0
        for i_atom in range(len(pocket_atom)):
            output.append(pocket_atom[i_atom])
            if coord_position[i_coord] == i_atom:
                output += pocket_coord[i_coord*6:(i_coord+1)*6]
                i_coord += 1
        return output

    def __len__(self):
        return len(self.pocket_atom)
    
    def vocs(self):
        return self.pocket_atom.vocs()|self.pocket_coord.vocs()