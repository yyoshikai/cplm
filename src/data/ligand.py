from logging import getLogger
import numpy as np
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import Conformer
from rdkit.Geometry import Point3D
from .data import LMDBDataset, CoordTransform
from .tokenizer import FloatTokenizer, StringTokenizer
from ..utils import logtime

# Pretrain時, 分子の処理用のデータセット
class MoleculeDataset(Dataset):
    logger = getLogger(f'{__module__}.{__qualname__}')
    def __init__(self, dataset: Dataset, coord_transform: CoordTransform, 
            smiles_tokenizer: StringTokenizer, coord_tokenizer: FloatTokenizer
        ):
        self.dataset = dataset
        self.coord_transform = coord_transform
        self.smiles_tokenizer = smiles_tokenizer
        self.coord_tokenizer = coord_tokenizer

    def __getitem__(self, idx):
        data = self.dataset[idx]
        with logtime(self.logger, f"[{idx}]"):
            smi = data['smi']
            coord = data['coordinate']
            coord = self.coord_transform(coord)
            return ['[LIGAND]']+self.smiles_tokenizer.tokenize(smi)+['[XYZ]']+self.coord_tokenizer.tokenize_array(coord.ravel())+['[END]']

    def __len__(self):
        return len(self.dataset)

    def vocs(self):
        return self.smiles_tokenizer.vocs()|self.coord_tokenizer.vocs()|{'[LIGAND]', '[XYZ]', '[END]'}

class UniMolLigandDataset(Dataset):
    logger = getLogger(f'{__module__}.{__qualname__}')
    def __init__(self, lmdb_path, n_conformer,
            atom_h: bool=False, coord_h: bool=True):
        """
        atom_h, coord_h: dataにspecificな処理を行っているので, ここで指定している。
            (水素の順番が違うデータセット等では成り立たないかもしれない)。
        
        Parameters
        ----------
        atom_h: SMILESに水素を追加するかどうか
        coord_h: 座標に水素を追加するかどうか

        atom_h=True, coord_h=True とするとBindGPT
        """
        self.net_dataset = LMDBDataset(lmdb_path, key_is_indexed=True)
        self.n_conformer = n_conformer
        self.atom_h = atom_h
        self.coord_h = coord_h
   
    def __getitem__(self, idx):
        mol_idx, conformer_idx = divmod(idx, self.n_conformer)
        data = self.net_dataset[mol_idx]
        with logtime(self.logger, f"[{idx}]"):
            smi = data['smi']
            coord = data['coordinates'][conformer_idx]

            mol = Chem.MolFromSmiles(smi)
            if self.atom_h:
                mol = Chem.AddHs(mol)
                smi = Chem.MolToSmiles(mol)
                atom_idxs = np.array(mol.GetProp('_smilesAtomOutputOrder', autoConvert=True))
                
                # rdkitのバージョンにより水素の数が違う場合, 重原子の座標から水素の座標を推定する。
                # (ただ, atom_h=False, coord_h=Trueの場合古いバージョンの結果をそのまま使っていることになるので, やや対応しない。)
                # experiments/241202_241201_mol_pocket5_debugの `2. 原子のconformerを追加する方法を調べる。`より。
                if np.max(atom_idxs) >= len(coord):
                    coordf = coord.astype(float)
                    mol = Chem.MolFromSmiles(smi)
                    natom = mol.GetNumAtoms()
                    conf = Conformer(natom)
                    for i in range(natom):
                        conf.SetAtomPosition(i, Point3D(*coordf[i]))
                    mol.AddConformer(conf)
                    mol = Chem.AddHs(mol, addCoords=True)
                    coord = mol.GetConformer().GetPositions()
                
                if self.coord_h:
                    coord = coord[atom_idxs]
                else:
                    atom_idxs = [idx for idx in atom_idxs if mol.GetAtomWithIdx(idx).GetSymbol() != 'H']
                    coord = coord[atom_idxs]
            else:
                if self.coord_h:
                    coord = coord
                else:
                    coord = coord[:mol.GetNumAtoms()]
            # self.logger.debug(f"__getitem__({idx})={{smi:{smi}, coord:{coord.shape}}}")
            output = {'smi': smi, 'coordinate': coord}
            return output
    
    def __len__(self):
        return len(self.net_dataset) * self.n_conformer
