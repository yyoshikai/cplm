import sys, os
from typing import Optional, Literal
from logging import getLogger
from torch.utils.data import Dataset
import numpy as np
from rdkit import Chem
from openbabel import openbabel as ob
from ..lmdb import PickleLMDBDataset
from ..protein import Pocket
from ..data import Subset
from ...chem import array_to_conf
WORKDIR = os.environ.get('WORKDIR', __file__.split('/cplm/')[0])
DEFAULT_UNIMOL_DIR = f"{WORKDIR}/cheminfodata/unimol"

logger = getLogger(__name__)

def mol_from_unimol_data(smi: str, coord: np.ndarray):
    coord = coord.astype(float)
    # Generate mol with conformer
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    n_atom  = mol.GetNumAtoms()
    # rdkitのバージョンにより水素の数が違う場合, 重原子の座標から水素の座標を推定する。
    # experiments/241202_241201_mol_pocket5_debugの `2. 原子のconformerを追加する方法を調べる。`より。
    if n_atom != len(coord):
        mol_heavy = Chem.RemoveHs(mol)
        n_heavy = mol_heavy.GetNumAtoms()
        mol_heavy.AddConformer(array_to_conf(coord[:n_heavy]))
        mol = Chem.AddHs(mol_heavy, addCoords=True)
    else:
        mol.AddConformer(array_to_conf(coord))
    coord = mol.GetConformer().GetPositions()
    if np.any(np.isnan(coord)): # 251111 QM7データでこうなる場合があり, 追加 experiments/241202_241201_mol_pocket5_debugの `4. (251111) Chem.AddHsでエラーになるものがあったため, 対策を調べる`より。
        logger.warning(f"coord has nan: {smi=}, {coord=}")
        mol = Chem.RWMol(mol)
        nan_idxs = np.where(np.any(np.isnan(coord), axis=1))[0]
        mol = Chem.RWMol(mol)
        for idx in nan_idxs[::-1]:
            mol.RemoveAtom(int(idx))
    return mol

class UniMolLigandDataset(Dataset[ob.OBMol|Chem.Mol]):
    def __init__(self, split: Literal['train', 'valid'], cls: Literal['rdkit', 'ob'], unimol_dir=DEFAULT_UNIMOL_DIR):
        self.dataset = PickleLMDBDataset(f"{unimol_dir}/ligands/{split}.lmdb", idx_to_key='str')
        self.n_conformer = 10
        self.cls = cls
        if self.cls == 'ob':
            self.obc = ob.OBConversion()
            self.obc.SetInFormat('smi')

    def __getitem__(self, idx) -> Chem.Mol:
        mol_idx, conformer_idx = divmod(idx, self.n_conformer)
        data = self.dataset[mol_idx]

        smi = data['smi']
        coord: np.ndarray = data['coordinates'][conformer_idx]
        if self.cls == 'ob':
            mol = ob.OBMol()
            self.obc.ReadString(mol, smi)
            mol.AddHydrogens()
            if len(coord) == mol.NumAtoms():
                for i, atom in enumerate(ob.OBMolAtomIter(mol)):
                    atom.SetVector(*coord[i].tolist())
            else:
                mol.DeleteHydrogens()
                for i, atom in enumerate(ob.OBMolAtomIter(mol)):
                    atom.SetVector(*coord[i].tolist())
                mol.AddHydrogens()
        else:
            mol = mol_from_unimol_data(smi, coord)
        return mol
    
    def __len__(self):
        return len(self.dataset) * self.n_conformer

# (TODO: not tested)
# Uni-Molが除いてないなら除く必要はないと思う。
class UniMolLigandNoMolNetDataset(Subset[Chem.Mol]):
    def __init__(self, split: Literal['train', 'valid'], sample_save_dir: Optional[str]=None, unimol_dir=DEFAULT_UNIMOL_DIR):
        dataset = UniMolLigandDataset(split, sample_save_dir, unimol_dir)
        indices = np.load(f"{unimol_dir}/ligands_mask/remove_molnet_test/large/{split}_idxs.npy")
        super().__init__(dataset, indices)
    def __str__(self):
        return type(self).__name__

class UniMolPocketDataset(Dataset[Pocket]):
    def __init__(self, split: Literal['train', 'valid'], unimol_dir=DEFAULT_UNIMOL_DIR):
        self.dataset = PickleLMDBDataset(f"{unimol_dir}/pockets/{split}.lmdb", idx_to_key='str')
    
    def __getitem__(self, idx) -> Pocket:
        data = self.dataset[idx]
        atoms = np.array(data['atoms'])
        coord =  data.pop('coordinates')[0] # * np.array([0, 1, 2])
        return Pocket(atoms=atoms, coord=coord)

    def __len__(self):
        return len(self.dataset)
    def __str__(self):
        return type(self).__name__
