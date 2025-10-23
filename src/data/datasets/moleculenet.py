from typing import Literal
import yaml
import numpy as np
import pandas as pd

from ..lmdb import PickleLMDBDataset
from ...utils.path import WORKDIR
from .unimol import mol_from_unimol_data
MOLNET_DIR = f"{WORKDIR}/cheminfodata/molnet"

with open(f"{MOLNET_DIR}/versions/unimol/tasks.yaml") as f:
    tasks = yaml.safe_load(f)
infos = pd.read_csv(f"{MOLNET_DIR}/versions/unimol/info.csv", index_col=0)

class MoleculeNetDataset(PickleLMDBDataset):
    data_names = infos.index.tolist()

    def __init__(self, data_name: str, split: Literal['train', 'valid', 'test']):
        self.data_name = data_name
        self.split = split
        dname2uname = {
            'qm7': 'qm7dft',
            'qm8': 'qm8dft',
            'qm9': 'qm9dft',
        }
        self.unimol_name = dname2uname.get(data_name, data_name)
        super().__init__(f"{MOLNET_DIR}/versions/unimol/molecular_property_prediction/{self.unimol_name}/{split}.lmdb", idx_to_key='str')

        self._lazy_target = None

    @property
    def tasks(self) -> list[str]:
        return tasks[self.data_name]
    
    @property
    def is_cls(self) -> bool:
        return bool(infos.loc[self.data_name, 'is_cls'])

    @property    
    def main_metric(self) -> Literal['AUROC', 'AUPR', 'RMSE', 'MAE']:
        return infos.loc[self.data_name, 'metric']
    
    def get_y(self, task: str) -> np.ndarray:
        assert task in self.tasks
        if self._lazy_target is None:
            self._lazy_target = np.load(f"{MOLNET_DIR}/versions/unimol/target/{self.data_name}/{self.split}.npy")
        return self._lazy_target[self.tasks.index(task)]

from rdkit import Chem
from ..data import TupleDataset
class UniMolMoleculeNetDataset(TupleDataset[tuple[Chem.Mol, tuple[int|float,...]]]):
    def __init__(self, data_name: str, split: Literal['train', 'valid', 'test']):
        super().__init__(2)
        self.dataset = MoleculeNetDataset(data_name, split)
        self.n_conformer = 10

    def __getitem__(self, idx) -> Chem.Mol:
        mol_idx, conformer_idx = divmod(idx, self.n_conformer)
        data = self.dataset[mol_idx]
        mol = mol_from_unimol_data(data['smi'], data['coordinates'][conformer_idx])
        return mol, data['target']
    
    def __len__(self):
        return len(self.dataset) * self.n_conformer