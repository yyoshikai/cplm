from torch.utils.data import Subset
from ..lmdb import StringLMDBDataset, IntLMDBDataset
from ...utils.path import WORKDIR

class PubchemAllDataset(StringLMDBDataset):
    def __init__(self):
        super().__init__(f"{WORKDIR}/cheminfodata/pubchem/251210_smiles/smi.lmdb")
