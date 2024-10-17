import logging
from torch.utils.data import Dataset

class PDBFragmentDataset(Dataset):
    logger = logging.getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, dir_path, cache_path=None, is_mmcif=False):
        self.dir_path = dir_path

        self.paths = []
        self.ch
    @property
    def dir_info(self):
        pass
    
    def __getitem__(self, idx):
        raise NotImplementedError
