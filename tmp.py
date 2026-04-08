import logging, time
from src.utils.logger import get_logger
from torch.utils.data import Dataset, StackDataset, DataLoader
from src.train.looper import DatasetTimeLooper, add_time_looper
logger = get_logger(stream=True, level=logging.DEBUG, stream_level=logging.DEBUG)
logger.debug('started')
class WaitDataset(Dataset):
    def __init__(self, t: float, size: int):
        self.t = t
        self.size = size
    def __getitem__(self, idx):
        time.sleep(self.t)
        return idx
    def __len__(self):
        return self.size
    
data = StackDataset(
    WaitDataset(0.1, 100), 
    WaitDataset(0.2, 100), 
    WaitDataset(0.3, 100), 
)

looper = DatasetTimeLooper(logger, 'step', 10)
data = add_time_looper(data, looper)

loader = DataLoader(data, batch_size=4, shuffle=True, num_workers=4)

looper.start_loops()
for batch in loader:
    looper.end_loop()
looper.end_loops()