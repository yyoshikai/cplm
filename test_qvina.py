import sys, os
import concurrent.futures as cf
from time import time
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import DataLoader, StackDataset
from src.data.datasets.targetdiff import TargetDiffScafCDDataset
from src.evaluate import eval_qvina3
from src.utils.logger import get_logger, add_file_handler

parser = ArgumentParser()
parser.add_argument('--cpu', type=int)
parser.add_argument('--num-workers', type=int, default=32)
parser.add_argument('--num-score-workers', type=int, required=True)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--studyname', required=True)
parser.add_argument('--opt', type=int, default=4)

args = parser.parse_args()
logger = get_logger(stream=True)
rdir = f"test_qvina/{args.studyname}"
os.makedirs(rdir, exist_ok=True)
add_file_handler(logger, f"{rdir}/debug.log")
logger.info(vars(args))
data = TargetDiffScafCDDataset('train')
_, _, _, ppath, lpath = data.untuple()
data = StackDataset(ppath, lpath)
loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=lambda x: x)

start = time()
for opt, batch in enumerate(loader):
    ppaths, lpaths = zip(*batch)
    with cf.ProcessPoolExecutor(args.num_score_workers) as e:
        futures = []
        for i in range(args.batch_size):
            futures.append(e.submit(eval_qvina3, lig_path=lpaths[i], rec_path=ppaths[i], out_dir=f"{rdir}/eval_qvina/{i}", timeout=60, cpu=args.cpu))
        logger.info([f.result() for f in tqdm(futures)])
    if opt+1 == args.opt: break
end = time()
logger.info(f"t={end-start}")

