import sys, os
import concurrent.futures as cf
from time import time
from argparse import ArgumentParser
from tqdm import tqdm
from src.data.datasets.targetdiff import TargetDiffScafCDDataset
from src.evaluate import eval_qvina3
from src.utils.logger import get_logger, add_file_handler

parser = ArgumentParser()
parser.add_argument('--cpu', type=int)
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--studyname', required=True)

args = parser.parse_args()
logger = get_logger(stream=True)
rdir = f"test_qvina/{args.studyname}"
os.makedirs(rdir, exist_ok=True)
add_file_handler(logger, f"{rdir}/debug.log")
logger.info(vars(args))
data = TargetDiffScafCDDataset('train')
_, _, _, ppath, lpath = data.untuple()
paths = [(ppath[i], lpath[i]) for i in range(args.n)]
ppaths, lpaths = zip(*paths)

start = time()
with cf.ProcessPoolExecutor(args.num_workers) as e:
    futures = []
    for i in range(args.n):
        futures.append(e.submit(eval_qvina3, lig_path=lpaths[i], rec_path=ppaths[i], out_dir=f"{rdir}/eval_qvina/{i}", timeout=60, cpu=args.cpu))
    logger.info([f.result() for f in tqdm(futures)])
end = time()
logger.info(f"t={end-start}")

