import sys, os, logging
import argparse
sys.path.append(".")
from src.data.finetune import FinetuneDataset
from src.utils.logger import get_logger, add_stream_handler
from src.utils.utils import set_logtime
add_stream_handler(get_logger(), level=logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("--output")
parser.add_argument("--radius", type=int)
parser.add_argument("--test", action='store_true')
parser.add_argument("--ends", nargs='+', default=['lig_tt_docked.sdf', 'lig_tt_min.sdf'])
parser.add_argument("--map-size", type=int, default=int(100e9))
parser.add_argument("--logtime", action='store_true')
parser.add_argument("--cf-size", type=int)
args = parser.parse_args()

set_logtime(args.logtime)
output = args.output
if output is None:
    output = f"r{args.radius}"
if args.test:
    output = "test_"+output

args0 = [vars(args), "/workspace/cheminfodata/crossdocked/CrossDocked2020", 
    f'./preprocess/results/finetune/{output}', args.radius, args.ends, args.map_size, args.test]

if args.cf_size is None:
    FinetuneDataset.preprocess(*args0)
else:
    import concurrent.futures as cf
    with cf.ProcessPoolExecutor(args.cf_size) as e:
        futures = []
        for rank in range(args.cf_size):
            futures.append(e.submit(FinetuneDataset.preprocess, *args0, rank=rank, size=args.cf_size))
        [f.result() for f in futures]
