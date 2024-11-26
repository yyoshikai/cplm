import sys, os, logging
import argparse
from src.data.finetune import FinetuneDataset
from src.utils.logger import get_logger, add_stream_handler
add_stream_handler(get_logger(), level=logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("--output")
parser.add_argument("--radius", type=int)
parser.add_argument("--test", action='store_true')
args = parser.parse_args()

output = args.output
if output is None:
    output = f"r{args.radius}"
if args.test:
    output = "test_"+output

FinetuneDataset.preprocess("/workspace/cheminfodata/crossdocked/CrossDocked2020", f'./preprocess/results/finetune/{output}', args.radius, args.test)


