import sys, os
import argparse
sys.path.append(os.environ.get('WORKDIR', "/workspace"))

from src.data.docking import DockingDataset

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help='Input file')
args = parser.parse_args()

input = f"/workspace/cheminfodata/crossdocked/types/{args.input}.types"
output =f"/workspace/cplm/preprocess/results/docking_types/{args.input}"

DockingDataset.process_types(input=input, output=output)
