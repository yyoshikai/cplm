import argparse

import yaml
from src.train import train, add_train_args, update_pretrain_args, set_default_args
from src.finetune import get_finetune_data
from src.data import StackDataset

logs = []

# arguments
parser = argparse.ArgumentParser()
## training
add_train_args(parser)
## pretrain
parser.add_argument("--pretrain-name", required=True)
parser.add_argument("--pretrain-opt", type=int)
## dataset
parser.add_argument('--pocket-atom-weight', type=float, default=0.0)
parser.add_argument('--pocket-coord-weight', type=float, default=0.0)
parser.add_argument('--lig-smiles-weight', type=float, default=1.0)
parser.add_argument('--lig-coord-weight', type=float, default=5.0)
parser.add_argument("--no-score", action='store_true')
parser.add_argument('--protein', action='store_true')
args = parser.parse_args()
pretrain_dir = f"training/results/{args.pretrain_name}"
targs = yaml.safe_load(open(f"{pretrain_dir}/args.yaml"))
update_pretrain_args(args, targs)
set_default_args(args)
if args.pretrain_opt is None:
    args.pretrain_opt = args['max_opt']
if args.seed is None:
    args.seed = targs.seed

# data
split2datas = {}
for split in ['valid', 'train']:
    voc_encoder, raw_data, token_data, weight_data, _center_data, _rotation_data, \
        _protein_filename_data, _ligand_filename_data \
        = get_finetune_data(args, split, True, True, set(), prompt_score='none' if args.no_score else 'data')
    logs.append(f"    {split} data: {len(token_data):,}/{len(raw_data):,}")
    data_names = [type(raw_data).__name__]
    split2datas[split] = [StackDataset(token_data, weight_data)]

train('finetune', args, split2datas['train'], split2datas['valid'], voc_encoder, logs, data_names, f"{pretrain_dir}/models/{args.pretrain_opt}.pth")
