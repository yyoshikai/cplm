import argparse

import yaml
from addict import Dict
from src.train import train, add_train_args, update_pretrain_args, set_default_args
from src.finetune import get_train_data

logs = []

# arguments
parser = argparse.ArgumentParser()
## training
add_train_args(parser)
## pretrain
parser.add_argument("--pretrain-name", required=True)
parser.add_argument("--pretrain-opt", type=int)
## dataset
parser.add_argument('--score-weight', type=float, default=5.0)
parser.add_argument('--reg', action='store_true')
args = parser.parse_args()
pretrain_dir = f"training/results/{args.pretrain_name}"
targs = Dict(yaml.safe_load(open(f"{pretrain_dir}/args.yaml")))
update_pretrain_args(args, targs)
set_default_args(args)
if args.pretrain_opt is None:
    args.pretrain_opt = args['max_opt']
if args.seed is None:
    args.seed = targs.seed

# data
split2datas = {}
for split in ['valid', 'train']:
    datas, voc_encoder, data_names, data_logs = get_train_data(targs, split, 'reg' if args.reg else 'cls', args.score_weight)
    logs += data_logs
    split2datas[split] = datas

train('downstream/finetune', args, split2datas['train'], split2datas['valid'], voc_encoder, logs, data_names, f"{pretrain_dir}/models/{args.pretrain_opt}.pth")
