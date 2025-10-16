import argparse

import yaml
from src.train import train, add_train_args, set_default_args
from src.finetune import get_data
from src.data import StackDataset

# arguments
parser = argparse.ArgumentParser()
add_train_args(parser)
## dataset
parser.add_argument('--pocket-atom-weight', type=float, default=0.0)
parser.add_argument('--pocket-coord-weight', type=float, default=0.0)
parser.add_argument('--lig-smiles-weight', type=float, default=1.0)
parser.add_argument('--lig-coord-weight', type=float, default=5.0)
parser.add_argument("--no-score", action='store_true')
parser.add_argument('--protein', action='store_true')
## pretrain
parser.add_argument("--pretrain-name", required=True)
parser.add_argument("--pretrain-opt", type=int)
parser.add_argument("--ignore-arg-diff", action='store_true')
args = parser.parse_args()
set_default_args(args)

logs = []

# get pretrain info
pretrain_dir = f"training/results/{args.pretrain_name}"
targs = yaml.safe_load(open(f"{pretrain_dir}/args.yaml"))
## check consistency of args
if not args.ignore_arg_diff:
    args_to_ignore = ['studyname', 'max_opt', 'gpu_size', 'no_commit', 'num_workers', 'warmup_ratio', 'eval_opt', 'patience_opt', 'log_opt', 'seed']
    args_to_warn = ['gpu_size_gb']
    changed_args_to_warn = []
    for aname, avalue in vars(args).items():
        if aname in targs and avalue != targs[aname]:
            if aname in args_to_ignore:
                continue
            elif aname in args_to_warn:
                changed_args_to_warn.append(aname)
            else:
                raise ValueError(f'args.{aname} is different: {targs[aname]}, {avalue}')
    if len(changed_args_to_warn) > 0:
        logs.append(f"following args were changed from training:")
        for aname in changed_args_to_warn:
            logs.append(f"    {aname}: {targs[aname]} -> {getattr(args, aname)}")

if args.pretrain_opt is None:
    args.pretrain_opt = targs['max_opt']

# data
split2datas = {}
for split in ['valid', 'train']:
    voc_encoder, raw_data, token_data, weight_data, _center_data, _rotation_data \
        = get_data(args, split, True, True, set())
    logs.append(f"    {split} data: {len(token_data):,}/{len(raw_data):,}")
    data_names = [type(raw_data).__name__]
    split2datas[split] = [StackDataset(token_data, weight_data)]

train('finetune', args, split2datas['train'], split2datas['valid'], voc_encoder, logs, data_names, f"{pretrain_dir}/models/{args.pretrain_opt}.pth")
