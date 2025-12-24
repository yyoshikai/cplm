from argparse import ArgumentParser, Namespace
import yaml
from addict import Dict
from src.train import train, add_train_args, update_pretrain_args, set_default_args
from src.finetune import get_train_data
from src.data.datasets.unimol import UniMolLigandDataset, UniMolLigandNoMolNetDataset, UniMolPocketDataset
from src.data.datasets.pdb import PDBUniMolRandomDataset

logs = []

# arguments
parser = ArgumentParser()
## training
add_train_args(parser)
## pretrain
parser.add_argument("--pretrain-name", required=True)
parser.add_argument("--pretrain-opt", type=int)
## dataset
parser.add_argument('--score-weight', type=float, default=5.0)
parser.add_argument('--reg', action='store_true')
# test
data_clss = [UniMolLigandDataset, UniMolLigandNoMolNetDataset, UniMolPocketDataset, PDBUniMolRandomDataset]
for cls in data_clss:
    dname = cls.__name__.removesuffix('Dataset')
    parser.add_argument(f'--{dname}-val-sample', type=float, default=1.0)
args = parser.parse_args()
pretrain_dir = f"training/results/{args.pretrain_name}"
targs = Dict(yaml.safe_load(open(f"{pretrain_dir}/args.yaml")))
for cls in data_clss:
    attr_name = f"{cls.__name__.removesuffix('Dataset')}_val_sample"
    if (attr:=getattr(args, attr_name)) is not None:
        targs[attr_name] = attr
update_pretrain_args(args, targs)
set_default_args(args)
if args.pretrain_opt is None:
    args.pretrain_opt = targs['max_opt']
    logs.append(f"pretrain_opt was set to {args.pretrain_opt}")
if args.seed is None:
    args.seed = targs.seed
targs = Namespace(**targs)

# data
split2datas = {}
for split in ['valid', 'train']:
    datas, voc_encoder, data_names, data_logs = get_train_data(targs, split, 'reg' if args.reg else 'cls', score_weight=args.score_weight)
    logs += data_logs
    split2datas[split] = datas

train('downstream/finetune', args, split2datas['train'], split2datas['valid'], voc_encoder, logs, data_names, f"{pretrain_dir}/models/{args.pretrain_opt}.pth")
