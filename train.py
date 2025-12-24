import argparse
from src.data.datasets.unimol import UniMolLigandDataset, UniMolLigandNoMolNetDataset, UniMolPocketDataset
from src.data.datasets.pdb import PDBUniMolRandomDataset
from src.finetune import get_train_data
from src.train import train, add_pretrain_args, add_train_args, set_default_args

# arguments
parser = argparse.ArgumentParser()
## settings
add_pretrain_args(parser)
add_train_args(parser)
## dataset
for cls in [UniMolLigandDataset, UniMolLigandNoMolNetDataset, UniMolPocketDataset, PDBUniMolRandomDataset]:
    dname = cls.__name__.removesuffix('Dataset')
    parser.add_argument(f'--{dname}', type=int, default=0)
    parser.add_argument(f'--{dname}-val-sample', type=float, default=1.0)
parser.add_argument('--init-state')
args = parser.parse_args()
set_default_args(args)
if args.seed is None:
    args.seed = 0

# datasets
vocs = set()
split2datas = {}
for split in ['valid', 'train']:
    split2datas[split], voc_encoder, dnames, logs = get_train_data(args, split, 'none')


train('training', args, split2datas['train'], split2datas['valid'], voc_encoder, logs, dnames, args.init_state)
