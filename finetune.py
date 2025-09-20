import sys, os
import argparse

import yaml
from torch.utils.data import Subset, StackDataset
WORKDIR = os.environ.get('WORKDIR', __file__.split('/cplm/')[0])
sys.path.append(WORKDIR)
from src.data import CacheDataset
from src.data.tokenizer import StringTokenizer, FloatTokenizer, \
    ProteinAtomTokenizer, TokenizeDataset, ArrayTokenizeDataset, \
    SentenceDataset, VocEncoder, TokenEncodeDataset, TokenWeightDataset, RemoveLastDataset
from src.data.datasets.targetdiff import TargetDiffScafCDDataset, TargetDiffScafCDProteinDataset
from src.data.protein import ProteinProcessDataset
from src.data.molecule import MolProcessDataset
from src.data.coord import CoordTransformDataset
from src.data.tokenizer import TokenEncodeDataset, VocEncoder, \
        ProteinAtomTokenizer, FloatTokenizer, StringTokenizer
from src.data.protein import CoordFollowDataset
from src.train import train, add_train_args, set_default_args

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
args = parser.parse_args()
set_default_args(args)

# get pretrain info
pretrain_dir = f"training/results/{args.pretrain_name}"
targs = yaml.safe_load(open(f"{pretrain_dir}/args.yaml"))
## check consistency of args
args_to_ignore = ['studyname', 'max_opt', ]
args_to_warn = ['num_workers', 'gpu_size_gb', ]
changed_args_to_warn = []
for aname, avalue in vars(args).items():
    if aname in targs and avalue != targs[aname]:
        if aname in args_to_ignore:
            continue
        elif aname in args_to_warn:
            changed_args_to_warn.append(aname)
        raise ValueError(f'args.{aname} is different: {targs[aname]}, {avalue}')
if len(changed_args_to_warn) > 0:
    print(f"WARNING: following args were changed from training:")
    for aname in changed_args_to_warn:
        print(f"    {aname}: {targs[aname]} -> {getattr(args, aname)}")

if args.pretrain_opt is None:
    args.pretrain_opt = targs['max_opt']

# data
smiles_tokenizer = StringTokenizer(open("src/data/smiles_tokens.txt").read().splitlines())
coord_tokenizer = FloatTokenizer('ligand', -args.coord_range, args.coord_range, log_interval=args.tokenizer_log_interval)
protein_coord_tokenizer = FloatTokenizer('pocket', -args.coord_range, args.coord_range, log_interval=args.tokenizer_log_interval)
if not args.no_score:
    score_tokenizer = FloatTokenizer('score', -args.coord_range, args.coord_range, log_interval=args.tokenizer_log_interval)
protein_atom_tokenizer = ProteinAtomTokenizer(log_interval=args.tokenizer_log_interval)

split2datas = {}
split2datas_to_log = {}
for split in ['valid', 'train']:
    if args.protein:
        cddata = TargetDiffScafCDProteinDataset(split)
    else:
        cddata = TargetDiffScafCDDataset(split)
    protein, lig, score = cddata.untuple()

    split2datas_to_log[split] = [cddata]

    lig_smi, lig_coord = MolProcessDataset(lig, args.seed, h_atom=not args.no_lig_h_atom, h_coord=args.no_lig_h_coord, randomize=args.lig_randomize).untuple()
    pocket_atom, pocket_coord, pocket_coord_position = ProteinProcessDataset(protein, heavy_atom=not args.no_pocket_heavy_atom, heavy_coord=not args.no_pocket_heavy_coord, h_atom=args.pocket_h_atom, h_coord=args.pocket_h_coord).untuple()

    lig_coord, pocket_coord, _center, _rotation_matrix \
        = CoordTransformDataset(lig_coord, pocket_coord, base_seed=args.seed, normalize_coord=True, random_rotate=True).untuple()

    ## sentence
    separates = {'[POCKET]', '[XYZ]', '[SCORE]', '[LIGAND]', '[END]'}
    pocket_atom = TokenizeDataset(pocket_atom, protein_atom_tokenizer)
    pocket_coord = ArrayTokenizeDataset(pocket_coord, protein_coord_tokenizer)
    if args.coord_follow_atom:
        assert args.pocket_atom_weight == args.pocket_coord_weight
        sentence = ['[POCKET]', CoordFollowDataset(pocket_atom, pocket_coord, pocket_coord_position), '[END]']
        weights = [None, args.pocket_coord_weight, 0.0]
    else:
        sentence = ['[POCKET]', pocket_atom, '[XYZ]', pocket_coord, '[END]']
        weights = [None, args.pocket_atom_weight, args.pocket_coord_weight, 0.0]
    if not args.no_score:
        score = TokenizeDataset(score, score_tokenizer)
        sentence += ['[SCORE]', score, '[END]']
        weights += [0.0, 0.0]
    lig_smi = TokenizeDataset(lig_smi, StringTokenizer(open("src/data/smiles_tokens.txt").read().splitlines()))
    lig_coord = ArrayTokenizeDataset(lig_coord, coord_tokenizer)
    sentence += ['[LIGAND]', lig_smi, '[XYZ]', lig_coord, '[END]']
    weights += [args.lig_smiles_weight, args.lig_coord_weight, 0.0]
    train_data = SentenceDataset(*sentence)
    vocs = train_data.vocs()
    train_data = CacheDataset(train_data)

    ## token
    voc_encoder = VocEncoder(vocs)
    token_data = TokenEncodeDataset(train_data, voc_encoder)
  
    ## weight
    weight_data = RemoveLastDataset(TokenWeightDataset(train_data, separates, weights, by_n_separate=True))

    split2datas[split] = [StackDataset(token_data, weight_data)]

train('finetune', args, split2datas['train'], split2datas['valid'], split2datas_to_log['train'], split2datas_to_log['valid'], voc_encoder, init_state_path=f"{pretrain_dir}/models/{args.pretrain_opt}.pth")
