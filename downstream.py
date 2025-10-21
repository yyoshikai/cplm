from argparse import ArgumentParser, Namespace

import yaml
from addict import Dict

from src.data import KeyDataset, CacheDataset, StackDataset
from src.data.molecule import MolProcessDataset
from src.data.coord import CoordTransformDataset
from src.data.datasets.moleculenet import UniMolMoleculeNetDataset, MoleculeNetDataset
from src.data.tokenizer import StringTokenizer, FloatTokenizer, TokenizeDataset, ArrayTokenizeDataset, SentenceDataset, VocEncoder, TokenEncodeDataset, RemoveLastDataset, TokenWeightDataset
from src.train import train, get_early_stop_opt, add_train_args, update_pretrain_args

# Environment
logs = []

# args
parser = ArgumentParser()
## train
add_train_args(parser)
## pretrain
parser.add_argument('--pretrain-name', required=True)
parser.add_argument('--pretrain-opt', type=int)
parser.add_argument('--pretrain-patience-val', type=int)
## downstream
parser.add_argument('--data', required=True)
parser.add_argument('--task')
args = parser.parse_args()
pretrain_dir = f"training/results/{args.pretrain_name}"
targs = Dict(yaml.safe_load(open(f"{pretrain_dir}/args.yaml")))
update_pretrain_args(args, targs)
if args.seed is None:
    args.seed = targs.seed
if args.task is None:
    tasks = MoleculeNetDataset(args.data, 'train').tasks
    assert len(tasks) == 1
    args.task = tasks[0]
if args.pretrain_opt is None:
    if args.pretrain_patience_val is not None:
        args.pretrain_opt = get_early_stop_opt(pretrain_dir, args.pretrain_patience_val)
    else:
        args.pretrain_opt = targs.max_opt
    logs.append(f"args.pretrain_opt was set to {args.pretrain_opt}")
# Data
def get_downstream_data(args: Namespace, split: str, data_name: str, task: str, seed: int, ):

    raw = UniMolMoleculeNetDataset(data_name, split)
    mol, target = raw.untuple()
    smi, coord = MolProcessDataset(mol, seed, h_atom=not args.no_lig_h_atom, h_coord=not args.no_lig_h_coord, randomize=args.lig_randomize).untuple()
    coord, _center, _rotation = CoordTransformDataset(coord, base_seed=seed, normalize_coord=True, random_rotate=True).untuple()
    target = KeyDataset(CacheDataset(target), raw.dataset.tasks.index(task))

    # tokenize
    sentence = []
    smi_tokenizer = StringTokenizer(open(f"src/data/smiles_tokens.txt").read().splitlines())
    smi = TokenizeDataset(smi, smi_tokenizer)
    sentence += ['[LIGAND]', smi]
    coord_tokenizer = FloatTokenizer('coord', -args.coord_range, args.coord_range)
    coord = ArrayTokenizeDataset(coord, coord_tokenizer)
    sentence += ['[XYZ]', coord, '[END]']
    target_tokenizer = FloatTokenizer('target', -args.coord_range, args.coord_range) # TODO: get range & to classification
    target = TokenizeDataset(target, target_tokenizer)
    sentence += ['[SCORE]', target, '[END]']
    sentence = SentenceDataset(*sentence)
    vocs = sentence.vocs()
    sentence = CacheDataset(sentence)
    voc_encoder = VocEncoder(vocs)
    token = TokenEncodeDataset(sentence, voc_encoder)

    # weight
    separates = {'[LIGAND]', '[SCORE]', '[END]'}
    weights = [None, 0.0, 0.0, 1.0, 0.0]
    weight = RemoveLastDataset(TokenWeightDataset(sentence, separates, weights, by_n_separate=True))
    return voc_encoder, token, weight

voc_encoder, train_token, train_weight = get_downstream_data(args, 
        'train', args.data, args.task, args.seed)
train_datas = [StackDataset(train_token, train_weight)]
_, valid_token, valid_weight = get_downstream_data(args, 'valid', args.data, args.task, args.seed)
valid_datas = [StackDataset(valid_token, valid_weight)]
train('donwstream', args, train_datas, valid_datas, voc_encoder, logs, ['UniMolMoleculeNetDataset'], f"{pretrain_dir}/models/{args.pretrain_opt}.pth")
