from argparse import ArgumentParser, Namespace

import yaml
import numpy as np
from addict import Dict

from src.data import KeyDataset, CacheDataset, StackDataset
from src.data.molecule import MolProcessDataset
from src.data.coord import CoordTransformDataset, RescaleDataset
from src.data.datasets.moleculenet import UniMolMoleculeNetDataset, MoleculeNetDataset
from src.data.tokenizer import StringTokenizer, FloatTokenizer, BinaryClassTokenizer, TokenizeDataset, ArrayTokenizeDataset, SentenceDataset, VocEncoder, TokenEncodeDataset, RemoveLastDataset, TokenWeightDataset
from src.train import get_early_stop_opt, add_train_args, update_pretrain_args, set_default_args

# Environment
logs = []

# args
parser = ArgumentParser()
## downstream
parser.add_argument('--studyname', required=True)

## pretrain
parser.add_argument('--pretrain-name', required=True)
parser.add_argument('--pretrain-opt', type=int)
parser.add_argument('--pretrain-patience-val', type=int)
## task
parser.add_argument('--data', required=True)
parser.add_argument('--task')
args = parser.parse_args()
## pretrain
pretrain_dir = f"training/results/{args.pretrain_name}"
targs = Namespace(**yaml.safe_load(open(f"{pretrain_dir}/args.yaml")))
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
def get_downstream_data(targs: Namespace, split: str, data_name: str, task: str, seed: int, ):

    raw = UniMolMoleculeNetDataset(data_name, split)
    mol, target = raw.untuple()
    smi, coord = MolProcessDataset(mol, seed, h_atom=not targs.no_lig_h_atom, h_coord=not targs.no_lig_h_coord, randomize=targs.lig_randomize).untuple()
    coord, _center, _rotation = CoordTransformDataset(coord, base_seed=seed, normalize_coord=True, random_rotate=True).untuple()
    target = KeyDataset(CacheDataset(target), raw.dataset.tasks.index(task))

    # tokenize
    sentence = []
    smi_tokenizer = StringTokenizer(open(f"src/data/smiles_tokens.txt").read().splitlines())
    smi = TokenizeDataset(smi, smi_tokenizer)
    sentence += ['[LIGAND]', smi]
    coord_tokenizer = FloatTokenizer('coord', -targs.coord_range, targs.coord_range)
    coord = ArrayTokenizeDataset(coord, coord_tokenizer)
    sentence += ['[XYZ]', coord, '[END]']
    if raw.dataset.is_cls:
        target_tokenizer = BinaryClassTokenizer()
    else:
        ys = MoleculeNetDataset(data_name, 'train').get_y(task)
        ymin, ymax = np.min(ys), np.max(ys)
        target = RescaleDataset(target, ymin, ymax, -targs.coord_range*0.8, targs.coord_range*0.8)
        logs.append(f"Rescaled ({ymin}, {ymax})->({-targs.coord_range*0.8}, {targs.coord_range*0.8})")
        target_tokenizer = FloatTokenizer('target', -targs.coord_range, targs.coord_range)
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

datas = {}
for split in ['train', 'valid', 'test']:
    voc_encoder, token, weight = get_downstream_data(args, 
            split, args.data, args.task, args.seed)
    datas[split] = StackDataset(token, weight)

# Model
init_state_path = f"{pretrain_dir}/models/{args.pretrain_opt}.pth"

# Environment
result_dir = f"downstream/{args.studyname}/{args.data}_{args.task}"
logger, token_logger, rank, device = set_env(f"downstream/{args.studyname}/{args.data}_{args.task}", args, logs, [])
MAIN_RANK, SAVE_RANK, DATA_RANK = get_process_ranks()

# train
from copy import copy
from optuna.trial import Trial
from src.train import *
def objective(trial: Trial):
    trial_dir = f"{result_dir}/trials/{trial.number}"
    trargs = copy(args)

    # Model
    model = get_model(args, voc_encoder, init_state_path, device)
    model = DistributedDataParallel(model)

    # 
