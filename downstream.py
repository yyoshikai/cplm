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
parser.add_argument('--weight-decay', type=float, default=0.01)
### param range
parser.add_argument('--batch-sizes', type=int, nargs='+', default=[32, 64, 128, 256])
parser.add_argument('--lrs', type=float, nargs='+', default=[5e-5, 8e-5, 1e-4, 4e-4, 5e-4])
parser.add_argument('--n-epochs', type=int, nargs='+', default=[40, 60, 80, 100])
parser.add_argument('--warmup-ratios', type=float, nargs='+', default=[0.0, 0.06, 0.1])
## pretrain
parser.add_argument('--pretrain-name', required=True)
parser.add_argument('--pretrain-opt', type=int)
parser.add_argument('--pretrain-patience-val', type=int)
## task
parser.add_argument('--data', required=True)
parser.add_argument('--task')
### Environment
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument("--sdp-kernel", choices=['FLASH', 'EFFICIENT'], default='FLASH')
parser.add_argument("--gpu-size-gb", type=float, required=True)
args = parser.parse_args()
## pretrain
pretrain_dir = f"training/results/{args.pretrain_name}"
targs = Namespace(**yaml.safe_load(open(f"{pretrain_dir}/args.yaml")))
raw = MoleculeNetDataset(args.data, 'train')
if args.seed is None: 
    args.seed = targs.seed
if args.task is None:
    tasks = raw.tasks
    assert len(tasks) == 1
    args.task = tasks[0]
if args.pretrain_opt is None:
    if args.pretrain_patience_val is not None:
        args.pretrain_opt = get_early_stop_opt(pretrain_dir, args.pretrain_patience_val)
    else:
        args.pretrain_opt = targs.max_opt
    logs.append(f"args.pretrain_opt was set to {args.pretrain_opt}")
args.gpu_size = args.gpu_size_gb * (2**30)
prefetch_factor = 1 if args.num_workers == 0 else 10

# Data
def get_downstream_data(targs: Namespace, split: str, data_name: str, task: str, seed: int, ):
    is_valid = split != 'train'
    raw = UniMolMoleculeNetDataset(data_name, split)
    mol, target = raw.untuple()
    smi, coord = MolProcessDataset(mol, seed, h_atom=not targs.no_lig_h_atom, h_coord=not targs.no_lig_h_coord, randomize=targs.lig_randomize).untuple()
    coord, _center, _rotation = CoordTransformDataset(coord, base_seed=seed, normalize_coord=True, random_rotate=False if is_valid else True).untuple()
    target = KeyDataset(CacheDataset(target), raw.dataset.tasks.index(task))

    # tokenize
    sentence = []
    smi_tokenizer = StringTokenizer(open(f"src/data/smiles_tokens.txt").read().splitlines())
    smi = TokenizeDataset(smi, smi_tokenizer)
    sentence += ['[LIGAND]', smi]
    coord_tokenizer = FloatTokenizer('coord', -targs.coord_range, targs.coord_range)
    coord = ArrayTokenizeDataset(coord, coord_tokenizer)
    sentence += ['[XYZ]', coord, '[END]']
    if is_valid:
        sentence += ['[SCORE]']
    else:
        if raw.dataset.is_cls:
            target_tokenizer = BinaryClassTokenizer()
            scaler = None
        else:
            ys = MoleculeNetDataset(data_name, 'train').get_y(task)
            ymin, ymax = np.min(ys), np.max(ys)
            target = RescaleDataset(target, ymin, ymax, -targs.coord_range*0.8, targs.coord_range*0.8)
            scaler = target.scaler
            logs.append(f"scaler={str(scaler)}")
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
    return voc_encoder, token, weight, scaler

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
import optuna
from copy import copy
from optuna.trial import Trial
from src.train import *
from torch.utils.data.distributed import DistributedSampler
from src.utils.ddp import dist_broadcast_object
from functools import partial
def objective(trial: Trial):

    # Trial params
    if rank == MAIN_RANK:
        trargs = Namespace(**{
            'batch_size': trial.suggest_categorical('batch_size', args.batch_sizes), 
            'lr': trial.suggest_categorical('lr', args.lrs), 
            'n_epoch': trial.suggest_categorical('n_epoch', args.n_epochs), 
            'warmup_ratio': trial.suggest_categorical('warmup_ratio', args.warmup_ratios)
        })
    else:
        trargs = None
    trargs = dist_broadcast_object(trargs, src=MAIN_RANK)

    # Environment
    trial_dir = f"{result_dir}/trials/{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)
    tlogger = getLogger(f'trial.{trial.number}')
    add_file_handler(tlogger, f"{trial_dir}/debug.log")

    # Data
    sampler = DistributedSampler(datas['train'], drop_last=True)
    def collate(batch: list[tuple[Tensor, Tensor]]):
        tokens, weights = zip(*batch)
        token_batch = pad_sequence(tokens, padding_value=voc_encoder.pad_token)
        weight_batch = pad_sequence(weights, padding_value=voc_encoder.pad_token)
        return token_batch, weight_batch
    loader = DataLoader(datas['train'], batch_size=trargs.batch_size, 
            sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True, 
            collate_fn=collate, prefetch_factor=prefetch_factor)

    # Model
    model = get_model(args, voc_encoder, init_state_path, device)
    gpuuse_getter = partial(model.get_gpuuse, bf16=True, kernel=args.sdp_kernel)
    model = DistributedDataParallel(model)
    criterion = CrossEntropyLoss(reduction='none', ignore_index=voc_encoder.pad_token)
    optimizer, scheduler = get_optimizer_scheduler(model, trargs.n_epoch, False, args.weight_decay, False, trargs.lr, trargs.warmup_ratio, False)

    # Environment
    ddp_set_random_seed(args.seed)

    # Training
    for epoch in range(trargs.n_epoch):
        ## train epoch
        sampler.set_epoch(epoch)
        model.train()
        for token_batch, weight_batch in loader:
            optimizer.zero_grad()
            with torch.autocast('cuda', torch.bfloat16):
                loss = (criterion(model(token_batch[:-1]), token_batch[1:])*weight_batch).sum()
            loss.backward()
            optimizer.step()
        scheduler.step()

        ## validation epoch
        model.eval()
        for split in ['valid', 'test']:
            valid_loader = DataLoader(datas[split], batch_size=None, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=prefetch_factor)
            valid_loader = DDPStringCollateLoader(valid_loader, collate, gpuuse_getter, args.gpu_size, device, 100000, DATA_RANK['valid'])
            for 



dist.destroy_process_group()