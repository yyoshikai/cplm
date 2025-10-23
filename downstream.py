import os
from argparse import ArgumentParser, Namespace
from functools import partial
from logging import getLogger

import yaml
import numpy as np
import optuna
import torch
import torch.distributed as dist
from optuna.trial import Trial
from sklearn.metrics import roc_auc_score, average_precision_score, root_mean_squared_error, mean_absolute_error, r2_score

from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel

from src.utils import IterateRecorder, should_show
from src.utils.ddp import dist_broadcast_object, dist_send_tensor, dist_recv_tensor
from src.utils.random import ddp_set_random_seed
from src.utils.rdkit import ignore_warning
from src.data import KeyDataset, CacheDataset, StackDataset
from src.data.molecule import MolProcessDataset
from src.data.coord import CoordTransformDataset, RescaleDataset
from src.data.datasets.moleculenet import UniMolMoleculeNetDataset, MoleculeNetDataset
from src.data.tokenizer import StringTokenizer, FloatTokenizer, BinaryClassTokenizer, TokenizeDataset, ArrayTokenizeDataset, SentenceDataset, VocEncoder, TokenEncodeDataset, RemoveLastDataset, TokenWeightDataset
from src.data.collator import DDPStringCollateLoader
from src.train import get_early_stop_opt, set_env, get_process_ranks, get_model, CrossEntropyLoss, get_optimizer_scheduler, log_batch, NO_DUP


# Environment
logs = []

# args
parser = ArgumentParser()
## downstream
parser.add_argument('--studyname', required=True)
parser.add_argument('--weight-decay', type=float, default=0.01)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--seed', type=int)
parser.add_argument('--n-trials', type=int, default=100)
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
## Environment
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument("--sdp-kernel", choices=['FLASH', 'EFFICIENT'], default='FLASH')
parser.add_argument("--gpu-size-gb", type=float, required=True)
## debug
parser.add_argument("--test")
parser.add_argument('--deterministic', action='store_true')
parser.add_argument('--no-commit', action='store_true')
args = parser.parse_args()
## pretrain args
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

datas = {}
vocs = None
for split in ['train', 'valid', 'test']:
    is_valid = split != 'train'
    unimol_raw = UniMolMoleculeNetDataset(args.data, split)
    mol, target = unimol_raw.untuple()
    smi, coord = MolProcessDataset(mol, args.seed, h_atom=not targs.no_lig_h_atom, h_coord=not targs.no_lig_h_coord, randomize=targs.lig_randomize).untuple()
    coord = CoordTransformDataset(coord, base_seed=args.seed, normalize_coord=True, random_rotate=False if is_valid else True).untuple()[0]
    target = KeyDataset(CacheDataset(target), raw.tasks.index(args.task))

    # tokenize
    sentence = []
    smi_tokenizer = StringTokenizer(open(f"src/data/smiles_tokens.txt").read().splitlines())
    smi = TokenizeDataset(smi, smi_tokenizer)
    sentence += ['[LIGAND]', smi]
    coord_tokenizer = FloatTokenizer('coord', -targs.coord_range, targs.coord_range)
    coord = ArrayTokenizeDataset(coord, coord_tokenizer)
    sentence += ['[XYZ]', coord, '[END]']
    if split == 'train':
        if raw.is_cls:
            target_tokenizer = BinaryClassTokenizer()
        else:
            ys = MoleculeNetDataset(args.data, 'train').get_y(args.task)
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

        ## weight
        separates = {'[LIGAND]', '[SCORE]', '[END]'}
        weights = [None, 0.0, 0.0, 1.0, 0.0]
        weight = RemoveLastDataset(TokenWeightDataset(sentence, separates, weights, by_n_separate=True))
        data = StackDataset(token, weight)
    else:
        sentence += ['[SCORE]']
        sentence = SentenceDataset(*sentence)
        token = TokenEncodeDataset(sentence, voc_encoder)
        data = StackDataset(token, target)
    datas[split] = data
def train_collate(batch: list[tuple[Tensor, Tensor]]):
    tokens, weights = zip(*batch)
    token_batch = pad_sequence(tokens, padding_value=voc_encoder.pad_token)
    weight_batch = pad_sequence(weights, padding_value=voc_encoder.pad_token)
    return token_batch, weight_batch
def valid_collate(batch: list[tuple[Tensor, float|int]]):
    tokens, targets = zip(*batch)
    token_batch = pad_sequence(tokens, padding_value=voc_encoder.pad_token)
    target_batch = torch.tensor(targets)
    return token_batch, target_batch
if raw.is_cls:
    choice_vocss = [['True', 'False']]
else:
    choice_vocss = [target_tokenizer.int_vocs(), target_tokenizer.frac_vocs()]
choice_idxss = [ torch.tensor([voc_encoder.voc2i[voc] for voc in vocs]) for vocs in choice_vocss]

# Metrics
metrics = {
    'AUROC': roc_auc_score, 'AUPR': average_precision_score, 
    'RMSE': root_mean_squared_error, 'MAE': mean_absolute_error, 
    'R^2': r2_score
} # (y_true, y_score)
roc_auc_score.maximize = average_precision_score.maximize = r2_score.maximize = True
root_mean_squared_error.maximize = mean_absolute_error.maximize = False

# Model
init_state_path = f"{pretrain_dir}/models/{args.pretrain_opt}.pth"

# Environment
result_dir = f"downstream/{args.studyname}/{args.data}_{args.task}"
logger, token_logger, rank, device = set_env(f"downstream/{args.studyname}/{args.data}_{args.task}", args, logs, [])
MAIN_RANK, SAVE_RANK, DATA_RANK = get_process_ranks()
ignore_warning()
ddp_size = dist.get_world_size()

# train
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
    logger.info(f"Trial[{trial.number}] trargs={vars(trargs)}", **NO_DUP)

    # Environment
    trial_dir = f"{result_dir}/trials/{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)

    # Data
    sampler = DistributedSampler(datas['train'], drop_last=True)
    loader = DataLoader(datas['train'], batch_size=trargs.batch_size, 
            sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True, 
            collate_fn=train_collate, prefetch_factor=prefetch_factor)
    logger.debug(f"Step per epoch={len(loader)}")

    # Model
    model = get_model(targs, voc_encoder, init_state_path, device)
    gpuuse_getter = partial(model.get_gpuuse, bf16=True, kernel=args.sdp_kernel)
    model = DistributedDataParallel(model)
    criterion = CrossEntropyLoss(reduction='none', ignore_index=voc_encoder.pad_token)
    optimizer, scheduler = get_optimizer_scheduler(model, trargs.n_epoch, False, args.weight_decay, False, 'warmup', trargs.lr, trargs.warmup_ratio, False)

    if rank == MAIN_RANK:
        epoch_recorder = IterateRecorder(f"{trial_dir}/scores.csv", 1)
    best_main_score = -np.inf
    best_main_score_epoch = -1

    # Environment
    ddp_set_random_seed(args.seed)

    # Training
    for epoch in range(trargs.n_epoch):
        prefix = f"Trial[{trial.number}]Epoch[{epoch}]"
        ## train epoch
        logger.debug(f"{prefix} train started.")
        sampler.set_epoch(epoch)
        model.train()
        for step, (token_batch, weight_batch) in enumerate(loader):
            token_batch = token_batch.to(device)
            input_batch, target_batch = token_batch[:-1], token_batch[1:]
            if epoch <= 1 and step <= 1:
                log_batch(prefix, logger, token_logger, target_batch, weight_batch, voc_encoder, step, False, gpuuse_getter)
            weight_batch = weight_batch.to(device)
            optimizer.zero_grad()
            with torch.autocast('cuda', torch.bfloat16):
                loss = (criterion(model(input_batch), target_batch)*weight_batch).sum()
            loss.backward()
            optimizer.step()
            if should_show(step, len(loader)):
                logger.debug(f"{prefix}Step[{step}] finished.")
        scheduler.step()

        ## validation epoch
        logger.debug(f"{prefix} validation started.")
        model.eval()
        with torch.inference_mode():
            scores = {}
            for split in ['valid', 'test']:
                logger.debug(f"Epoch[{epoch}] validating {split} set...")
                valid_loader = DataLoader(datas[split], batch_size=None, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=prefetch_factor)
                valid_loader = DDPStringCollateLoader(valid_loader, valid_collate, gpuuse_getter, args.gpu_size, device, 100000, DATA_RANK['valid'])
                worker_preds = []
                worker_targets = []
                for step, batch in enumerate(valid_loader):
                    if batch is None: continue
                    token_batch, target_batch = batch
                    L, B = token_batch.shape
                    prompt_sizes = [torch.sum(token_batch[:,b]==voc_encoder.pad_token)
                            for b in range(B)]
                    input_batch = token_batch
                    generateds = [[] for _ in range(B)]
                    for i_gen, choice_idxs in enumerate(choice_idxss):
                        output_batch = model(input_batch) # [L, B, N]
                        next_input_batch = torch.cat([input_batch, torch.full((1, B), 
                                fill_value=voc_encoder.pad_token, 
                                dtype=input_batch.dtype, device=input_batch.device)])
                        for b in range(args.batch_size):
                            log_prob = output_batch[prompt_sizes[b]-1+i_gen, b] # [N]
                            gen = choice_idxs[torch.argmax(log_prob[choice_idxs])].item()
                            next_input_batch[prompt_sizes[b]+i_gen, b] = gen
                            generateds[b].append(voc_encoder.i2voc[gen])
                    
                    if raw.is_cls:
                        worker_preds += [int(generated[0] == 'True') for generated in generateds]
                    else:
                        worker_preds += [float(generated[0]+generated[1]) for generated in generateds]
                    worker_targets.append(target_batch)
                worker_preds = torch.tensor(worker_preds, device=device)
                worker_targets = torch.cat(worker_targets)
                if rank == MAIN_RANK:
                    preds = []
                    targets = []
                    for dst in range(ddp_size):
                        if dst == rank:
                            dst_preds = worker_preds
                            dst_targets = worker_targets
                        else:
                            dst_preds = dist_recv_tensor(dst, device)
                            dst_targets = dist_recv_tensor(dst, device)
                        preds.append(dst_preds.cpu().numpy())
                        targets.append(dst_targets.cpu().numpy())
                    preds = np.concatenate(preds)
                    if not raw.is_cls:
                        preds = scaler.rescale(preds)
                    targets = np.concatenate(targets)

                    metric_names = ['AUROC', 'AUPR'] if raw.is_cls else ['RMSE', 'MAE', 'R^2']
                    for m in metric_names:
                        scores[f"{split} {m}"] = metrics[m](targets, preds)
                else:
                    dist_send_tensor(worker_preds, MAIN_RANK)
                    dist_send_tensor(worker_targets, MAIN_RANK)
            if rank == MAIN_RANK:
                logger.info(f"{prefix} {scores=}")
                epoch_recorder.record(**scores)
        
        # Early stopping
        is_early_stop = False
        if rank == MAIN_RANK:
            main_score = scores[f"valid {raw.main_metric}"]
            if not metrics[raw.main_metric].maximize: 
                main_score = -main_score
            if main_score > best_main_score:
                best_main_score = main_score
                best_main_score_epoch = epoch
            else:
                if epoch >= best_main_score_epoch + args.patience:
                    logger.info(f"Early stopping.")
            is_early_stop = True
        is_early_stop = dist_broadcast_object(is_early_stop, MAIN_RANK)
        if is_early_stop:
            break
    if rank == MAIN_RANK:
        return best_main_score
    else:
        return 0.0

study = optuna.create_study(direction='maximize' if metrics[raw.main_metric].maximize else 'minimize')
study.optimize(objective, n_trials=args.n_trials)

dist.destroy_process_group()