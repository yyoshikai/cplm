import os
from argparse import ArgumentParser, Namespace
from functools import partial

import yaml
import numpy as np
import pandas as pd
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
from src.utils.rdkit import ignore_rdkit_warning
from src.utils.logger import NO_DUP
from src.data import KeyDataset, CacheDataset, StackDataset, TensorDataset
from src.data.coord import CoordTransformDataset, RescaleDataset
from src.data.datasets.moleculenet import UniMolMoleculeNetDataset, MoleculeNetDataset
from src.data.tokenizer import FloatTokenizer, BinaryClassTokenizer, TokenizeDataset, SentenceDataset, VocEncoder, TokenEncodeDataset, RemoveLastDataset, TokenWeightDataset
from src.data.collator import DDPStringCollateLoader
from src.data.molecule import MolProcessDataset, MolTokenizeDataset
from src.train import get_early_stop_opt, set_env, get_process_ranks, get_model, CrossEntropyLoss, get_optimizer_scheduler, log_batch

# Environment
logs = []

# args
parser = ArgumentParser()
## downstream
parser.add_argument('--studyname', required=True)
parser.add_argument('--weight-decay', type=float, default=0.01)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--seed', type=int)
parser.add_argument('--n-trials', type=int, default=100)
### param range
parser.add_argument('--batch-size-max', type=int, default=16384)
parser.add_argument('--batch-size-min', type=int, default=32)
parser.add_argument('--lr-max', type=float, default=)
parser.add_argument('--lr-min', type=float, default=)
parser.add_argument('--n-epoch-max', type=int, default=80)
parser.add_argument('--n-epoch-min', type=int, default=20)
parser.add_argument('--warmup-ratios', type=float, nargs='+', default=[0.0, 0.05, 0.1])
parser.add_argument('--local-batch-size', type=int, default=128)
## pretrain
parser.add_argument('--pretrain-dir', required=True)
parser.add_argument('--pretrain-opt', type=int)
parser.add_argument('--pretrain-patience-val', type=int)
## task
parser.add_argument('--data', required=True)
parser.add_argument('--task')
## Environment
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument("--sdp-kernel", choices=['FLASH', 'EFFICIENT'], default='FLASH')
parser.add_argument("--gpu-size-gb", type=float, required=True)
### verbosity
parser.add_argument
## debug
parser.add_argument("--test")
parser.add_argument('--deterministic', action='store_true')
parser.add_argument('--no-commit', action='store_true')
args = parser.parse_args()
## pretrain args
pretrain_dir = args.pretrain_dir
targs = Namespace(**yaml.safe_load(open(f"{pretrain_dir}/args.yaml")))
raw = MoleculeNetDataset(args.data, 'train')
if args.seed is None: 
    args.seed = targs.seed
if args.task is None:
    tasks = raw.tasks
    if len(tasks) != 1:
        raise ValueError(f"Please select task from {raw.tasks}")
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
def get_downstream_data(split, is_valid, voc_encoder=None):
    unimol_raw = UniMolMoleculeNetDataset(args.data, split, 1, True, args.seed)
    mol, target = unimol_raw.untuple()
    target = KeyDataset(CacheDataset(target), raw.tasks.index(args.task))
    
    # coord transform
    mol = CoordTransformDataset(mol, base_seed=args.seed, normalize_coord=True, random_rotate=False if is_valid else True).untuple()[0]

    # tokenize
    sentence = []
    mol = MolProcessDataset(mol, args.seed, not targs.no_lig_h_atom, targs.lig_randomize)
    mol = MolTokenizeDataset(mol, coord_follow_atom=getattr(targs, 'lig_coord_follow_atom', False), atoms=getattr(targs, 'lig_atoms', False), atom_order=getattr(targs, 'atom_order', False), coord_range=targs.coord_range, no_h_coord=targs.no_lig_h_coord)
    sentence += ['[LIGAND]', mol, '[END]']
    if is_valid:
        sentence += ['[SCORE]']
        sentence = SentenceDataset(*sentence)
        token, order = sentence.untuple()
        token = TokenEncodeDataset(token, voc_encoder)
        order = TensorDataset(order, torch.long)
        data = StackDataset(token, order, target)
        return data
    else:
        if raw.is_cls:
            target_tokenizer = BinaryClassTokenizer()
            scaler = None
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
        token, position = sentence.untuple()
        if voc_encoder is None:
            voc_encoder = VocEncoder(vocs)
        token = TokenEncodeDataset(token, voc_encoder)
        position = TensorDataset(position, torch.long)

        ## weight
        separates = {'[LIGAND]', '[SCORE]', '[END]'}
        weights = [None, 0.0, 0.0, 1.0, 0.0]
        weight = RemoveLastDataset(TokenWeightDataset(token, separates, weights, by_n_separate=True))
        data = StackDataset(token, position, weight)
        return data, voc_encoder, target_tokenizer, scaler


train_data, voc_encoder, target_tokenizer, scaler = get_downstream_data('train', False) 
valid_datas = {split: get_downstream_data(split, True, voc_encoder) for split in ['train', 'valid', 'test']}
vocs = None

if args.batch_size_max > len(train_data):
    logs.append(f"{args.batch_size_max=} was modified to {len(train_data)=}")
    args.batch_size_max = len(train_data)

def train_collate(batch: list[tuple[Tensor, Tensor]]):
    tokens, positions, weights = zip(*batch)
    token_batch = pad_sequence(tokens, padding_value=voc_encoder.pad_token)
    position_batch = pad_sequence(positions, padding_value=-1)
    weight_batch = pad_sequence(weights, padding_value=voc_encoder.pad_token)
    return token_batch, position_batch, weight_batch
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
result_dir = f"downstream/results/{args.studyname}/{args.data}_{args.task}"
logger, token_logger, rank, device = set_env(f"downstream/{args.studyname}/{args.data}_{args.task}", args, logs, [])
MAIN_RANK, SAVE_RANK, DATA_RANK = get_process_ranks()
ignore_rdkit_warning()
ddp_size = dist.get_world_size()
maximize = metrics[raw.main_metric].maximize

# train
def objective(trial: Trial):

    # Trial params
    if rank == MAIN_RANK:
        trargs = Namespace(**{
            'batch_size': trial.suggest_int('batch_size', args.batch_size_min, args.batch_size_max, log=True), 
            'lr': trial.suggest_float('lr', args.lr_min, args.lr_max, log=True), 
            'n_epoch': trial.suggest_int('n_epoch', args.n_epoch_min, args.n_epoch_max, log=True), 
            'warmup_ratio': trial.suggest_categorical('warmup_ratio', args.warmup_ratios)
        })
    else:
        trargs = None
    trargs = dist_broadcast_object(trargs, src=MAIN_RANK)
    logger.info(f"Trial[{trial.number}] trargs={vars(trargs)}", **NO_DUP)

    # Environment
    trial_dir = f"{result_dir}/trials/{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)

    # Model
    model = get_model(targs, voc_encoder, init_state_path, device)
    gpuuse_getter = partial(model.get_gpuuse, bf16=True, kernel=args.sdp_kernel)
    model = DistributedDataParallel(model)
    criterion = CrossEntropyLoss(reduction='none', ignore_index=voc_encoder.pad_token)
    optimizer, scheduler = get_optimizer_scheduler(model, trargs.n_epoch, False, args.weight_decay, False, 'warmup', trargs.lr, trargs.warmup_ratio, False)
    
    # Data
    sampler = DistributedSampler(train_data, drop_last=True)
    loader = DataLoader(train_data, batch_size=trargs.batch_size, 
            sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True, 
            collate_fn=train_collate, prefetch_factor=prefetch_factor)
    logger.debug(f"Step per epoch={len(loader)}")
    valid_loaders = {}
    for split in ['train', 'valid', 'test']:
        item_loader = DataLoader(valid_datas[split], batch_size=None, shuffle=False, num_workers=args.num_workers, pin_memory=True, prefetch_factor=prefetch_factor)
        valid_loaders[split] = DDPStringCollateLoader(item_loader, valid_collate, gpuuse_getter, args.gpu_size, device, 100000, DATA_RANK['valid'])


    if rank == MAIN_RANK:
        epoch_recorder = IterateRecorder(f"{trial_dir}/epochs/{rank}.csv", 1)
    best_main_score = np.inf
    best_main_score_epoch = -1
    step_recorder = IterateRecorder(f"{trial_dir}/steps/{rank}.csv", len(loader))

    # Environment
    ddp_set_random_seed(args.seed)

    # Training
    sampler.set_epoch(0)
    for epoch in range(trargs.n_epoch):
        prefix = f"Trial[{trial.number}]Epoch[{epoch}]"
        ## train epoch
        logger.debug(f"{prefix} train started.")
        # sampler.set_epoch(epoch) TODO: temp!
        os.environ['EPOCH'] = str(epoch)

        model.train()
        for step, (token_batch, position_batch, weight_batch) in enumerate(loader):
            optimizer.zero_grad()
            token_batch = token_batch.to(device)
            weight_batch = weight_batch.to(device)
            max_len, batch_size = token_batch.shape

            ### log batch
            input_batch, target_batch = token_batch[:-1], token_batch[1:]
            if epoch <= 1 and step <= 1:
                log_batch(prefix, logger, token_logger, target_batch, weight_batch, voc_encoder, step, False, gpuuse_getter)

            ### forward & backward
            for start in range(0, trargs.batch_size, args.local_batch_size):
                with torch.autocast('cuda', torch.bfloat16):
                    loss = (criterion(model(input_batch[:,start:start+args.local_batch_size]), 
                        target_batch[:,start:start+args.local_batch_size])
                        *weight_batch[:,start:start+args.local_batch_size]).sum()
                loss.backward()
            optimizer.step()

            ### log
            if should_show(step, len(loader)):
                logger.debug(f"{prefix}Step[{step}] finished.")
            step_recorder.record(epoch=epoch, batch_size=batch_size, max_len=max_len, 
                    loss_per_item=loss.item()/batch_size)
        scheduler.step()

        ## validation epoch
        logger.debug(f"{prefix} validation started.")
        model.eval()
        os.environ['EPOCH'] = '0'
        with torch.inference_mode():
            scores = {}
            for split in ['train', 'valid', 'test']:
                logger.debug(f"{prefix} validating {split} set...")
                worker_preds = []
                worker_targets = []
                n_free_match = 0
                n_gen = 0
                for step, batch in enumerate(valid_loaders[split]):
                    if batch is None: continue
                    token_batch, position_batch, target_batch = batch
                    L, B = token_batch.shape
                    prompt_sizes = [torch.sum(token_batch[:,b]==voc_encoder.pad_token).item()
                            for b in range(B)]
                    
                    input_batch = token_batch
                    generateds = [[] for _ in range(B)]
                    for i_gen, choice_idxs in enumerate(choice_idxss):
                        with torch.autocast('cuda', torch.bfloat16):
                            output_batch = model(input_batch) # [L, B, N]
                        next_input_batch = torch.cat([input_batch, torch.full((1, B), 
                                fill_value=voc_encoder.pad_token, 
                                dtype=input_batch.dtype, device=input_batch.device)])
                    
                        for b in range(B):
                            log_prob = output_batch[prompt_sizes[b]-1+i_gen, b] # [N]
                            gen = choice_idxs[torch.argmax(log_prob[choice_idxs])].item()
                            gen_free = torch.argmax(log_prob).item()
                            
                            # token_logger.info(f"prompt[{b}]={voc_encoder.decode(input_batch[:,b].tolist())}, prompt_size[{b}]={prompt_sizes[b]}")
                            n_free_match += int(gen_free == gen)
                            n_gen += 1
                            
                            next_input_batch[prompt_sizes[b]+i_gen, b] = gen
                            generateds[b].append(voc_encoder.i2voc[gen])
                        input_batch = next_input_batch
                    
                    if raw.is_cls:
                        worker_preds += [int(generated[0] == 'True') for generated in generateds]
                    else:
                        worker_preds += [float(generated[0]+generated[1]) for generated in generateds]
                    worker_targets.append(target_batch)
                logger.info(f"{prefix}validation[{split}] free_match={n_free_match/n_gen:.04f}")
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
            if maximize: 
                main_score = -main_score
            if main_score < best_main_score:
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

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=args.n_trials)
study.trials_dataframe().to_csv(f"{result_dir}/study.csv")

if rank == MAIN_RANK:
    best_trial = study.best_trial.number
    df = pd.read_csv(f"{result_dir}/trials/{best_trial}/epochs/{rank}.csv")
    best_epoch = np.argmin(df[f"valid {raw.main_metric}"].values
            *(-1 if maximize else 1))
    metric_names = ['AUROC', 'AUPR'] if raw.is_cls else ['RMSE', 'MAE', 'R^2']
    df = pd.DataFrame({'value': {mname: df.loc[best_epoch, f'test {mname}']
            for mname in metric_names}})
    df.to_csv(f"{result_dir}/scores.tsv", sep='\t', header=False)

dist.destroy_process_group()