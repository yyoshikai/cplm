# 241025 若干argumentを変更した。
import sys, os
import argparse, logging
import math, gc
from glob import glob

import psutil, yaml
from addict import Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.dataset import Subset
from torch.nn.parallel import DistributedDataParallel
import torch.optim.lr_scheduler as lrs
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path.append(WORKDIR)

from src.data import LMDBDataset, FinetuneDataset, DDPStringCollateLoader
from src.data.tokenizer import StringTokenizer, FloatTokenizer, ProteinAtomTokenizer,\
    VocEncoder, TokenEncodeDataset
from src.model import Model
from src.utils import RandomState, set_logtime, rectime
from src.utils.path import timestamp, cleardir
from src.utils.logger import add_file_handler, add_stream_handler, INFO_WORKER
from src.utils.rdkit import set_rdkit_logger

# arguments
parser = argparse.ArgumentParser()

## settings
parser.add_argument("--studyname", default='default')
parser.add_argument("--test", action='store_true')

parser.add_argument("--pretrain-name", required=True)
parser.add_argument("--pretrain-step", type=int)

## hyperparameters
parser.add_argument("--token-per-step", type=int, default=int(1.6e6))
parser.add_argument("--max-step", type=int, default=1000000)
parser.add_argument("--max-opt-step", type=int, default=float('inf'))
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=0.01)
parser.add_argument("--clip-grad-value", type=float, default=None)
parser.add_argument("--clip-grad-norm", type=float, default=1.0)
parser.add_argument("--coord-noise-std", type=float, default=50.0)
parser.add_argument("--coord-range", type=float, help='Defaults to value in training')
parser.add_argument("--loss-scale")
parser.add_argument("--pocket-coord-heavy", action='store_true')
parser.add_argument("--reset-nan-grad", action='store_true')

### scheduler
parser.add_argument("--scheduler", default='warmup', choices=['warmup', 'step'])

## data
parser.add_argument("--finetune-save-dir", required=True)
parser.add_argument("--index-lmdb")

## environments
parser.add_argument("--token-per-batch", type=int, default=25000)
parser.add_argument("--record-opt-step", type=int)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--pin-memory", action='store_true')
parser.add_argument("--prefetch-factor", type=int)
parser.add_argument("--sdp-kernel", choices=['FLASH', 'CUDNN', 'MATH', 'EFFICIENT'])
parser.add_argument("--gc", action='store_true')
parser.add_argument("--logtime", action='store_true')
parser.add_argument("--tokenizer-log-interval", type=int)
parser.add_argument("--duplicate", default='ask')

args = parser.parse_args()

# environment
if args.test: args.studyname+='_test'

pretrain_dir = f"training/results/{args.pretrain_name}"
if args.record_opt_step is None:
    args.record_opt_step = 1 if args.test else 1000
if args.tokenizer_log_interval is None:
    args.tokenizer_log_interval = 10000 if args.test else int(1e6)

auto_pretrain_step = False
if args.pretrain_step is None:
    print(f"{pretrain_dir}/models/*", flush=True)
    steps = [os.path.splitext(os.path.basename(step))[0] for step in glob(f"{pretrain_dir}/models/*")]
    steps = [int(step) for step in steps if step.isdigit()]
    args.pretrain_step = max(steps)
    auto_pretrain_step = True

main_rank = 0
batch_first = False
set_logtime(args.logtime)

## DDP
dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
rank = dist.get_rank()
size = dist.get_world_size()
device = torch.device('cuda', index=rank % torch.cuda.device_count()) \
    if torch.cuda.is_available() else torch.device('cpu')
is_main = rank == main_rank

## make result dir
result_dir = f"finetune/results/{timestamp()}_{args.studyname}"
if is_main:
    cleardir(result_dir)
    os.makedirs(f"{result_dir}/models", exist_ok=True)
    os.makedirs(f"{result_dir}/step_data", exist_ok=True)
    os.makedirs(f"{result_dir}/optimizers", exist_ok=True)
dist.barrier()

## load config
pretrain_config = Dict(yaml.safe_load(open(f"{pretrain_dir}/config.yaml")))
if args.coord_range is None:
    args.coord_range = pretrain_config.coord_range

## save args
if is_main:
    with open(f"{result_dir}/config.yaml", 'w') as f:
        yaml.dump(vars(args), f)

## seed
rstate = RandomState(args.seed)
if args.test:
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

## scaled dot product attention kernel
if args.sdp_kernel is not None:
    torch.backends.cuda.enable_flash_sdp(args.sdp_kernel == 'FLASH')
    torch.backends.cuda.enable_cudnn_sdp(args.sdp_kernel == 'CUDNN')
    torch.backends.cuda.enable_math_sdp(args.sdp_kernel == 'MATH')
    torch.backends.cuda.enable_mem_efficient_sdp(args.sdp_kernel == 'EFFICIENT')

## logger
fmt = "[{asctime}]"+f"[{rank}/{size}]"+"[{name}][{levelname}]{message}"
logger = logging.getLogger()
add_stream_handler(logger, logging.INFO, fmt=fmt)
add_file_handler(logger, f"{result_dir}/debug.log", logging.DEBUG, fmt=fmt, mode='a')
add_file_handler(logger, f"{result_dir}/info.log", logging.INFO, fmt=fmt, mode='a')
logger.setLevel(logging.NOTSET if is_main else INFO_WORKER)
log_step = 1 if args.test else 1000
set_rdkit_logger()

### log environment
logger.info(f"num_workers={args.num_workers}")
if auto_pretrain_step:
    logger.info(f"pretrain_step was set to {args.pretrain_step}")

# data
smiles_tokenizer = StringTokenizer(open("src/data/smiles_tokens.txt").read().splitlines())
coord_tokenizer = FloatTokenizer(-args.coord_range, args.coord_range)
protein_atom_tokenizer = ProteinAtomTokenizer()

train_data = FinetuneDataset(args.finetune_save_dir, 
    protein_atom_tokenizer, smiles_tokenizer, coord_tokenizer, args.seed, mol_atom_h=True, mol_coord_h=True, 
    pocket_coord_heavy=args.pocket_coord_heavy)
vocs = train_data.vocs()
if args.index_lmdb is not None:
    index_data = LMDBDataset(args.index_lmdb, key_is_indexed=True)
    train_data = Subset(train_data, index_data)

voc_encoder = VocEncoder(vocs)
train_data = TokenEncodeDataset(train_data, voc_encoder)
if rank != main_rank:
    del train_data
    train_data = None

train_loader = DDPStringCollateLoader(train_data, args.num_workers, args.pin_memory, args.prefetch_factor, 
    args.token_per_batch, batch_first, voc_encoder.pad_token, device, size, rank, main_rank)

# model
model = Model(8, 768, 12, 4, 0.1, 'gelu', True, voc_encoder.i2voc, voc_encoder.pad_token)
model.to(torch.bfloat16)
model.to(device)
model = DistributedDataParallel(model)

## load state dict
state_dict = torch.load(f"{pretrain_dir}/models/{args.pretrain_step}.pth", 
    map_location=device, weights_only=True)
model.load_state_dict(state_dict)

## criterion, optimizer
criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=voc_encoder.pad_token)
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
optimizer.zero_grad()
match args.loss_scale:
    case None:
        loss_scale = 1
    case 'token_per_batch':
        loss_scale = 1/args.token_per_step
    case _:
        loss_scale = float(args.loss_scale)

match args.scheduler:
    case 'warmup':
        def schedule(step: int):
            if step <= 2000:
                return step / 2000
            elif step <= 55000:
                return math.cos(math.pi*((step-2000)/(55000-2000)))*0.49+0.51
            else:
                return 0.02
    case 'step':
        def schedule(step: int):
            return 0.02 ** (step / 55000)
    case _:
        raise ValueError
scheduler = lrs.LambdaLR(optimizer, schedule)
    

accum_loss = 0
opt_step = 0
n_accum_token = 0
accum_losses = []
accum_n_tokens = []
lrs = []
mems = []

data_times = []
loss_times = []
batch_sizes = []
max_lens = []
nan_grad_step_saved = False

# save at step 0
if is_main:
    torch.save(model.state_dict(), f"{result_dir}/models/{opt_step}.pth")
    cleardir(f"{result_dir}/optimizers")
    torch.save(optimizer.state_dict(), f"{result_dir}/optimizers/{opt_step}.pth")
            

logger.info("Training started.")
for step in range(args.max_step):

    # get batch
    with rectime() as data_timer:
        batch = train_loader.__next__()
        batch = batch.to(device)
        n_token = torch.sum(batch != voc_encoder.pad_token).item()
        n_accum_token += n_token

        batch_sizes.append(batch.shape[1])
        max_lens.append(batch.shape[0])
        # make weight
        weight = torch.zeros_like(batch, dtype=torch.float)
        smi_count = torch.cumsum(batch == voc_encoder.voc2i['[LIGAND]'], dim=0)
        weight[smi_count >= 1] = 1
        coord_count = torch.cumsum(batch == voc_encoder.voc2i['[XYZ]'], dim=0)
        weight[coord_count >= 2] = 5
        end_count = torch.cumsum(batch == voc_encoder.voc2i['[END]'], dim=0)
        weight[end_count >= 1] = 0
        
        # log tokens in initial few steps
        if step < 5:
            rstate = np.random.RandomState(args.seed+step)
            idxs = np.arange(batch.shape[1])
            if len(idxs) > 10: 
                idxs = np.sort(rstate.choice(batch.shape[1], size=10, replace=False))
            logger.log(INFO_WORKER, f"batch of step {step}:")
            for idx in idxs:
                logger.log(INFO_WORKER, f"  [{idx:3}]={','.join(voc_encoder.decode(batch[:,idx].cpu().tolist()))}")
                logger.log(INFO_WORKER, f"  weight[{idx:3}]={weight[:,idx].cpu().tolist()}")

    data_times.append(data_timer.time)

    with rectime() as loss_timer:
        with torch.autocast('cuda', dtype=torch.bfloat16):

            pred = model(batch[:-1])
            loss = torch.sum(criterion(pred.reshape(-1, voc_encoder.voc_size), batch[1:].ravel())*weight[:-1].ravel()) * loss_scale
            loss.backward()

            """ gradientの分布を表示
            grad_max = -torch.inf
            grad_min = torch.inf
            grad_norm = 0.0
            for param in model.parameters():
                grad = param.grad
                grad_max = max(grad_max, torch.max(grad))
                grad_min = min(grad_min, torch.min(grad))
                grad_norm += torch.sum(grad**2)
            logger.info(f"grad_max={grad_max.item()}, grad_min={grad_min}, grad_norm={grad_norm.item()}")
            """
            
            # check nan
            if args.reset_nan_grad:
                grad_is_finite = np.all([torch.all(torch.isfinite(param.grad)).item() for param in model.parameters()])
                if not grad_is_finite:
                    if is_main:
                        logger.warning("nan or infinite value in gradient. Gradient is reset.")
                        for name, param in model.module.named_parameters():
                            n_nan = torch.sum(torch.isnan(param.grad)).item()
                            n_inf = torch.sum(torch.isinf(param.grad)).item()
                            if n_nan > 0 or n_inf > 0:
                                logger.warning(f"{name}: {n_nan=}, {n_inf=}")

                    ## save situation
                    if is_main and not nan_grad_step_saved:
                        nan_dir = f"{result_dir}/nan_step_{step}/{rank}"
                        os.makedirs(nan_dir, exist_ok=True)
                        torch.save(batch.detach().cpu(), f"{nan_dir}/batch.pt")
                        torch.save(model.state_dict(), f"{nan_dir}/model.pth")
                        nan_grad_step_saved = True
                    
                    ## reset grad
                    n_accum_token = 0
                    accum_loss = 0
                    optimizer.zero_grad()

        accum_loss += loss.item()
    loss_times.append(loss_timer.time)

    # sum accum_token
    reduced_accum_token = torch.tensor(n_accum_token, dtype=torch.int, device=device)
    dist.all_reduce(reduced_accum_token)

    if reduced_accum_token >= args.token_per_step:
        with rectime() as optim_timer:
            if args.clip_grad_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_value)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        if args.test:
            logger.info(f"optim_time={optim_timer.time:.03f}")
        opt_step += 1
        accum_losses.append(accum_loss)
        accum_n_tokens.append(n_accum_token)
        lrs.append(scheduler.get_last_lr()[0])

        mems.append(psutil.virtual_memory().used/(2**30))

        scheduler.step()
        if opt_step % args.record_opt_step == 0:
            df = pd.DataFrame({
                'loss': accum_losses,
                'n_token': accum_n_tokens,
                'lr': lrs,
                'memory': mems
            })
            df.to_csv(f"{result_dir}/step_data/{rank}.csv")
            df = pd.DataFrame({
                'data_time': data_times,
                'forward_time': loss_times,
                'batch_size': batch_sizes,
                'max_len': max_lens
            })
            df.to_csv(f"{result_dir}/step_data/{rank}_step.csv")
            if is_main:
                torch.save(model.state_dict(), f"{result_dir}/models/{opt_step}.pth")
                cleardir(f"{result_dir}/optimizers")
                torch.save(optimizer.state_dict(), f"{result_dir}/optimizers/{opt_step}.pth")
            if args.gc:
                gc.collect()
        n_accum_token = 0
        accum_loss = 0

        if opt_step >= args.max_opt_step:
            break
    if (step+1) % log_step == 0:
        logger.info(f"{step+1} step finished.")
logger.info("Training finished!")
dist.destroy_process_group()

