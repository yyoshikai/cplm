import sys, os, math, gc, yaml, psutil, logging
from argparse import ArgumentParser, Namespace
from logging import getLogger
from collections.abc import Iterator
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from ..data.tokenizer import VocEncoder
from ..utils.logger import INFO_WORKER, add_stream_handler, add_file_handler
from ..utils.rdkit import set_rdkit_logger
from ..model import Model
from ..utils.path import cleardir
from src.utils import RANDOM_STATE, rectime

MAIN_RANK = 0

## criterion, optimizer
from torch import Tensor

class CELoss(nn.Module):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, voc_encoder: VocEncoder, seed: int):
        super().__init__()
        self.voc_encoder = voc_encoder
        self.step = 0
        self.seed = seed

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        input: Tensor[L-1, B, N]
        target: Tensor[L-1, B]    
        
        """
        # log tokens in initial few steps
        if self.step < 5:
            rstate = np.random.RandomState(self.seed+self.step)
            idxs = np.arange(target.shape[1])
            if len(idxs) > 10: 
                idxs = np.sort(rstate.choice(target.shape[1], size=10, replace=False))
            self.logger.log(INFO_WORKER, f"target of step {self.step}:")
            for idx in idxs:
                self.logger.log(INFO_WORKER, f"  [{idx:3}]={','.join(self.voc_encoder.decode(target[:,idx].cpu().tolist()))}")
        self.step += 1

        return F.cross_entropy(input.reshape(-1, self.voc_encoder.voc_size), target.ravel(), reduction='sum', ignore_index=self.voc_encoder.pad_token)

class WeightedCELoss(nn.Module):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, voc_encoder: VocEncoder, seed: int):
        super().__init__()
        self.step = 0
        self.voc_encoder = voc_encoder
        self.seed = seed

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        input: Tensor[L-1, B, N]
        target: Tensor[L-1, B]
        
        """
        
        # make weight
        L, B = target.shape
        weight = torch.zeros_like(target, dtype=torch.float) # [L, B]
        smi_count = torch.cumsum(target == self.voc_encoder.voc2i['[LIGAND]'], dim=0)
        weight[smi_count >= 1] = 1
        coord_count = torch.cumsum(target == self.voc_encoder.voc2i['[XYZ]'], dim=0)
        weight[coord_count >= 2] = 5
        end_count = torch.cumsum(target == self.voc_encoder.voc2i['[END]'], dim=0)
        weight[end_count >= 1] = 0
        weight = torch.cat([
            torch.zeros(1, B, dtype=torch.float, device=weight.device), weight[:-1]
        ], dim=0)
        

        # log tokens in initial few steps
        if self.step < 5:
            rstate = np.random.RandomState(self.seed+self.step)
            idxs = np.arange(target.shape[1])
            if len(idxs) > 10: 
                idxs = np.sort(rstate.choice(target.shape[1], size=10, replace=False))
            self.logger.log(INFO_WORKER, f"target of step {self.step}:")
            for idx in idxs:
                self.logger.log(INFO_WORKER, f"  [{idx:3}]={','.join(self.voc_encoder.decode(target[:,idx].cpu().tolist()))}")
                self.logger.log(INFO_WORKER, f"  weight[{idx:3}]={weight[:,idx].cpu().tolist()}")
        self.step += 1
        return torch.sum(F.cross_entropy(input.reshape(-1, self.voc_encoder.voc_size), target.ravel(), reduction='none')*weight.ravel())

def make_train_dir(result_dir):
    if dist.get_rank() == MAIN_RANK:
        cleardir(result_dir)
        os.makedirs(f"{result_dir}/models", exist_ok=True)
        os.makedirs(f"{result_dir}/step_data", exist_ok=True)
        os.makedirs(f"{result_dir}/optimizers", exist_ok=True)
    dist.barrier()

def set_sdp_kernel(sdp_kernel: str|None):
    if sdp_kernel is not None:
        assert sdp_kernel in ['FLASH', 'CUDNN', ]
        torch.backends.cuda.enable_flash_sdp(sdp_kernel == 'FLASH')
        torch.backends.cuda.enable_cudnn_sdp(sdp_kernel == 'CUDNN')
        torch.backends.cuda.enable_math_sdp(sdp_kernel == 'MATH')
        torch.backends.cuda.enable_mem_efficient_sdp(sdp_kernel == 'EFFICIENT')

def get_train_logger(result_dir):
    rank = dist.get_rank()
    size = dist.get_world_size()
    fmt = "[{asctime}]"+f"[{rank}/{size}]"+"[{name}][{levelname}]{message}"
    logger = getLogger()
    add_stream_handler(logger, logging.INFO, fmt=fmt)
    add_file_handler(logger, f"{result_dir}/debug.log", logging.DEBUG, fmt=fmt, mode='a')
    add_file_handler(logger, f"{result_dir}/info.log", logging.INFO, fmt=fmt, mode='a')
    add_file_handler(logger, f"{result_dir}/warning.log", logging.WARNING, fmt=fmt, mode='a')
    logger.setLevel(logging.NOTSET if rank == MAIN_RANK else INFO_WORKER)
    set_rdkit_logger()
    return logger

def add_train_args(parser: ArgumentParser):
    # training
    parser.add_argument("--token-per-step", type=int, default=int(1.6e6))
    parser.add_argument("--max-step", type=int, default=1000000)
    parser.add_argument("--max-opt-step", type=int, default=float('inf'))
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad-value", type=float, default=None)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--coord-noise-std", type=float, default=50.0)
    parser.add_argument("--loss-scale")

    ## scheduler
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scheduler", default='warmup', choices=['warmup', 'step'])

    # environments
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
    parser.add_argument("--reset-nan-grad", action='store_true')

def train(args: Namespace, train_loader: Iterator, model: Model, criterion: nn.Module, result_dir: str, pad_token: int, device: torch.device, log_step):
    logger = getLogger('train')

    # Environment
    ## rank
    rank = dist.get_rank()
    is_main = rank == MAIN_RANK

    ## save args
    if is_main:
        with open(f"{result_dir}/config.yaml", 'w') as f:
            yaml.dump(vars(args), f)

    ## fix seed
    RANDOM_STATE.seed(args.seed)
    if args.test:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False

    ## scaled dot product attention kernel
    set_sdp_kernel(args.sdp_kernel)

    ## optimizer
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    optimizer.zero_grad()
    match args.loss_scale:
        case None:
            loss_scale = 1
        case 'token_per_batch':
            loss_scale = 1/args.token_per_step
        case _:
            loss_scale = float(args.loss_scale)

    ## scheduler
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
    scheduler = LambdaLR(optimizer, schedule)

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
        logger.warning(f'{step=} get batch started.')
        with rectime() as data_timer:
            batch = train_loader.__next__()
            batch = batch.to(device)
            n_token = torch.sum(batch != pad_token).item()
            n_accum_token += n_token

            batch_sizes.append(batch.shape[1])
            max_lens.append(batch.shape[0])
        data_times.append(data_timer.time)
        logger.warning(f'{step=} get batch ended.')

        with rectime() as loss_timer:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logger.warning(f'{step=} forward started.')
                pred = model(batch[:-1])
                logger.warning(f'{step=} forward ended.')
                loss = criterion(pred, batch[1:]) * loss_scale
            logger.warning(f'{step=} autocast ended.')

            loss.backward()
            logger.warning(f'{step=} backward ended.')
            accum_loss += loss.item()

            # check nan
            if args.reset_nan_grad:
                logger.warning(f'{step=} checking nan grad...')
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
            logger.warning(f'{step=} check nan grad ended')

        logger.warning('rectime eended.')    
        loss_times.append(loss_timer.time)
        logger.warning(f'{step=} all f&b ended.')

        # sum accum_token
        """
        reduced_accum_token = torch.tensor(n_accum_token, dtype=torch.int, device=device)
        logger.warning(f'{step=} {reduced_accum_token=}')
        logger.warning(f'{step=} reduce accum token started.')
        dist.all_reduce(reduced_accum_token)
        logger.warning(f'{step=} reduce accum token ended.')
        print(reduced_accum_token)
        logger.warning(f'{step=} reduce token printed.')
        """
        reduced_accum_token = args.token_per_step
        
        if reduced_accum_token >= args.token_per_step:
            logger.warning(f'{step=} optimizer step started.')
            with rectime() as optim_timer:
                if args.clip_grad_value is not None:
                    torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_value)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            logger.warning(f'{step=} optimizer step ended.')
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
            logger.warning(f'optimizer process ended.')
        if (step+1) % log_step == 0:
            logger.info(f"{step+1} step finished.")
        logger.warning(f'{step=} ended.')
    logger.info("Training finished!")