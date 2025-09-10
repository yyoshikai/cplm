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
from .data.tokenizer import VocEncoder
from .utils import git_commit, git_get_hash
from .utils.logger import INFO_WORKER, add_stream_handler, add_file_handler
from .utils.rdkit import set_rdkit_logger
from .model import Model
from .utils.path import cleardir
from src.utils import rectime, set_random_seed
from torch.optim import Optimizer
MAIN_RANK = 0

## criterion, optimizer
from torch import Tensor
a = torch.tensor([0])
a.item()

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
    token_logger = getLogger(f"{__module__}.{__qualname__}.tokens")
    token_logger_is_set = False

    def __init__(self, voc_encoder: VocEncoder, seed: int, 
                pocket_atom_weight: float, pocket_coord_weight: float, 
                lig_smiles_weight: float, lig_coord_weight: float):
        super().__init__()
        self.step = 0
        self.voc_encoder = voc_encoder
        self.seed = seed
        self.pocket_atom_weight = pocket_atom_weight
        self.pocket_coord_weight = pocket_coord_weight
        self.lig_smiles_weight = lig_smiles_weight
        self.lig_coord_weight = lig_coord_weight
        
        # logger
        self.token_logger.propagate = False

    @classmethod
    def set_token_logger(cls, log_path: str):

        cls.token_logger_is_set = True

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        input: Tensor[L-1, B, N]
        target: Tensor[L-1, B]
        
        """
        
        # make weight
        L, B = target.shape
        weight = torch.full_like(target, fill_value=self.pocket_atom_weight, 
                dtype=torch.float, device=target.device) # [L, B]
        coord_count = torch.cumsum(target == self.voc_encoder.voc2i['[XYZ]'], dim=0)
        lig_count = torch.cumsum(target == self.voc_encoder.voc2i['[LIGAND]'], dim=0)
        end_count = torch.cumsum(target == self.voc_encoder.voc2i['[END]'], dim=0)
        
        weight[coord_count >= 1] = self.pocket_coord_weight
        weight[lig_count >= 1] = self.lig_smiles_weight
        weight[coord_count >= 2] = self.lig_coord_weight
        weight[end_count >= 1] = 0
        weight = torch.cat([
            torch.full((1, B), fill_value=self.pocket_atom_weight, dtype=torch.float, 
                    device=target.device), 
            weight[:-1]
        ], dim=0)
        
        # log tokens in initial few steps

        self.step += 1
        return torch.sum(F.cross_entropy(input.reshape(-1, self.voc_encoder.voc_size), target.ravel(), reduction='none')*weight.ravel())


def sync_train_dir(result_dir):
    if dist.get_rank() == MAIN_RANK:
        cleardir(result_dir)
        os.makedirs(f"{result_dir}/models", exist_ok=True)
        os.makedirs(f"{result_dir}/step_data", exist_ok=True)
        os.makedirs(f"{result_dir}/optimizers", exist_ok=True)
        for dst in range(dist.get_world_size()):
            if dst != MAIN_RANK:
                dist.send_object_list([result_dir], dst=dst)
    else:
        result_dirs = [None]
        dist.recv_object_list(result_dirs, src=MAIN_RANK)
        result_dir = result_dirs[0]
    return result_dir


def set_sdp_kernel(sdp_kernel: str|None):
    if sdp_kernel is not None:
        assert sdp_kernel in ['FLASH', 'CUDNN', 'MATH', 'EFFICIENT', 'ALL'] # 全て無効にするとエラーになる
        torch.backends.cuda.enable_flash_sdp(sdp_kernel in ['FLASH', 'ALL'])
        torch.backends.cuda.enable_cudnn_sdp(sdp_kernel in ['CUDNN', 'ALL'])
        torch.backends.cuda.enable_math_sdp(sdp_kernel in ['MATH', 'ALL'])
        torch.backends.cuda.enable_mem_efficient_sdp(sdp_kernel in ['EFFICIENT', 'ALL'])


def get_train_logger(result_dir):
    rank = dist.get_rank()
    size = dist.get_world_size()
    fmt = "[{asctime}]"+f"[{rank}/{size}]"+"[{name}][{levelname}]{message}"
    os.makedirs(f"{result_dir}/logs", exist_ok=True)

    # main logger
    logger = getLogger()
    add_stream_handler(logger, logging.INFO, fmt=fmt)
    add_file_handler(logger, f"{result_dir}/main.log", logging.INFO, fmt=fmt, mode='a')
    add_file_handler(logger, f"{result_dir}/logs/main_debug.log", logging.DEBUG, fmt=fmt, mode='a')
    logger.setLevel(logging.NOTSET if rank == MAIN_RANK else INFO_WORKER)

    # data logger
    data_logger = getLogger('dexs')
    add_file_handler(data_logger, f"{result_dir}/logs/data_examples.log", logging.INFO, fmt=fmt, mode='a')
    add_file_handler(data_logger, f"{result_dir}/logs/data_examples_debug.log", logging.DEBUG, fmt=fmt, mode='a')
    data_logger.setLevel(logging.NOTSET)
    data_logger.propagate = False

    # third-party modules
    set_rdkit_logger()

    return logger, data_logger


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
    parser.add_argument("--gpu-size-gb", type=int, required=True)
    parser.add_argument("--record-opt-step", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action='store_true')
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--sdp-kernel", choices=['FLASH', 'EFFICIENT'])
    parser.add_argument("--gc", action='store_true')
    parser.add_argument("--logtime", action='store_true')
    parser.add_argument("--tokenizer-log-interval", type=int)
    parser.add_argument("--duplicate", default='ask')
    parser.add_argument("--reset-nan-grad", action='store_true')


def get_scheduler(optimizer: Optimizer, scheduler: str, epoch_step: int, 
        warmup_step: int=2000):
    match scheduler:
        case 'warmup':
            def schedule(step: int):
                if step <= warmup_step:
                    return step / warmup_step
                elif step <= epoch_step:
                    return math.cos(math.pi*((step-warmup_step)/(epoch_step-warmup_step)))*0.49+0.51
                else:
                    return 0.02
        case 'warmup_inverse':
            def schedule(step: int):
                if step <= warmup_step:
                    return step / warmup_step
                else:
                    return (warmup_step / step)**0.5
        case 'step':
            def schedule(step: int):
                return 0.02 ** (step / epoch_step)
        case 'constant':
            def schedule(step: int):
                return 1.0
        case _:
            raise ValueError
    return LambdaLR(optimizer, schedule)


def train(args: Namespace, train_loader: Iterator[tuple[Tensor, Tensor]], voc_encoder: VocEncoder, model: Model, result_dir: str, device: torch.device, log_step, seed: int):
    logger = getLogger('train')
    data_logger = getLogger('dexs.train')

    # Environment
    ## rank
    rank = dist.get_rank()
    is_main = rank == MAIN_RANK

    ## save args
    if is_main:
        with open(f"{result_dir}/config.yaml", 'w') as f:
            yaml.dump(vars(args), f)

    ## commit & log hash of git
    if is_main:
        committed = git_commit()
        logger.debug('git committed.' if committed else 'git not committed.')
        logger.debug(f"git hash={git_get_hash(True)}")

    ## fix seed
    set_random_seed(args.seed)
    if args.test:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False

    ## scaled dot product attention kernel
    set_sdp_kernel(args.sdp_kernel)
    logger.debug(f"{torch.backends.cuda.cudnn_sdp_enabled()=}")
    logger.debug(f"{torch.backends.cuda.flash_sdp_enabled()=}")
    logger.debug(f"{torch.backends.cuda.math_sdp_enabled()=}")
    logger.debug(f"{torch.backends.cuda.mem_efficient_sdp_enabled()=}")
    logger.debug(f"{os.environ.get('TORCH_CUDNN_SDPA_ENABLED')=}")
    
    ## criterion
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=voc_encoder.pad_token)

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
    scheduler = get_scheduler(optimizer, args.scheduler, 55000)

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
        # reset gpu to watch gpu use
        if step < 5:
            torch.cuda.reset_peak_memory_stats(device)

        # get batch
        with rectime() as data_timer:
            batch, weight_batch = train_loader.__next__()
            batch = batch.to(device)
            n_token = torch.sum(batch != voc_encoder.pad_token).item()
            n_accum_token += n_token

            batch_sizes.append(batch.shape[1])
            max_lens.append(batch.shape[0])
        data_times.append(data_timer.time)
        
        with rectime() as loss_timer:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(batch[:-1])
                loss = (criterion(pred.ravel(), batch[1:].ravel())*weight_batch.ravel()).sum() * loss_scale

                # Log output for final check
                if step < 5:
                    batch_size = batch.shape[1]
                    idxs = np.arange(batch_size)
                    if len(idxs) > 10: 
                        rstate = np.random.RandomState(seed+step)
                        idxs = np.sort(rstate.choice(batch_size, size=10, replace=False))
                    data_logger.info(f"target of step {step}:")
                    for idx in idxs:
                        msg = f"  [{idx:3}]="
                        tokens = voc_encoder.decode(batch[1:,idx].cpu().tolist())
                        for t, w in zip(tokens, weight_batch[:,idx]):
                            msg += f"{t}[{w}],"
                        data_logger.info(msg)

            loss.backward()
            l = loss.item()
            accum_loss += l
            
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
                        nan_dir = f"{result_dir}/nan_steps/{step}/{rank}"
                        os.makedirs(nan_dir, exist_ok=True)
                        torch.save(batch.detach().cpu(), f"{nan_dir}/batch.pt")
                        torch.save(model.state_dict(), f"{nan_dir}/model.pth")
                        nan_grad_step_saved = True
                    
                    ## reset grad
                    n_accum_token = 0
                    accum_loss = 0
                    optimizer.zero_grad()
            
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
        
        # Log gpuuse
        logger.debug(f"GPU use={torch.cuda.max_memory_allocated(device)/2**30:.03f}")


        if (step+1) % log_step == 0:
            logger.info(f"{step+1} step finished.")
    logger.info("Training finished!")
