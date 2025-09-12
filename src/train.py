import os, math, yaml, psutil, logging
import itertools as itr
from argparse import ArgumentParser, Namespace
from logging import Logger, getLogger
from typing import Literal
import numpy as np
import pandas as pd
import transformers
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel

from .data import RepeatDataset, RevealIterator
from .data.tokenizer import VocEncoder
from .data.sampler import InfiniteRandomSampler
from .data.collator import DDPStringCollateLoader
from .utils import git_commit, git_get_hash, reveal_data
from .utils.logger import INFO_WORKER, add_stream_handler, add_file_handler
from .utils.rdkit import set_rdkit_logger
from .utils.model import get_num_params, get_model_size
from .model import Model
from .utils.path import cleardir
from src.utils import set_random_seed, TimerTqdm
from torch.optim import Optimizer
MAIN_RANK = 0

def sync_train_dir(result_dir):
    if dist.get_rank() == MAIN_RANK:
        cleardir(result_dir)
        os.makedirs(f"{result_dir}/models", exist_ok=True)
        os.makedirs(f"{result_dir}/steps", exist_ok=True)
        os.makedirs(f"{result_dir}/opts", exist_ok=True)
        os.makedirs(f"{result_dir}/optimizers", exist_ok=True)
        os.makedirs(f"{result_dir}/time")
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
    os.makedirs(f"{result_dir}/logs/main_debug", exist_ok=True)
    os.makedirs(f"{result_dir}/logs/data_examples", exist_ok=True)

    # main logger
    logger = getLogger()
    add_stream_handler(logger, logging.INFO, fmt=fmt)
    add_file_handler(logger, f"{result_dir}/main.log", logging.INFO, fmt=fmt, mode='a')
    add_file_handler(logger, f"{result_dir}/logs/main_debug/{rank}.log", logging.DEBUG, fmt=fmt, mode='a')
    logger.setLevel(logging.NOTSET if rank == MAIN_RANK else INFO_WORKER)

    # data logger
    data_logger = getLogger('dexs')
    add_file_handler(data_logger, f"{result_dir}/logs/data_examples/{rank}.log", fmt=fmt, mode='a')
    data_logger.propagate = False

    # unknown logger
    unk_logger = getLogger('unk')
    add_file_handler(unk_logger, f"{result_dir}/logs/unknowns.log", fmt=fmt, mode='a')
    unk_logger.propagate = False

    # third-party modules
    set_rdkit_logger()
    getLogger('.prody').disabled = True
    transformers.utils.logging.enable_propagation()
    transformers.utils.logging.disable_default_handler()

    return logger, data_logger

def add_train_args(parser: ArgumentParser):
    # training
    parser.add_argument("--weight-per-opt", type=int, default=int(1.6e6))
    parser.add_argument("--max-opt", type=int, required=True) # TODO: decide default value
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad-value", type=float, default=None)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--coord-noise-std", type=float, default=50.0)
    parser.add_argument("--loss-scale")

    ## scheduler
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scheduler", default='warmup', choices=['warmup', 'step'])

    # environments
    parser.add_argument("--eval-opt", type=int, required=True) # temp
    parser.add_argument("--patience-opt", type=int, required=True)
    parser.add_argument("--patience-eval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pin-memory", action='store_true')
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--logtime", action='store_true')
    parser.add_argument("--tokenizer-log-interval", type=int)
    parser.add_argument("--duplicate", default='ask')
    parser.add_argument("--reset-nan-grad", action='store_true')
    ## hardware
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sdp-kernel", choices=['FLASH', 'EFFICIENT'], default='FLASH')
    parser.add_argument("--gpu-size-gb", type=float, required=True)
    
def set_default_args(args: Namespace):
    if args.eval_opt is None:
        args.eval_opt = 1 if args.test else 1000
    if args.tokenizer_log_interval is None:
        args.tokenizer_log_interval = int(1e6) if args.test else int(1e7)
    if args.num_workers > 0 and args.prefetch_factor is None:
        args.prefetch_factor = 10
    args.gpu_size = args.gpu_size_gb * (2**30)
    return args

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

def just_data(datas: list, side: Literal['l', 'r']) -> list[str]:
    strings = [str(d) for d in datas]
    max_len = max(len(s) for s in strings)
    return [s.ljust(max_len) if side == 'l' else s.rjust(max_len) for s in strings]

def log_dataset(logger: Logger, split: str, datasets: list[Dataset]):
    # info
    n_data = len(datasets)
    data_names = [type(d).__name__ for d in datasets]
    lens = [len(d) for d in datasets]
    repeats = [d.n_repeat if isinstance(d, RepeatDataset) else 1 for d in datasets]
    net_lens = [len_ // repeat for len_, repeat in zip(lens, repeats)]

    # just data
    data_names, lens, repeats, net_lens = map(just_data, [data_names, lens, repeats, net_lens], 'lrrr')

    # log
    logger.info(f"{split} data:")
    for i in range(n_data):
        logger.info(f"    {data_names[i]}: {net_lens[i]}*{repeats[i]}={lens[i]}")

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor):
        output = super().forward(input.reshape(target.numel(), -1), target.ravel())
        if self.reduction == 'none':
            output = output.reshape_as(target)
        return output

def log_target_weight(logger: Logger, target: Tensor, weight: Tensor, voc_encoder: VocEncoder, seed: int):

    batch_size = weight.shape[1]
    if batch_size > 10: 
        rstate = np.random.RandomState(seed)
        idxs = np.sort(rstate.choice(batch_size, size=10, replace=False))
    else:
        idxs = np.arange(batch_size)
    for idx in idxs:
        msg = f"    [{idx:3}]="
        tokens = voc_encoder.decode(target[:,idx].cpu().tolist())
        for t, w in zip(tokens, weight[:,idx]):
            msg += f"{t}[{w}],"
        msg += f" remaining weights= {reveal_data(weight[len(tokens):,idx])}"
        logger.info(msg)

def reduce_float(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float, device=device)
    dist.all_reduce(tensor)
    return tensor.item()

def train(args: Namespace, train_data: Dataset[tuple[Tensor, Tensor]], valid_datas: list[Dataset[tuple[Tensor, Tensor]]], valid2train_r: np.ndarray, voc_encoder: VocEncoder, model: Model, result_dir: str, device: torch.device, log_step):
    
    # Environment
    ## logging
    logger = getLogger('train')
    data_logger = getLogger('dexs.train')

    ## rank
    rank = dist.get_rank()
    is_main = rank == MAIN_RANK

    ## save args
    if is_main:
        with open(f"{result_dir}/config.yaml", 'w') as f:
            yaml.dump(vars(args), f)

    ## commit & log hash of git
    if is_main:
        committed = git_commit() if not args.test else False
        logger.debug('git committed.' if committed else 'git not committed.')
        logger.debug(f"git hash={git_get_hash()}")

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
    
    # Model
    model.to(torch.bfloat16)
    model.to(device)
    model = DistributedDataParallel(model)

    # DataLoader
    if is_main:
        sampler = InfiniteRandomSampler(train_data, generator=torch.Generator().manual_seed(args.seed))
        train_loader = DataLoader(train_data, batch_size=None, sampler=sampler, num_workers=args.num_workers, pin_memory=args.pin_memory, persistent_workers=True, prefetch_factor=args.prefetch_factor)
        train_loader = RevealIterator(train_loader, 'train_loader')
        reveal_loader = train_loader
    else:
        train_loader = None
    
    ## Define functions for batch collation
    def collate(data_list: list[tuple[Tensor, Tensor]]):
        batch = pad_sequence([data[0] for data in data_list],
            batch_first=False, padding_value=voc_encoder.pad_token)
        weight_batch = pad_sequence([data[1] for data in data_list],
            batch_first=False, padding_value=0.0)
        return batch, weight_batch
    
    def get_gpuuse(batch_size: int, length: int):
        if isinstance(model.module, Model):
            return model.module.get_gpuuse(batch_size, length, True, args.sdp_kernel)
        else:
            return model.module.get_gpuuse(batch_size, length, True)
    def get_length(item: tuple[Tensor, Tensor]):
        return len(item[0])
    
    ## collated data loader
    train_loader = DDPStringCollateLoader(train_loader, collate, get_gpuuse, get_length, args.gpu_size, device, MAIN_RANK)
    train_iter = train_loader.__iter__()

    # Model
    ## criterion
    criterion = CrossEntropyLoss(reduction='none', ignore_index=voc_encoder.pad_token)

    ## optimizer
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    optimizer.zero_grad()
    match args.loss_scale:
        case None:
            loss_scale = 1
        case _:
            loss_scale = float(args.loss_scale)
    ## scheduler
    scheduler = get_scheduler(optimizer, args.scheduler, args.max_opt)

    # Initial state
    opt_step = 0
    worker_opt_accum_loss = 0.0
    worker_opt_accum_weight = 0.0
    nan_grad_step_saved = False

    ## opt info
    opt_losses = []
    opt_weights = []
    lrs = []
    mems = []

    # step info
    batch_sizes = []
    max_lens = []
    worker_step_losses = []
    worker_step_weights = []

    # Evaluate & early stopping
    valid2train_r = torch.tensor(valid2train_r, device=device)
    valid_opt_steps = []
    valid_mean_losses = []
    early_stop = False

    # save at step 0
    if is_main:
        torch.save(model.module.state_dict(), f"{result_dir}/models/{opt_step}.pth")
        cleardir(f"{result_dir}/optimizers")
        torch.save(optimizer.state_dict(), f"{result_dir}/optimizers/{opt_step}.pth")

    # Show model size
    if is_main:
        logger.debug(f"# of params={get_num_params(model):,}")
        logger.debug(f"Model size={get_model_size(model)/2**30:.02f}GB")


    logger.info("Training started.")
    for step in (step_pbar:=TimerTqdm(itr.count(), 
            time_path=f"{result_dir}/time/steps.csv" if is_main else None,
            log_interval=1, desc='batch')):
        is_starting = step < 5
        # reset gpu to watch gpu use
        if is_starting:
            torch.cuda.reset_peak_memory_stats(device)

        # get batch
        step_pbar.start('get_batch')
        token_batch, weight_batch = train_iter.__next__()

        batch_sizes.append(token_batch.shape[1])
        max_lens.append(token_batch.shape[0])
        
        step_pbar.start('forward')
        with torch.autocast('cuda', torch.bfloat16):
            target = token_batch[1:]
            pred = model(token_batch[:-1])
            loss = (criterion(pred, target)*weight_batch).sum() * loss_scale

        ## Log target for final check
        if is_starting:
            data_logger.debug(f"Train step {step} data:")
            log_target_weight(data_logger, target, weight_batch, voc_encoder, args.seed+step)

        step_pbar.start('backward')
        loss.backward()

        ## add step record
        worker_step_loss = loss.item()
        worker_step_weight = weight_batch.sum().item()
        worker_step_losses.append(worker_step_loss)
        worker_step_weights.append(worker_step_weight)
        worker_opt_accum_loss += worker_step_loss
        worker_opt_accum_weight += worker_step_weight

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
                worker_opt_accum_loss = 0
                optimizer.zero_grad()
        
        # optimizer step
        opt_accum_weight = reduce_float(worker_opt_accum_weight, device)
        if opt_accum_weight >= args.weight_per_opt:

            ## optimizer
            step_pbar.start('optim')
            if args.clip_grad_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_value)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            opt_step += 1

            ## sum opt_accum_loss
            opt_accum_loss = reduce_float(worker_opt_accum_loss, device)

            ## save step data
            opt_losses.append(opt_accum_loss)
            opt_weights.append(opt_accum_weight)
            lrs.append(scheduler.get_last_lr()[0])
            mems.append(psutil.virtual_memory().used/(2**30))

            ## scheduler
            scheduler.step()

            # evaluate
            if opt_step % args.eval_opt == 0:
                step_pbar.start('evaluation')
                
                # validation
                logger.info(f"Validation at {opt_step=}...")
                total_weights = []
                total_losses = []
                for valid_data in valid_datas:
                    logger.info(f"    Validating ")
                    ## Make Loader
                    if is_main:
                        valid_loader = DataLoader[tuple[Tensor, Tensor]](valid_data, batch_size=None, shuffle=False, 
                                num_workers=args.num_workers, pin_memory=args.pin_memory, prefetch_factor=args.prefetch_factor)
                    else:
                        valid_loader = None
                    valid_loader = DDPStringCollateLoader(valid_loader, collate, get_gpuuse, get_length, args.gpu_size, device, MAIN_RANK)

                    ## Accumulate losses
                    total_weight = 0.0
                    total_loss = 0.0
                    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
                        for batch in valid_loader:
                            if batch is None: continue
                            token_batch, weight_batch = batch
                            input, target = token_batch[:-1], token_batch[1:]
                            loss = criterion(model(input), target)
                            total_loss += (loss*weight_batch).sum()
                            total_weight += weight_batch.sum()
                    total_weights.append(total_weight)
                    total_losses.append(total_loss)

                ## reduce all weights & losses
                total_weight = (torch.stack(total_weights)*valid2train_r).sum()
                total_loss = (torch.stack(total_losses)*valid2train_r).sum()
                dist.all_reduce(total_weight, dst=MAIN_RANK)
                dist.all_reduce(total_loss, dst=MAIN_RANK)
                mean_loss = total_loss.item() / total_weight.item()
                logger.info(f"opt={opt_step} {mean_loss=}")

                    
                # save
                dfstep = pd.DataFrame({
                    'batch_size': batch_sizes,
                    'max_len': max_lens, 
                    'step_weight': worker_step_weights,
                    'step_loss': worker_step_loss
                })
                dfstep.to_csv(f"{result_dir}/steps/{rank}.csv")
                dfopt = pd.DataFrame({
                    'loss': opt_losses,
                    'n_token': opt_weights,
                    'lr': lrs,
                    'memory': mems
                })
                dfopt.to_csv(f"{result_dir}/opt/{rank}.csv") # 多分全てのworkerで同じものだが, 念のため確認
                if is_main:
                    dfeval = pd.DataFrame({
                        'opt': valid_opt_steps,
                        'mean_loss': valid_mean_losses
                    })
                    dfeval.to_csv(f"{result_dir}/eval.csv")
                if is_main:
                    torch.save(model.state_dict(), f"{result_dir}/models/{opt_step}.pth")
                    cleardir(f"{result_dir}/optimizers")
                    torch.save(optimizer.state_dict(), f"{result_dir}/optimizers/{opt_step}.pth")

                # Judge early stopping
                valid_opt_steps.append(opt_step)
                valid_mean_losses.append(mean_loss)
                best_opt = valid_opt_steps[np.argmin(valid_mean_losses)]
                if opt_step - best_opt >= args.patiance_opt:
                    logger.info(f"Early stop.")
                    early_stop = True

            worker_opt_accum_loss = worker_opt_accum_weight = 0.0

            if opt_step >= args.max_opt or early_stop:
                break
        
        # Log gpuuse
        if is_starting:
            logger.debug(f"Actual GPU use={torch.cuda.max_memory_allocated(device)/2**30:.03f}")

        # End starting
        if step+1 == 5: 
            step_pbar.log_interval = 10000
            if is_main:
                reveal_loader.enabled = False

        if (step+1) % log_step == 0:
            logger.info(f"{step+1} step finished.")
    logger.info("Training finished!")
