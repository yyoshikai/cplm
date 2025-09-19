import os, math, yaml, psutil, logging
import itertools as itr
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from contextlib import nullcontext
from logging import Logger, getLogger
from time import time
from typing import Literal
import numpy as np
import pandas as pd
import transformers
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from schedulefree import RAdamScheduleFree

from .data import RepeatDataset, RevealIterator
from .data.tokenizer import VocEncoder
from .data.collator import DDPStringCollateLoader, InfiniteLoader
from .model import Model
from .utils import git_commit, git_get_hash, reveal_data, should_show
from .utils.logger import add_stream_handler, add_file_handler
from .utils.rdkit import set_rdkit_logger
from .utils.model import get_num_params, get_model_size
from .utils.path import cleardir
from .utils.ddp import reduce_float
from .utils import TimerTqdm, IterateRecorder

MAIN_RANK = 1
SAVE_RANK = 0
DATA_RANK = {'train': 3, 'valid': 2}
NO_DUP = {'extra': {'no_dup': True}}

def sync_train_dir(result_dir):
    size = dist.get_world_size()
    main_rank = MAIN_RANK % size
    if dist.get_rank() == main_rank:
        cleardir(result_dir)
        for subdir in ['models', 'opts', 'vals', 'optimizers', 'steps/times', 'data/example_log']:
            os.makedirs(f"{result_dir}/{subdir}", exist_ok=True)
        result_dirs = [result_dir]
    else:
        result_dirs = [None]
    dist.broadcast_object_list(result_dirs, src=main_rank)
    return result_dir


def set_sdp_kernel(sdp_kernel: str|None):
    if sdp_kernel is not None:
        assert sdp_kernel in ['FLASH', 'CUDNN', 'MATH', 'EFFICIENT', 'ALL'] # 全て無効にするとエラーになる
        torch.backends.cuda.enable_flash_sdp(sdp_kernel in ['FLASH', 'ALL'])
        torch.backends.cuda.enable_cudnn_sdp(sdp_kernel in ['CUDNN', 'ALL'])
        torch.backends.cuda.enable_math_sdp(sdp_kernel in ['MATH', 'ALL'])
        torch.backends.cuda.enable_mem_efficient_sdp(sdp_kernel in ['EFFICIENT', 'ALL'])

def get_train_logger(result_dir):

    # ddp info
    rank = dist.get_rank()
    size = dist.get_world_size()
    is_main = rank == MAIN_RANK % size


    fmt = "[{asctime}]"+f"[{rank}/{size}]"+"[{name}][{levelname}]{message}"
    os.makedirs(f"{result_dir}/logs/debug", exist_ok=True)
    os.makedirs(f"{result_dir}/data/example_log", exist_ok=True)

    # main logger
    logger = getLogger()
    token_logger = getLogger('dexs')
    token_logger.propagate = False
    unk_logger = getLogger('unk')
    unk_logger.propagate = False

    # stream & info
    stream_handler = add_stream_handler(logger, logging.INFO, fmt=fmt)
    info_handler = add_file_handler(logger, f"{result_dir}/info.log", logging.INFO, fmt=fmt, mode='a')
    if not is_main:
        for handler in [stream_handler, info_handler]:
            handler.addFilter(lambda record: not getattr(record, 'no_dup', False))
    
    # debug
    add_file_handler(logger, f"{result_dir}/logs/debug/{rank}.log", fmt=fmt, mode='a')

    # tokens
    add_file_handler(logger, f"{result_dir}/data/example_log/{rank}.log", fmt=fmt, mode='a')
    add_file_handler(token_logger, f"{result_dir}/data/example_log/{rank}.log", fmt=fmt, mode='a')

    # unknown tokens
    add_file_handler(unk_logger, f"{result_dir}/logs/unknowns.log", fmt=fmt, mode='a')

    # third-party modules
    set_rdkit_logger()
    getLogger('.prody').setLevel(logging.CRITICAL)
    transformers.utils.logging.enable_propagation()
    transformers.utils.logging.disable_default_handler()

    return logger, logger, token_logger

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
    parser.add_argument("--schedule-free", action='store_true')

    # environments
    parser.add_argument("--eval-opt", type=int, required=True) # temp
    parser.add_argument("--patience-opt", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pin-memory", action='store_true')
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument("--duplicate", default='ask')
    parser.add_argument("--reset-nan-grad", action='store_true')
    ## hardware
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sdp-kernel", choices=['FLASH', 'EFFICIENT'], default='FLASH')
    parser.add_argument("--gpu-size-gb", type=float, required=True)
    ## log interval
    parser.add_argument("--tokenizer-log-interval", type=int)
    parser.add_argument("--log-large-freq", type=int)
    parser.add_argument("--log-step", type=int)
    parser.add_argument("--log-opt", type=int)

    ## test
    parser.add_argument("--sync-every-step", action='store_true')
    parser.add_argument("--model-bfloat16", action='store_true')
    parser.add_argument("--deterministic", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--check", nargs='*', default=[], choices=['early_stop', 
            'data_dist', 'data_epoch', 'data_loading', 'grad', 'random_state', 
            'forward_backward_time', 'optimizer'])

def set_default_args(args: Namespace):
    if args.num_workers > 0 and args.prefetch_factor is None:
        args.prefetch_factor = 10

    # log interval
    if args.tokenizer_log_interval is None:
        args.tokenizer_log_interval = int(1e6) if args.test else int(1e8)
    if args.log_large_freq is None:
        args.log_large_freq = 1 if args.test else 100000
    if args.log_step is None:
        args.log_step = 1 if args.test else 10000
    if args.log_opt is None:
        args.log_opt = 1 if args.test else max(1, args.eval_opt//5)

    # post_init
    args.gpu_size = args.gpu_size_gb * (2**30)
    return args

def get_scheduler(optimizer: Optimizer, scheduler: str, epoch_step: int, 
        warmup_ratio: float=0.04):
    warmup_step = epoch_step*warmup_ratio
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

def just_nums(datas: list[int|float]) -> list[str]:
    if any(not isinstance(d, int) for d in datas):
        datas = [f"{d}   " if isinstance(d, int) else f"{d:.02f}" for d in datas]
    return just_data(datas, 'r')

def log_dataset(logger: Logger, split: str, datasets: list[Dataset]):
    # info
    n_data = len(datasets)
    names = []
    net_sizes = []
    augments = []
    sizes = []
    for data in datasets:
        sizes.append(len(data))
        augment = 1
        if isinstance(data, RepeatDataset):
            augment = data.n_repeat
            data = data.net_dataset
        elif type(data) == Subset:
            augment = len(data) / len(data.dataset) 
            data = data.dataset
        augments.append(augment)
        names.append(type(data).__name__)
        net_sizes.append(len(data))

    # just data
    names = just_data(names, 'l')
    net_sizes, augments, sizes = map(just_nums, (net_sizes, augments, sizes))

    # log
    logger.info(f"{split} data:")
    for i in range(n_data):
        logger.info(f"    {names[i]}: {net_sizes[i]}*{augments[i]}={sizes[i]}")

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor):
        output = super().forward(input.reshape(target.numel(), -1), target.ravel())
        if self.reduction == 'none':
            output = output.reshape_as(target)
        return output

def log_batch(data_name: str, logger: Logger, token_logger: Logger, target_batch: Tensor, weight_batch: Tensor, voc_encoder: VocEncoder, step: int, check_data_dist: bool, gpuuse_getter: Callable[[int, int], float]):

    # weight
    weight_items = weight_batch.sum(dim=0).tolist()
    length, batch_size = target_batch.shape
    gpuuse_gb = gpuuse_getter(batch_size, length) / (2**30)
    logger.debug(f"{data_name}[{step}] batch shape={tuple(target_batch.shape)} GPU use={gpuuse_gb}")
    if check_data_dist:
        logger.debug(f"{data_name}[{step}] weights={weight_items}")

    # tokens
    if batch_size > 10: 
        rstate = np.random.RandomState(step)
        idxs = np.sort(rstate.choice(batch_size, size=10, replace=False))
    else:
        idxs = np.arange(batch_size)
    for idx in idxs:
        msg = f"{data_name}[{step}] batch[{idx:3}]="
        tokens = voc_encoder.decode(target_batch[:,idx].cpu().tolist())
        for t, w in zip(tokens, weight_batch[:,idx]):
            msg += f"{t}[{w}],"
        msg += f" remaining weights= {reveal_data(weight_batch[len(tokens):,idx])}"
        token_logger.debug(msg)

def train(args: Namespace, train_data: Dataset[tuple[Tensor, Tensor]], valid_datas: list[Dataset[tuple[Tensor, Tensor]]], data_names: list[str], valid2train_r: Tensor, voc_encoder: VocEncoder, model, result_dir: str, device: torch.device):
    global DATA_RANK, MAIN_RANK, SAVE_RANK

    # Environment
    ## logging
    logger, token_logger = map(getLogger, [None, 'dexs.train'])

    ## rank
    rank = dist.get_rank()
    size = dist.get_world_size()
    DATA_RANK = {key: r % size for key, r in DATA_RANK.items()}
    MAIN_RANK = MAIN_RANK % size
    SAVE_RANK = SAVE_RANK % size
    is_main = rank == MAIN_RANK
    is_save_rank = rank == SAVE_RANK

    ## save args
    if is_main:
        with open(f"{result_dir}/args.yaml", 'w') as f:
            yaml.dump({**vars(args), 'ddp_size': size}, f)

    ## commit & log hash of git
    if is_main:
        committed = git_commit() if not args.test else False
        logger.debug('git committed.' if committed else 'git not committed.')
        logger.debug(f"git hash={git_get_hash()}")

    ## scaled dot product attention kernel
    set_sdp_kernel(args.sdp_kernel)
    logger.debug(f"{torch.backends.cuda.cudnn_sdp_enabled()=}")
    logger.debug(f"{torch.backends.cuda.flash_sdp_enabled()=}")
    logger.debug(f"{torch.backends.cuda.math_sdp_enabled()=}")
    logger.debug(f"{torch.backends.cuda.mem_efficient_sdp_enabled()=}")
    logger.debug(f"{os.environ.get('TORCH_CUDNN_SDPA_ENABLED')=}")

    logger.debug(f"{args.log_opt=}")
    logger.debug(f"{args.log_step=}")

    ## checks
    check_data_dist = 'data_dist' in args.check
    check_grad = 'grad' in args.check
    check_random_state = 'random_state' in args.check
    check_fb_time = 'forward_backward_time' in args.check
    
    # Model
    model.to(device)
    if args.model_bfloat16:
        model.to(torch.bfloat16)
    model = DistributedDataParallel(model)

    # Make timer here to send to DDPStringCollateLoader
    step_timer = TimerTqdm(itr.count(), time_path=f"{result_dir}/steps/times/{rank}.csv", file_interval=10000, log_interval=10000, desc='step', disable_bar=True)

    # DataLoader
    if rank == DATA_RANK['train']:
        sampler = RandomSampler(train_data, generator=torch.Generator().manual_seed(args.seed))
        train_loader = DataLoader(train_data, batch_size=None, sampler=sampler, num_workers=args.num_workers, pin_memory=args.pin_memory, persistent_workers=False, prefetch_factor=args.prefetch_factor)
        train_loader = InfiniteLoader(train_loader)
        if check_data_dist:
            train_loader = RevealIterator(train_loader, 'train')
            train_reveal_loader = train_loader
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
    train_loader = DDPStringCollateLoader(train_loader, collate, get_gpuuse, get_length, 
            args.gpu_size, device, args.log_large_freq, DATA_RANK['train'], 
            f"{result_dir}/data/train_large_items.csv", step_timer if 'data_loading' in args.check else None)
    train_iter = train_loader.__iter__()

    # criterion
    criterion = CrossEntropyLoss(reduction='none', ignore_index=voc_encoder.pad_token)

    # optimizer & scheduler 
    ## get param group: Partially from TRL
    not_norm_params = get_parameter_names(model,
            forbidden_layer_types=ALL_LAYERNORM_LAYERS, 
            forbidden_layer_names=["bias", "layernorm", "rmsnorm"])
    params = [{   
        "weight_decay": args.weight_decay,
        "params": [param for name, param in model.named_parameters() 
            if name in not_norm_params and param.requires_grad], 
    }, {
        "weight_decay": 0.0,
        "params": [param for name, param in model.named_parameters()
            if name not in not_norm_params and param.requires_grad]
    }]
    if 'optimizer' in args.check:
        logger.debug(f"weight_decay: {[name for name, param in model.named_parameters() if name in not_norm_params and param.requires_grad]}")
        logger.debug(f"non weight_decay: {[name for name, param in model.named_parameters() if name not in not_norm_params and param.requires_grad]}")

    ## optimizer & scheduler
    if args.schedule_free:
        optimizer = RAdamScheduleFree(params, lr=args.lr)
        scheduler = None
        optimizer.train()
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr)
        optimizer.zero_grad()
        scheduler = get_scheduler(optimizer, args.scheduler, args.max_opt)
    ## loss scale
    match args.loss_scale:
        case None:
            loss_scale = 1
        case _:
            loss_scale = float(args.loss_scale)
    # opt recorder
    opt_recorder = IterateRecorder(f"{result_dir}/opts/{rank}.csv", ['loss', 'weight', 'lr', 'memory'], args.eval_opt)

    # Evaluate & early stopping
    val2opt_step = []
    val2mean_loss = []
    val2process_weights = []
    val2process_losses = []
    next_eval_opt = 0

    # step
    ## initial state
    opt = 0
    worker_opt_accum_loss = 0.0
    worker_opt_accum_weight = 0.0
    nan_grad_step_saved = False
    ## recorder
    step_recorder = IterateRecorder(f"{result_dir}/steps/{rank}.csv", 
            ["batch_size", "max_len", "worker_weight" , "worker_loss"], 1000)

    # Show model size
    logger.info(f"# of params={get_num_params(model):,}", **NO_DUP)
    logger.info(f"Model size={get_model_size(model)/2**30:.02f}GB", **NO_DUP)
    if 'optimizer' in args.check:
        logger.debug(f"{optimizer=}")
        logger.debug(f"param_groups={[len(group['params']) for group in optimizer.param_groups]}")

    train_start = time()
    logger.info("Training started.", **NO_DUP)
    for step in step_timer:
        is_starting = step < 5

        # evaluate & save
        if opt == next_eval_opt:
            step_timer.start('evaluation')
            valid_is_starting = len(val2mean_loss) <= 1
            
            # validation 
            logger.info(f"Validation at {opt=}...", **NO_DUP)
            process_weights = []
            process_losses = []
            if args.schedule_free: optimizer.eval()
            for i_data, data_name in enumerate(data_names):
                logger.info(f"    Validating {data_name}", **NO_DUP)
                ## Make Loader
                if rank == DATA_RANK['valid']:
                    valid_data = valid_datas[i_data]
                    os.environ['EPOCH'] = '0'
                    valid_loader = DataLoader[tuple[Tensor, Tensor]](valid_data, batch_size=None, 
                            shuffle=True if check_data_dist else False, num_workers=args.num_workers, 
                            pin_memory=args.pin_memory, prefetch_factor=args.prefetch_factor)
                    if valid_is_starting:
                        if check_data_dist:
                            valid_loader = RevealIterator(valid_loader, f'valid[{data_name}]', logger)
                            valid_reveal_loader = valid_loader
                else:
                    valid_loader = None
                valid_loader = DDPStringCollateLoader(valid_loader, collate, get_gpuuse, get_length, args.gpu_size, device, args.log_large_freq if valid_is_starting else math.inf, DATA_RANK['valid'])

                ## Accumulate losses
                process_weight = torch.tensor(0.0, device=device, dtype=torch.float)
                process_loss = torch.tensor(0.0, device=device, dtype=torch.float)
                with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
                    for val_step, batch in enumerate(valid_loader):
                        if batch is None: continue

                        ### Verbosity
                        val_step_is_starting = val_step < 3
                        if check_data_dist:
                            if val_step == 3 and rank == DATA_RANK['valid']:
                                valid_reveal_loader.enabled = False

                        ### Forward
                        token_batch, weight_batch = batch
                        input, target = token_batch[:-1], token_batch[1:]
                        loss = criterion(model(input), target)
                        
                        ### Log result
                        if valid_is_starting and val_step_is_starting:
                            log_batch(f"valid[{opt}][{data_name}]", logger, token_logger, token_batch, 
                                    weight_batch, voc_encoder, val_step, check_data_dist, get_gpuuse)

                        ### Add
                        process_loss += (loss*weight_batch).sum()
                        process_weight += weight_batch.sum()

                process_weight = process_weight.item()
                process_loss = process_loss.item()
                process_weights.append(process_weight)
                process_losses.append(process_loss)
                logger.debug(f"        loss={process_loss:3.03f} weight={process_weight:3.03f}")
            if args.schedule_free: optimizer.train()

            ## reduce all weights & losses
            total_weights = torch.tensor(process_weights, device=device)
            total_losses = torch.tensor(process_losses, device=device)
            dist.all_reduce(total_weights)
            dist.all_reduce(total_losses)
            mean_losses = total_losses / total_weights
            estimated_train_weights = total_weights * valid2train_r
            mean_loss = ((mean_losses*estimated_train_weights).sum() / estimated_train_weights.sum()).item()
            if 'early_stop' in args.check:
                mean_loss = [0.7, 0.1, 0.5, 0.8, 0.2, 0.6][len(val2mean_loss) % 6]
            logger.info(f"    {mean_loss=}", **NO_DUP)

            ## add & record results
            val2process_losses.append(process_losses)
            val2process_weights.append(process_weights)
            val2opt_step.append(opt)
            val2mean_loss.append(mean_loss)
            dfval = pd.DataFrame({
                ('opt', ''): val2opt_step,
                ('mean_loss', ''): val2mean_loss
            })
            dfval[[('process_weight', data_name) for data_name in data_names]] \
                = np.array(val2process_weights)
            dfval[[('process_loss', data_name) for data_name in data_names]] \
                = np.array(val2process_losses)
            dfval.to_csv(f"{result_dir}/vals/{rank}.csv", index_label='val')
            
            if is_main:
                torch.save(model.state_dict(), f"{result_dir}/models/{opt}.pth")
                cleardir(f"{result_dir}/optimizers")
                torch.save(optimizer.state_dict(), f"{result_dir}/optimizers/{opt}.pth")

            # Judge early stopping
            best_opt = val2opt_step[np.argmin(val2mean_loss)]
            if opt - best_opt >= args.patience_opt:
                logger.info(f"Early stop.", **NO_DUP)
                break
            next_eval_opt += args.eval_opt
        
        # End of training
        if opt >= args.max_opt:
            break

        # step
        ## reset gpu to watch gpu use
        if is_starting:
            torch.cuda.reset_peak_memory_stats(device)
        ## get batch
        step_timer.start('get_batch')
        token_batch, weight_batch = train_iter.__next__()
        target = token_batch[1:]
        max_len, batch_size = token_batch.shape
        worker_step_weight = weight_batch.sum().item()

        ## Log target for final check
        step_timer.start('log_target')
        if is_starting:
            token_logger.debug(f"Train step {step} data:")
            log_batch('train', logger, token_logger, target, weight_batch, 
                    voc_encoder, step, check_data_dist, get_gpuuse)

        ## If do_opt, forward will have to sync gradients
        worker_opt_accum_weight += worker_step_weight
        opt_accum_weight = reduce_float(worker_opt_accum_weight, device)
        do_opt = opt_accum_weight >= args.weight_per_opt
        
        ## forward
        if check_random_state: 
            logger.debug(f"step[{step}] random_state={torch.cuda.get_rng_state()}")
        step_timer.start('forward')
        with nullcontext() if do_opt or args.sync_every_step else model.no_sync():
            with torch.autocast('cuda', torch.bfloat16):
                pred = model(token_batch[:-1])
                loss = (criterion(pred, target)*weight_batch).sum() * loss_scale
                worker_step_loss = loss.item()

            if check_fb_time:
                torch.cuda.synchronize()
            step_timer.start('backward')
            loss.backward()
            if check_fb_time:
                torch.cuda.synchronize()

        ## add step info 2
        worker_opt_accum_loss += worker_step_loss

        ## write step info online
        step_recorder.record(batch_size=batch_size, max_len=max_len, 
                worker_weight=worker_step_weight, worker_loss=worker_step_loss)

        ## check nan
        step_timer.start('check_nan')
        if args.reset_nan_grad:
            grad_is_finite = np.all([torch.all(torch.isfinite(param.grad)).item() for param in model.parameters()])
            if not grad_is_finite:
                logger.warning("nan or infinite value in gradient. Gradient is reset.", **NO_DUP)
                for name, param in model.module.named_parameters():
                    n_nan = torch.sum(torch.isnan(param.grad)).item()
                    n_inf = torch.sum(torch.isinf(param.grad)).item()
                    if n_nan > 0 or n_inf > 0:
                        logger.debug(f"{name}: {n_nan=}, {n_inf=}", **NO_DUP)

                ## save situation
                if is_save_rank and not nan_grad_step_saved:
                    nan_dir = f"{result_dir}/nan_steps/{step}/{rank}"
                    os.makedirs(nan_dir, exist_ok=True)
                    torch.save(batch.detach().cpu(), f"{nan_dir}/batch.pt")
                    torch.save(model.state_dict(), f"{nan_dir}/model.pth")
                    nan_grad_step_saved = True
                
                ## reset grad
                worker_opt_accum_loss = 0
                optimizer.zero_grad()
        
        ## Log gpuuse
        step_timer.start('log')
        if is_starting:
            logger.debug(f"Actual GPU use={torch.cuda.max_memory_allocated(device)/2**30:.03f}")

        ## Log grad
        if check_grad:
            name, param = next(model.named_parameters())
            logger.debug(f"step[{step}] grad[{name}]={param.grad.ravel()[:5]}")

        ## End starting
        if step+1 == 5: 
            step_timer.log_interval = 10000
            if check_data_dist:
                if rank == DATA_RANK['train']:
                    train_reveal_loader.enabled = False
        

        if  should_show(step+1, args.log_step):
            logger.info(f"[Finish]{step+1:>6} step t={time()-train_start:>9.02f}", **NO_DUP)
        
        # opt
        step_timer.start('optim_init')
        if do_opt:
            ## optimizer
            step_timer.start('optim')
            if args.clip_grad_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_value)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            if check_grad:
                name, param = next(model.named_parameters())
                logger.debug(f"opt[{opt}] grad[{name}]={param.grad.ravel()[:5]}")
            optimizer.zero_grad()

            ## sum opt_accum_loss
            opt_accum_loss = reduce_float(worker_opt_accum_loss, device)

            ## save opt data
            kwargs = {} if args.schedule_free else {'lr': scheduler.get_last_lr()[0]}
            opt_recorder.record(loss=opt_accum_loss, weight=opt_accum_weight, 
                memory=psutil.virtual_memory().used/(2**30), **kwargs)
            opt += 1
            worker_opt_accum_loss = worker_opt_accum_weight = 0.0

            ## scheduler
            if not args.schedule_free:
                scheduler.step()

            if should_show(opt, args.log_opt):
                logger.info(f"[Finish]{opt:>6}  opt t={time()-train_start:>9.02f}", **NO_DUP)

    opt_recorder.flush()
    step_recorder.flush()
    logger.info("Training finished!")
