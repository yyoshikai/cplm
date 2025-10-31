import sys, os, math, yaml, psutil, logging
import itertools as itr
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from contextlib import nullcontext
from logging import Logger, getLogger
from time import time
from glob import glob
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
import rdkit
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, RandomSampler, ConcatDataset
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from schedulefree import RAdamScheduleFree

from .data import RevealIterator
from .data.tokenizer import VocEncoder
from .data.collator import DDPStringCollateLoader, InfiniteLoader
from .model import Model, MambaModel
from .utils import git_commit, git_get_hash, reveal_data, should_show
from .utils.logger import add_stream_handler, add_file_handler
from .utils.rdkit import set_rdkit_logger
from .utils.model import get_num_params, get_model_size
from .utils.path import cleardir
from .utils.ddp import reduce_float
from .utils.random import ddp_set_random_seed, set_deterministic
from .utils.time import TimerTqdm
from .utils import IterateRecorder, remove_module

def init_ddp():
    dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
    rank = dist.get_rank()
    size = dist.get_world_size()
    device = torch.device('cuda', index=rank % torch.cuda.device_count()) \
        if torch.cuda.is_available() else torch.device('cpu')
    return rank, size, device

NO_DUP = {'extra': {'no_dup': True}}
def get_process_ranks() -> tuple[int, int, dict[str, int]]:
    MAIN_RANK = 1
    SAVE_RANK = 0
    DATA_RANK = {'train': 3, 'valid': 2}
    size = dist.get_world_size()
    return MAIN_RANK % size, SAVE_RANK % size, \
        {key: rank % size for key, rank in DATA_RANK.items()}

def get_early_stop_opt(result_dir: str, patience_val: int) -> int:
    df = pd.read_csv(f"{result_dir}/vals/0.csv")
    losses = df['mean_loss'].values
    for val in range(len(losses)):
        if losses[val] <= np.min(losses[val:val+patience_val+1]):
            return df['opt'].values[val]

def _sync_result_dir(result_dir, subdirs):
    main_rank = get_process_ranks()[0]
    if dist.get_rank() == main_rank:
        try:
            # Check if trained model already exists
            exist_max_step = max([int(path.split('/')[-1].split('.')[0]) 
                    for path in glob(f"{result_dir}/models/*.pth")]+[0])
            if exist_max_step >= 50:
                raise ValueError(f"{result_dir} already exists(step={exist_max_step})")

            cleardir(result_dir)
            for subdir in subdirs:
                os.makedirs(f"{result_dir}/{subdir}", exist_ok=True)
            result_dirs = [result_dir]
        except Exception as e:
            print("Exception at _sync_result_dir", e, file=sys.stderr, flush=True)
            result_dirs = [None]
    else:
        result_dirs = [None]
    dist.broadcast_object_list(result_dirs, src=main_rank)
    result_dir = result_dirs[0]
    if result_dir is None:
        raise ValueError

def _set_sdp_kernel(sdp_kernel: str|None):
    if sdp_kernel is not None:
        assert sdp_kernel in ['FLASH', 'CUDNN', 'MATH', 'EFFICIENT', 'ALL'] # 全て無効にするとエラーになる
        torch.backends.cuda.enable_flash_sdp(sdp_kernel in ['FLASH', 'ALL'])
        torch.backends.cuda.enable_cudnn_sdp(sdp_kernel in ['CUDNN', 'ALL'])
        torch.backends.cuda.enable_math_sdp(sdp_kernel in ['MATH', 'ALL'])
        torch.backends.cuda.enable_mem_efficient_sdp(sdp_kernel in ['EFFICIENT', 'ALL'])

def _get_train_logger(result_dir):

    # ddp info
    rank = dist.get_rank()
    size = dist.get_world_size()
    is_main = rank == get_process_ranks()[0] % size


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
    add_file_handler(logger, f"{result_dir}/data/example_log/{rank}.log", fmt=fmt, mode='a')
    add_file_handler(token_logger, f"{result_dir}/data/example_log/{rank}.log", fmt=fmt, mode='a')
    add_file_handler(unk_logger, f"{result_dir}/logs/unknowns.log", fmt=fmt, mode='a')

    # third-party modules
    set_rdkit_logger()
    getLogger('.prody').setLevel(logging.CRITICAL)
    transformers.utils.logging.enable_propagation()
    transformers.utils.logging.disable_default_handler()

    return logger, token_logger

def add_train_args(parser: ArgumentParser):
    """
    環境変数等の入力
    finetuning, downstream等でも毎回入力
    """
    # training
    parser.add_argument("--studyname", default='noname')
    parser.add_argument("--weight-per-opt", type=int, default=int(1.6e6))
    parser.add_argument("--max-opt", type=int, required=True)
    ## optimizer
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad-value", type=float, default=None)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--loss-scale")
    ## scheduler
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--scheduler", default='warmup', choices=['warmup', 'warmup_inverse', 'step'])
    parser.add_argument("--schedule-free", action='store_true')
    # process
    parser.add_argument("--coord-noise-std", type=float, default=50.0)
    # model
    ## Transformer
    parser.add_argument('--pos-buffer-len', type=int, default=1600)
    # environments
    parser.add_argument("--eval-opt", type=int, required=True) # temp
    parser.add_argument("--patience-opt", type=int)
    parser.add_argument("--seed", type=int)
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
    parser.add_argument("--log-step", type=int)
    parser.add_argument("--log-opt", type=int)
    ## compatibility
    parser.add_argument("--weight-decay-all", action='store_true')
    ## test
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument("--sync-every-step", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--check", nargs='*', default=[], choices=['early_stop', 
            'data_dist', 'data_epoch', 'data_loading', 'grad', 'random_state', 
            'forward_backward_time', 'optimizer'])
    parser.add_argument('--no-commit', action='store_true')

def add_pretrain_args(parser: ArgumentParser):
    """
    finetune時にpretrain時のものを使うもの (変えてはいけないもの)
    """
    # process
    # bool系は何も指定しない場合BindGPTの設定になるようにしている
    # pocket-heavy-coordはデフォルトで入れるようにした。
    parser.add_argument("--lig-randomize", action='store_true')
    parser.add_argument("--no-lig-h-atom", action='store_true')
    parser.add_argument("--no-lig-h-coord", action='store_true')
    parser.add_argument("--no-pocket-heavy-atom", action='store_true')
    parser.add_argument("--no-pocket-heavy-coord", action='store_true')
    parser.add_argument("--pocket-h-atom", action='store_true') # Datasetによっては無効？
    parser.add_argument("--pocket-h-coord", action='store_true') # Datasetによっては無効？
    parser.add_argument("--coord-range", type=int, default=250)
    parser.add_argument("--coord-follow-atom", action='store_true')
    # model
    parser.add_argument('--mamba', action='store_true')
    parser.add_argument('--n-layer', type=int)
    ## compatibility
    parser.add_argument("--model-bfloat16", action='store_true')

def update_pretrain_args(args: Namespace, targs: dict):
    for key, value in targs.items():
        if not hasattr(args, key):
            setattr(args, key, value)

def set_default_args(args: Namespace):
    if args.num_workers > 0 and args.prefetch_factor is None:
        args.prefetch_factor = 10

    # log interval
    if args.tokenizer_log_interval is None:
        args.tokenizer_log_interval = int(1e6) if args.test else int(1e8)
    if args.log_step is None:
        args.log_step = 1 if args.test else 10000
    if args.log_opt is None:
        args.log_opt = 1 if args.test else max(1, args.eval_opt//5)
    if args.test: args.studyname+='_test'

    # post_init
    args.gpu_size = args.gpu_size_gb * (2**30)
    return args

def get_scheduler(optimizer: Optimizer, scheduler: str, epoch_step: int, 
        warmup_ratio: int):
    warmup_step = warmup_ratio * epoch_step
    match scheduler:
        case 'warmup':
            def schedule(step: int):
                if step < warmup_step:
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

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor):
        output = super().forward(input.reshape(target.numel(), -1), target.ravel())
        if self.reduction == 'none':
            output = output.reshape_as(target)
        return output

def get_model(args: Namespace, voc_encoder: VocEncoder, init_state_path: str, device: torch.device):
    
    if args.mamba:
        kwargs = {}
        if args.n_layer is not None: kwargs['num_hidden_layers'] = args.n_layer
        model = MambaModel(voc_encoder.i2voc, voc_encoder.pad_token, '[END]', **kwargs)
    else:
        num_layers = args.n_layer or 12
        model = Model(num_layers, 768, 12, 4, 0.0, 'gelu', True, voc_encoder.i2voc, voc_encoder.pad_token, args.pos_buffer_len)
    model.to(device)
    if args.model_bfloat16:
        model.to(torch.bfloat16)
    if init_state_path is not None:
        state = torch.load(init_state_path, map_location=device, weights_only=True)
        model.load_state_dict(remove_module(state)) # temp
    return model

def log_batch(prefix: str, logger: Logger, token_logger: Logger, target_batch: Tensor, weight_batch: Tensor, voc_encoder: VocEncoder, step: int, check_data_dist: bool, gpuuse_getter: Callable[[int, int], float]):

    # weight
    weight_items = weight_batch.sum(dim=0).tolist()
    length, batch_size = target_batch.shape
    gpuuse_gb = gpuuse_getter(batch_size, length) / (2**30)
    logger.debug(f"{prefix}[{step}] batch shape={tuple(target_batch.shape)} GPU use={gpuuse_gb}")
    if check_data_dist:
        logger.debug(f"{prefix}[{step}] weights={weight_items}")

    # tokens
    if batch_size > 10: 
        rstate = np.random.RandomState(step)
        idxs = np.sort(rstate.choice(batch_size, size=10, replace=False))
    else:
        idxs = np.arange(batch_size)
    for idx in idxs:
        msg = f"{prefix}[{step}] batch[{idx:3}]="
        tokens = voc_encoder.decode(target_batch[:,idx].cpu().tolist())
        for t, w in zip(tokens, weight_batch[:,idx]):
            msg += f"{t}[{w}],"
        msg += f" remaining weights= {reveal_data(weight_batch[len(tokens):,idx])}"
        token_logger.debug(msg)

def set_env(result_dir: str, args: Namespace, preparation_logs, subdirs):
    # init DDP
    rank, size, device = init_ddp()
    MAIN_RANK, SAVE_RANK, DATA_RANK = get_process_ranks()
    is_main = rank == MAIN_RANK

    # make result dir
    _sync_result_dir(result_dir, subdirs)

    # logging
    logger, token_logger = _get_train_logger(result_dir)
    logger.debug(f"{device=}, {torch.cuda.device_count()=}")
    logger.info(f"{rdkit.__version__=}", **NO_DUP)
    logger.info(f"{transformers.__version__=}", **NO_DUP)

    # save args
    if is_main:
        with open(f"{result_dir}/args.yaml", 'w') as f:
            yaml.dump({**vars(args), 'ddp_size': size}, f)

    # commit & log hash of git
    if is_main:
        if args.test or args.no_commit:
            committed = False
        else:
            committed = git_commit()
        logger.debug('git committed.' if committed else 'git not committed.')
        logger.info(f"git hash={git_get_hash()}")
    
    # Fix seed
    ddp_set_random_seed(args.seed)
    if args.test or args.deterministic: 
        set_deterministic()

    # scaled dot product attention kernel
    _set_sdp_kernel(args.sdp_kernel)
    logger.debug(f"{torch.backends.cuda.cudnn_sdp_enabled()=}")
    logger.debug(f"{torch.backends.cuda.flash_sdp_enabled()=}")
    logger.debug(f"{torch.backends.cuda.math_sdp_enabled()=}")
    logger.debug(f"{torch.backends.cuda.mem_efficient_sdp_enabled()=}")
    logger.debug(f"{os.environ.get('TORCH_CUDNN_SDPA_ENABLED')=}")

    ## Log args
    logger.info(f"[Logs in preparation]", **NO_DUP)
    for log in preparation_logs:
        logger.info(log, **NO_DUP)
    logger.info('')

    return logger, token_logger, rank, device

def get_optimizer_scheduler(model: nn.Module, max_opt: int, 
        weight_decay_all: bool, weight_decay: float, schedule_free: bool, 
        scheduler: str, lr: float, warmup_ratio: float, log_optimizer: bool):
    logger = getLogger('optimizer_scheduler')
    
    ## param groups
    if weight_decay_all or weight_decay == 0:
        params = [{
            "weight_decay": weight_decay, 
            "params": list(model.parameters())
        }]
    else:
        # Partially from TRL
        params_to_decay = get_parameter_names(model,
                forbidden_layer_types=ALL_LAYERNORM_LAYERS, 
                forbidden_layer_names=["bias", "layernorm", "rmsnorm"])
        params = [{   
            "weight_decay": weight_decay,
            "params": [param for name, param in model.named_parameters() 
                if name in params_to_decay and param.requires_grad], 
        }, {
            "weight_decay": 0.0,
            "params": [param for name, param in model.named_parameters()
                if name not in params_to_decay and param.requires_grad]
        }]
        if log_optimizer:
            logger.debug(f"weight_decay: {[name for name, param in model.named_parameters() if name in params_to_decay and param.requires_grad]}")
            logger.debug(f"non weight_decay: {[name for name, param in model.named_parameters() if name not in params_to_decay and param.requires_grad]}")
    ## optimizer & scheduler
    if schedule_free:
        optimizer = RAdamScheduleFree(params, lr=lr)
        scheduler_ = None
        optimizer.train()
    else:
        if weight_decay == 0:
            optimizer = torch.optim.Adam(params, lr=lr)
        else:
            optimizer = torch.optim.AdamW(params, lr=lr)
        optimizer.zero_grad()
        scheduler_ = get_scheduler(optimizer, scheduler, max_opt, warmup_ratio)
    if log_optimizer:
        logger.debug(f"{optimizer=}")
        logger.debug(f"param_groups={[len(group['params']) for group in optimizer.param_groups]}")
    return optimizer, scheduler_

def train(tname: str, args: Namespace, train_datas: list[Dataset[tuple[Tensor, Tensor]]], valid_datas: list[Dataset[tuple[Tensor, Tensor]]], voc_encoder: VocEncoder, preparation_logs: list[str], data_names: list[str], init_state_path: str=None):

    result_dir = os.path.join(tname, 'results', args.studyname)
    logger, token_logger, rank, device = set_env(result_dir, args, preparation_logs, 
            subdirs=['models', 'opts', 'vals', 'optimizers', 'steps/times', 'data/example_log'])
    logger.info(f"num_workers={args.num_workers}", **NO_DUP)
    logger.debug(f"{args.log_opt=}")
    logger.debug(f"{args.log_step=}")
    MAIN_RANK, SAVE_RANK, DATA_RANK = get_process_ranks()

    ## checks
    check_data_dist = 'data_dist' in args.check
    check_grad = 'grad' in args.check
    check_random_state = 'random_state' in args.check
    check_fb_time = 'forward_backward_time' in args.check
    
    # Model
    model = get_model(args, voc_encoder, init_state_path, device)
    model = DistributedDataParallel(model)
    logger.info(f"# of params={get_num_params(model):,}", **NO_DUP)
    logger.info(f"Model size={get_model_size(model)/2**30:.02f}GB", **NO_DUP)

    # Make timer here to send to DDPStringCollateLoader
    step_timer = TimerTqdm(itr.count(), time_path=f"{result_dir}/steps/times/{rank}.csv", file_interval=10000, log_interval=10000, desc='step', disable_bar=True)

    # DataLoader
    if rank == DATA_RANK['train']:
        train_data = ConcatDataset(train_datas)
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
    
    ## collated data loader
    train_loader = DDPStringCollateLoader(train_loader, collate, get_gpuuse, 
            args.gpu_size, device, 100000, DATA_RANK['train'], 
            f"{result_dir}/data/train_large_items.csv", step_timer if 'data_loading' in args.check else None)
    train_iter = train_loader.__iter__()

    # criterion
    criterion = CrossEntropyLoss(reduction='none', ignore_index=voc_encoder.pad_token)

    # optimizer & scheduler 
    optimizer, scheduler = get_optimizer_scheduler(model, args.max_opt, args.weight_decay_all, 
            args.weight_decay, args.schedule_free, args.scheduler, args.lr, args.warmup_ratio, 
            'optimizer' in args.check)
    ## loss scale
    match args.loss_scale:
        case None:
            loss_scale = 1
        case _:
            loss_scale = float(args.loss_scale)
    # opt recorder
    opt_recorder = IterateRecorder(f"{result_dir}/opts/{rank}.csv", args.eval_opt)

    # val
    valid2train_r = torch.tensor([len(train_data)/len(valid_data) 
            for train_data, valid_data in zip(train_datas, valid_datas)], 
            device=device, dtype=torch.float)
    val_recorder = IterateRecorder(f"{result_dir}/vals/{rank}.csv", 1)
    val2opt = []
    val2mean_loss = []
    next_eval_opt = 0

    # step
    ## initial state
    opt = 0
    worker_opt_accum_loss = 0.0
    worker_opt_accum_weight = 0.0
    nan_grad_step_saved = False
    ## recorder
    step_recorder = IterateRecorder(f"{result_dir}/steps/{rank}.csv", 1000)

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
            for i_data, prefix in enumerate(data_names):
                logger.info(f"    Validating [{prefix}]", **NO_DUP)
                ## Make Loader
                if rank == DATA_RANK['valid']:
                    os.environ['EPOCH'] = '0'
                    valid_loader = DataLoader[tuple[Tensor, Tensor]](valid_datas[i_data], batch_size=None, shuffle=True if check_data_dist else False, num_workers=args.num_workers, pin_memory=args.pin_memory, prefetch_factor=args.prefetch_factor)
                    if valid_is_starting:
                        if check_data_dist:
                            valid_loader = RevealIterator(valid_loader, f'valid[{prefix}]', logger)
                            valid_reveal_loader = valid_loader
                else:
                    valid_loader = None
                valid_loader = DDPStringCollateLoader(valid_loader, collate, get_gpuuse, args.gpu_size, device, 100000 if valid_is_starting else math.inf, DATA_RANK['valid'])

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
                            log_batch(f"valid[{opt}][{prefix}]", logger, token_logger, target, 
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
            val_recorder.record(opt=opt, mean_loss=mean_loss,  
                    **{f'process_weight {prefix}': process_weight 
                        for prefix, process_weight in zip(data_names, process_weights)}, 
                    **{f'process_loss {prefix}': process_loss
                        for prefix, process_loss in zip(data_names, process_losses)})
            val2opt.append(opt)
            val2mean_loss.append(mean_loss)

            if rank == MAIN_RANK:
                torch.save(model.module.state_dict(), f"{result_dir}/models/{opt}.pth")
                cleardir(f"{result_dir}/optimizers")
                torch.save(optimizer.state_dict(), f"{result_dir}/optimizers/{opt}.pth")

            # Judge early stopping
            best_opt = val2opt[np.argmin(val2mean_loss)]
            if args.patience_opt is not None and opt - best_opt >= args.patience_opt:
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
                if rank == SAVE_RANK and not nan_grad_step_saved:
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
    val_recorder.flush()
    logger.info("Training finished!")

    dist.destroy_process_group()