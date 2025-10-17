import sys, os, yaml, shutil, psutil, gc, math, random, re
import itertools as itr
import concurrent.futures as cf
from argparse import ArgumentParser
from logging import getLogger
import numpy as np, pandas as pd
from addict import Dict
from glob import glob
from contextlib import nullcontext
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from torch.utils.data import Dataset, DataLoader, StackDataset
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical

from src.data._sampler import InfiniteRandomSampler
from src.data import index_dataset
from src.utils import IterateRecorder
from src.utils.path import cleardir
from src.evaluate import parse_mol_tokens, parse_mol
from src.evaluate import eval_vina, eval_qvina3
from src.train import set_env, get_model, get_optimizer_scheduler, get_process_ranks, log_batch
from src.model import Model, MambaModel
from src.finetune import get_finetune_data
WORKDIR = os.environ.get('WORKDIR', os.path.abspath('..'))

# arguments
parser = ArgumentParser()
def add_env_args(parser: ArgumentParser):
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--no-commit', action='store_true')
    parser.add_argument("--deterministic", action='store_true')
    parser.add_argument("--sdp-kernel", choices=['FLASH', 'EFFICIENT'], default='FLASH')
## reward
parser.add_argument('--target', choices=['min_vina', 'vina', 'mw_max', 'logp', 'qvina', 'dummy'], default='min_vina')
parser.add_argument('--error-score', type=float, default=None)
parser.add_argument('--ignore-error', action='store_true')
parser.add_argument('--min-score', type=float, default=-math.inf)
parser.add_argument('--max-len', type=int, default=2500)
parser.add_argument('--reward-scale', choices=['none', 'all_mean', 'sample_mean', 'rank_mean', 'rank_mean_std'], default='none')
parser.add_argument('--alpha', type=float, default=0.05) # same as BindGPT
## Data
parser.add_argument('--generate-per-sample', type=int, default=1)
## training
parser.add_argument('--studyname', required=True)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--max-opt', type=int, default=1000)
## optimizer
parser.add_argument('--weight-decay', type=float, default=0.0) # same as BindGPT
parser.add_argument('--clip-grad-value', type=float, default=None)
parser.add_argument('--clip-grad-norm', type=float, default=1.0) # same as BindGPT
parser.add_argument('--loss-scale')
## scheduler
parser.add_argument('--lr', type=float, default=1.4e-5) # same as BindGPT
parser.add_argument('--scheduler', default='constant') # same as BindGPT
parser.add_argument("--schedule-free", action='store_true')
parser.add_argument("--warmup-ratio", type=float, default=0.04)
## finetune
parser.add_argument('--finetune-name', required=True)
parser.add_argument('--finetune-opt', type=int)
## environment
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--pin-memory', action='store_true')
parser.add_argument('--prefetch-factor', type=int)
parser.add_argument('--num-score-workers', type=int, default=1)
parser.add_argument('--reset-nan-grad', action='store_true')
parser.add_argument('--gc', action='store_true')
parser.add_argument('--path-to-qvina', default=f"{WORKDIR}/github/qvina/bin/qvina02")
## verbosity
parser.add_argument('--tqdm-generate', action='store_true')
parser.add_argument('--record-opt', type=int)
parser.add_argument('--tokenizer-log-interval', type=int)
## test
parser.add_argument('--use-categorical', action='store_true')
parser.add_argument('--fix-pocket', action='store_true')
parser.add_argument("--weight-decay-all", action='store_true')
parser.add_argument('--check', nargs='*', default=[], choices=['data_dist'])
add_env_args(parser)
args = parser.parse_args()
## set default args
if args.test: args.studyname+='_test'
if args.record_opt is None:
    args.record_opt = 1 if args.test else 100
if args.tokenizer_log_interval is None:
    args.tokenizer_log_interval = 10000 if args.test else int(1e7)

logs = []

# get finetune info
finetune_dir = f"finetune/results/{args.finetune_name}"
fargs = Dict(yaml.safe_load(open(f"{finetune_dir}/args.yaml")))
pname = fargs.pretrain_name
pretrain_dir = f"training/results/{pname}"
pargs = Dict(yaml.safe_load(open(f"{pretrain_dir}/args.yaml")))
coord_follow_atom = pargs.get('coord_follow_atom', False)

## check finetuning
assert fargs.no_score

## get last finetune step
if args.finetune_opt is None:
    steps = [os.path.splitext(os.path.basename(step))[0] for step in glob(f"{finetune_dir}/models/*")]
    steps = [int(step) for step in steps if step.isdigit()]
    if len(steps) == 0:
        raise ValueError("No checkpoint was found to get args.finetune_opt")
    args.finetune_opt = max(steps)
    logs.append(f"finetune_opt was set to {args.finetune_opt}")

batch_first = False
log_sample_step = 3
do_save_steps = [0, 1, 2, 3, 4, 50, 100]+list(range(200, 1000, 200)) \
        +list(range(1000, args.max_opt, 1000))

# data
## vocs from state_dict
state_dict = torch.load(f"{finetune_dir}/models/{args.finetune_opt}.pth", weights_only=True)
vocs = state_dict['vocs' if 'vocs' in state_dict else 'module.vocs'][1:]

added_vocs = set(vocs)
voc_encoder, raw_data, token_data, weight_data, center_data, rotation_data,\
        protein_filename_data, ligand_filename_data \
        = get_finetune_data(fargs, 'train', False, True, added_vocs, 'none')
index_data, token_data = index_dataset(token_data)
train_data = StackDataset(index_data, token_data, center_data, rotation_data, 
        protein_filename_data, ligand_filename_data)
## Scoring function 最大化したいものとする
match args.target:
    case 'min_vina':
        def get_score(lig_path: str, rec_path: str, out_dir: str):
            score, min_score = eval_vina(lig_path, rec_path, out_dir)
            return -min_score if min_score is not None else np.nan
        error_score = -50
    case 'vina':
        def get_score(lig_path: str, rec_path: str, out_dir: str):
            score, min_score = eval_vina(lig_path, rec_path, out_dir)
            return -score if min_score is not None else np.nan
        error_score = -50
    case 'mw_max':
        def get_score(lig_path: str, rec_path: str, out_dir: str):
            mol = Chem.SDMolSupplier(lig_path).__next__()
            return rdMolDescriptors.CalcExactMolWt(mol) if mol is not None else np.nan
        error_score = 0
    case 'qvina':
        def get_score(lig_path: str, rec_path: str, out_dir: str):
            score = eval_qvina3(lig_path, rec_path, out_dir, timeout=60, path_to_qvina=args.path_to_qvina)
            return -score if score is not None else np.nan
        error_score = -50
    case 'dummy':
        def get_score(lig_path: str, rec_path: str, out_dir: str):
            return random.random()
        error_score = 0
if args.error_score is not None:
    error_score = args.error_score

# Environment
result_dir = f"reinforce/results/{args.studyname}"
logger, token_logger, rank, device = set_env(result_dir, args, logs, 
        subdirs=['models', 'opts', 'errors', 'scores'])
MAIN_RANK, SAVE_RANK, DATA_RANK = get_process_ranks()
os.makedirs(f"{result_dir}/generated/{rank}", exist_ok=True)
for i in range(args.batch_size):
    os.makedirs(f"{result_dir}/eval_vina_tmp/{rank}/{i}", exist_ok=True)
RDLogger.DisableLog("rdApp.*")
## check generate_per_sample
ddp_size = dist.get_world_size()

# model
init_state_path = f"{finetune_dir}/models/{args.finetune_opt}.pth"
init_model = get_model(pargs, voc_encoder, init_state_path, device)
init_model.to(device)
net_model = get_model(pargs, voc_encoder, init_state_path, device)
net_model.to(device)
model = DistributedDataParallel(net_model)
model.to(device)
from src.utils.ddp import dist_broadcast_tensor
def get_gpuuse(batch_size: int, length: int):
    if isinstance(model.module, Model):
        return model.module.get_gpuuse(batch_size, length, True, args.sdp_kernel)
    else:
        return model.module.get_gpuuse(batch_size, length, True)

# DataLoader
class ReinforceIter:
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, dataset: Dataset, num_workers:int, pin_memory: bool, prefetch_factor: int, 
            batch_size: int, batch_first: bool, padding_value: int, repeat_per_sample: int,
            fix_pocket: bool):
        self.size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.main_rank = DATA_RANK['train']
        self.batch_size = batch_size
        self.repeat_per_sample = repeat_per_sample
        self.batch_first = batch_first
        self.padding_value = padding_value
        assert self.batch_size * self.size % self.repeat_per_sample == 0
        self.fix_pocket = fix_pocket

        if self.rank == self.main_rank:
            loader = DataLoader(dataset, batch_size=None, 
                sampler=InfiniteRandomSampler(dataset, generator=torch.Generator().manual_seed(args.seed)),
                num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
            self.logger.info(f"Loading filenames.csv.gz ...")
            self.logger.info("Loaded.")
            self.iter = loader.__iter__()
            self.next_item = None
            self.step = 0

            if self.fix_pocket:
                self.fixed_item = self.iter.__next__()
                del self.iter

    def __next__(self) -> tuple[Tensor, Tensor, Tensor]:
        if self.rank == self.main_rank:
            all_idxs = []
            all_items = []
            for _ in range(self.batch_size*self.size // self.repeat_per_sample):
                idx_item = self.fixed_item if self.fix_pocket else self.iter.__next__()
                all_idxs += [idx_item[0]] * self.repeat_per_sample
                all_items += [idx_item[1:]] * self.repeat_per_sample
            all_idxs = torch.tensor(all_idxs, dtype=torch.int, device=device)
            batched_items = [all_items[r*self.batch_size:(r+1)*self.batch_size] for r in range(self.size)]
        else:
            all_idxs = batched_items = None
        all_idxs = dist_broadcast_tensor(all_idxs, device, self.main_rank, (self.batch_size*self.size,), torch.int)
        items_box = [None]
        dist.scatter_object_list(items_box, batched_items)
        tokens, centers, rotations, pfnames, lfnames = zip(*items_box[0])
        return all_idxs, tokens, centers, rotations, pfnames, lfnames
train_iter = ReinforceIter(train_data, args.num_workers, 
    args.pin_memory, args.prefetch_factor, args.batch_size, batch_first, 
    voc_encoder.pad_token, args.generate_per_sample, args.fix_pocket)

# optimizer
optimizer, scheduler = get_optimizer_scheduler(args, model, 55000)
optimizer.zero_grad()
match args.loss_scale:
    case None:
        loss_scale = 1.0
    case _:
        loss_scale = float(args.loss_scale)

## records
opt_recorder = IterateRecorder(f"{result_dir}/opts/{rank}.csv", 
        ['batch_size', 'max_len', 'reward_loss', 'kl_loss'], 1000)
errorss = []
scoress = []
nan_grad_step_saved = False
step = 0

# save at step 0
if rank == SAVE_RANK:
    torch.save(model.state_dict(), f"{result_dir}/models/{step}.pth")
    cleardir(f"{result_dir}/optimizers")
    torch.save(optimizer.state_dict(), f"{result_dir}/optimizers/{step}.pth")

logger.info("Training started.")
for step in range(args.max_opt): 

    # get batch
    all_idxs, prompt_tokens, centers, rotations, pfnames, lfnames= train_iter.__next__()
    prompt_sizes = [len(token) for token in prompt_tokens]
    prompt_batch = pad_sequence(prompt_tokens, False, voc_encoder.pad_token)
    prompt_batch = prompt_batch.to(device)
    L, B = prompt_batch.shape

    # forward
    ## generate sample
    model.eval()
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
        outputs = net_model.generate2(prompt_batch, '[END]', args.max_len, voc_encoder.pad_token, 10, args.tqdm_generate) # [B, L]

    out_batch = pad_sequence([torch.cat([prompt.to(device), output]) for prompt, output in zip(prompt_tokens, outputs)], batch_first, padding_value=voc_encoder.pad_token) # [L, B]
    Lo, B = out_batch.shape
    dtype = torch.float
    weight = torch.zeros((Lo-1, B), device=device, dtype=dtype) # [Lo-1, B]
    for b, (prompt_size, output) in enumerate(zip(prompt_sizes, outputs)):
        weight[prompt_size-1:prompt_size+len(output)-1] = 1.0

    ## Log output
    if step < log_sample_step:
        log_batch('train', logger, token_logger, out_batch[1:], weight, voc_encoder, step, 'data_dist' in args.check, get_gpuuse)
        ## check distribution
        if rank == ddp_size-1:
            logger.debug(f"step[{step}]{all_idxs=}")
            logger.debug(f"step[{step}]{pfnames=}")
            logger.debug(f"step[{step}]{lfnames=}")
            logger.debug(f"step[{step}]{centers=}")
            
    ## Get score
    do_save = step in do_save_steps
    errors = []
    with cf.ProcessPoolExecutor(args.num_score_workers) if (args.num_score_workers >= 2) else nullcontext() as e:
        futures = []
        valid_scores = []
        for idx in range(len(outputs)):

            center = centers[idx].numpy()
            rotation = rotations[idx].numpy()
            lfname = lfnames[idx]
            pfname = pfnames[idx]
            if do_save:
                eval_dir = f"{result_dir}/eval_vina/{step}/{rank}/{idx}"
                os.makedirs(eval_dir, exist_ok=True)
            else:
                eval_dir = f"{result_dir}/eval_vina_tmp/{rank}/{idx}"

            score = np.nan
            out_tokens = ['[LIGAND]']+voc_encoder.decode(outputs[idx].tolist())

            if do_save:
                info = {'pfname': pfname, 'lfname': lfname, 'idx': idx, 
                        'center': center.tolist(), 'rotation': rotation.tolist()}
                with open(f"{eval_dir}/info.yaml", 'w') as f:
                    yaml.dump(info, f)
                with open(f"{eval_dir}/tokens.txt", 'w') as f:
                    f.write(','.join(out_tokens)+'\n')
            error, smiles, coords = parse_mol_tokens(out_tokens)
            
            if error != '':
                errors.append(error)
                continue
            
            rotation_inv = np.linalg.inv(rotation)
            coords = np.matmul(coords, rotation_inv) + center
            error, mol = parse_mol(smiles, coords)
            errors.append(error)

            if error != '':
                continue
            
            with open(f"{eval_dir}/lig.sdf", 'w') as f:
                f.write(Chem.MolToMolBlock(mol))
            
            lig_path = f"{eval_dir}/lig.sdf"
            actual_pfname = re.match(r"(.+?/.+?_rec)_.+", pfname).group(1)+'.pdb'
            rec_path = f"{WORKDIR}/cheminfodata/crossdocked/targetdiff/crossdocked_v1.1_rmsd1.0/{actual_pfname}"
            if args.num_score_workers >= 2:
                futures.append(e.submit(get_score, 
                        lig_path=lig_path, rec_path=rec_path, out_dir=eval_dir))
            else:
                valid_scores.append(get_score(lig_path=lig_path, rec_path=rec_path, out_dir=eval_dir))
        if args.num_score_workers >= 2:
            valid_scores = np.array([f.result() for f in futures])
        else:
            valid_scores = np.array(valid_scores)

    errors = np.array(errors)
    scores = np.full(len(errors), np.nan)
    scores[errors == ""] = valid_scores
    errors[errors == ""][np.isnan(valid_scores)] = 'VINA'

    errorss.append(errors)
    scores = torch.tensor(scores, device=device, dtype=torch.float)
    if not args.ignore_error:
        scores[torch.isnan(scores)] = error_score
    torch.clamp_(scores, min=args.min_score)
    scoress.append(scores.cpu().tolist())

    ## gather & normalize score
    all_scores = [torch.zeros(args.batch_size, dtype=torch.float, device=device)
            for _ in range(ddp_size)]
    dist.all_gather(all_scores, scores)
    all_scores = torch.cat(all_scores)
    
    match args.reward_scale:
        case 'none': 
            pass
        case 'all_mean':
            if torch.any(torch.isfinite(all_scores)):
                scores = scores - torch.nanmean(all_scores)
        case 'sample_mean':
            idxs = all_idxs[rank*args.batch_size:(rank+1)*args.batch_size]
            unique_idxs = idxs.unique()
            for uidx in unique_idxs:
                if torch.any(torch.isfinite(all_scores[all_idxs == uidx])):
                    scores[idxs == uidx] -= torch.nanmean(all_scores[all_idxs == uidx])
        case 'rank_mean':
            if torch.any(torch.isfinite(scores)):
                scores = scores - torch.nanmean(scores)
        case 'rank_mean_std':
            if torch.any(torch.isfinite(scores)):
                scores = scores - torch.nanmean(scores)
                if torch.sum(torch.isfinite(scores)) >= 2:
                    scores = scores / (torch.std(scores[torch.isfinite(scores)])+1.0e-8)
    if args.ignore_error:
        scores[torch.isnan(scores)] = 0.0

    if step < 5:
        logger.info(f"step {step} scores={scores.cpu().tolist()}")

    ## Get prob & reward loss
    model.train()
    with torch.autocast('cuda', torch.bfloat16):
        logits = model(out_batch[:-1]) # [Lo-1, B, T]
        init_logits = init_model(out_batch[:-1]) # [L, B, N]
        if args.use_categorical:
            cat = Categorical(logits=logits) # ~[Lo-1, B]
            log_probs = cat.log_prob(out_batch[1:]) # [Lo-1, B]
        else:
            log_probs_all = F.log_softmax(logits, dim=-1) # [Lo-1, B, N]
            log_probs = torch.gather(log_probs_all, dim=-1, index=out_batch[1:].unsqueeze(-1)).squeeze(-1) # [Lo-1, B]
        reward_loss = torch.sum(-scores*(log_probs*weight).sum(dim=0)/weight.sum(dim=0))

        ## KL loss
        log_probs_all = F.log_softmax(logits, dim=-1)
        with torch.inference_mode():
            init_log_probs_all = F.log_softmax(init_logits, dim=-1).detach() # [Lo-1, B, N]
        kl_loss = F.kl_div(input=log_probs_all, target=init_log_probs_all, reduction='none', 
            log_target=True) # [Lo-1, B, N]
        kl_loss = kl_loss.sum(dim=-1) # [Lo-1, B]
        kl_loss = torch.sum(kl_loss*weight)
        
        loss = (reward_loss + kl_loss * args.alpha) * loss_scale

    loss.backward()

    # check nan
    if args.reset_nan_grad:
        grad_is_finite = np.all([torch.all(torch.isfinite(param.grad)).item() for param in model.parameters()])
        if not grad_is_finite:
            logger.warning("nan or infinite value in gradient. Gradient is reset.")
            for name, param in model.module.named_parameters():
                n_nan = torch.sum(torch.isnan(param.grad)).item()
                n_inf = torch.sum(torch.isinf(param.grad)).item()
                if n_nan > 0 or n_inf > 0:
                    logger.warning(f"{name}: {n_nan=}, {n_inf=}")

            ## save situation
            if rank == SAVE_RANK and not nan_grad_step_saved:
                nan_dir = f"{result_dir}/nan_step_{step}/{rank}"
                os.makedirs(nan_dir, exist_ok=True)
                torch.save(out_batch.detach().cpu(), f"{nan_dir}/batch.pt")
                torch.save(model.state_dict(), f"{nan_dir}/model.pth")
                nan_grad_step_saved = True
            
            ## reset grad
            optimizer.zero_grad()

    if args.clip_grad_value is not None:
        torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_value)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    step += 1
    opt_recorder.record(batch_size=B, max_len=L, lr=scheduler.get_last_lr()[0], 
            memory=psutil.virtual_memory().used/(2**30), 
            reward_loss=reward_loss.item(), kl_loss=kl_loss.item())
    scheduler.step()
    
    if step % args.record_opt == 0:
        pd.DataFrame(scoress).to_csv(f"{result_dir}/scores/{rank}.csv")
        pd.DataFrame(errorss).to_csv(f"{result_dir}/errors/{rank}.csv")
        if rank == SAVE_RANK:
            torch.save(model.state_dict(), f"{result_dir}/models/{step}.pth")
            cleardir(f"{result_dir}/optimizers")
            torch.save(optimizer.state_dict(), f"{result_dir}/optimizers/{step}.pth")
        if args.gc:
            gc.collect()

    logger.info(f"{step=} finished.")

    if step == log_sample_step:
        getLogger('rdkit').propagate = False
    
logger.info("Training finished!")

dist.destroy_process_group()
