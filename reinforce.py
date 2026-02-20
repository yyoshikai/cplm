import os, yaml, psutil, math, random
import itertools as itr
import concurrent.futures as cf
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from logging import getLogger
from typing import Literal
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, StackDataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.distributed.elastic.multiprocessing.errors import record
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from src.data._sampler import InfiniteRandomSampler
from src.data import index_dataset
from src.utils import IterateRecorder, get_git_hash
from src.utils.path import cleardir
from src.utils.rdkit import ignore_rdkit_warning
from src.utils.ddp import dist_all_gather, reduce_float
from src.utils.logger import NO_DUP
from src.chem import rdmol2obmol, pdb2obmol
from src.evaluate import eval_vina, eval_qvina
from src.train.collator import solve_increasing_fn_left
from src.data.protein import Protein2PDBDataset
from src.data.tokenizer import VocEncoder
from src.model import Model
from src.train import set_env, get_model, get_optimizer_scheduler, get_process_ranks
from src.train.data import get_finetune_data
from src.train.looper import Loopers, TimeWriteLooper, LogLooper, TimeLogLooper, GPUUseLooper, MemorySnapshotLooper
from src.generate.streamer import LigandStreamer, TokenSaveStreamer, PositionSaveStreamer, TimeLogStreamer
WORKDIR = os.environ.get('WORKDIR', os.path.abspath('..'))

## Scoring function
def get_score(target, lig_rdmol: Chem.Mol, rec_pdb: str, out_dir: str, cpu: int, print_prepare: bool):
    match target:
        case 'min_vina':
            score, min_score, error = eval_vina(rdmol2obmol(lig_rdmol), pdb2obmol(rec_pdb), f"{out_dir}/protein_h.pdbqt")
            return -min_score if min_score is not None else np.nan
        case 'vina':
            score, min_score, error = eval_vina(rdmol2obmol(lig_rdmol), pdb2obmol(rec_pdb), f"{out_dir}/protein_h.pdbqt")
            return -score if min_score is not None else np.nan
        case 'mw_max':
            return rdMolDescriptors.CalcExactMolWt(lig_rdmol)
        case 'qvina':
            os.makedirs(out_dir, exist_ok=True)
            with open(f"{out_dir}/rec_input.pdb", 'w') as f:
                f.write(rec_pdb)
            score = eval_qvina(lig_rdmol, f"{out_dir}/rec_input.pdb", out_dir, timeout=60, cpu=cpu, print_prepare=print_prepare)[0]
            return -score if score is not None else np.nan
        case 'dummy':
            return random.random()
# 参考
error_scores = { 'min_vina': -50.0, 'vina': -50.0, 'mw_max': 0.0, 'qvina': -50.0, 'dummy': 0.0}

def get_grads(model: nn.Module, prev_grads: dict[str, Tensor]|None):
    grads = { name: param.grad.clone() if param.grad is not None else None 
            for name, param in model.named_parameters() }
    if prev_grads is not None:
        diff_grads = {}
        for name, grad in grad.items():
            prev_grad = prev_grads[name]
            if grad is None:
                assert prev_grad is None
                diff_grads[name] = None
            else:
                diff_grads[name] = grad if prev_grad is None else grad - prev_grad
        return grads, diff_grads
    else:
        return grads, grads

class ReinforceModel(nn.Module):
    def __init__(self, model: Model, baseline):
        super().__init__()
        self.baseline = baseline
        self.model = model
        if baseline:
            self.value_head = nn.Linear(model.state_size, 1)
    
    def forward(self, src: Tensor, position: Tensor):
        logits, states = self.model(src, position, out_state=True)
        if self.baseline:
            values = self.value_head(states).squeeze(-1)
        else:
            values = None
        return logits, values

class ReinforceDataIter:
    def __init__(self, train_data: Dataset, device: torch.device, batch_size: int, generate_per_sample: int, max_prompt_len: int, fix_pocket: bool, num_workers: int, seed: int):
        _, _, DATA_RANK = get_process_ranks()
        self.data_rank = DATA_RANK['train']
        self.rank = dist.get_rank()
        self.ddp_size = dist.get_world_size()

        self.device = device
        self.batch_size = batch_size
        self.generate_per_sample = generate_per_sample
        self.fix_pocket = fix_pocket
        assert self.batch_size * self.ddp_size % self.generate_per_sample == 0
        if self.rank == self.data_rank:
            loader = DataLoader(train_data, batch_size=None, 
                sampler=InfiniteRandomSampler(train_data, generator=torch.Generator().manual_seed(seed)),
                num_workers=num_workers, pin_memory=True, prefetch_factor=10 if num_workers > 0 else None)
            self.train_iter = loader.__iter__()
            if max_prompt_len is not None:
                self.train_iter = itr.filterfalse(lambda x: len(x[2]) > max_prompt_len, self.train_iter)
            if self.fix_pocket:
                self.train_fixed_item = self.train_iter.__next__()
                del self.train_iter


    def get(self) -> tuple[Tensor, list[str], list[Tensor], list[list[int]]]:
        if self.rank == self.data_rank:
            all_items = []
            for si in range(self.batch_size*self.ddp_size // self.generate_per_sample):
                all_items += [self.train_fixed_item if self.fix_pocket else self.train_iter.__next__()] * self.generate_per_sample
            batched_items = [all_items[r*self.batch_size:(r+1)*self.batch_size] for r in range(self.ddp_size)]
        else:
            batched_items = None
        items_box = [None]
        dist.scatter_object_list(items_box, batched_items, src=self.data_rank)
        idxs, pdbs, prompt_tokens, positions = zip(*items_box[0])
        all_idxs = torch.cat(dist_all_gather(torch.tensor(idxs, dtype=torch.int, device=self.device)))
        prompt_tokens = [token.to(self.device) for token in prompt_tokens]
        positions = [position.tolist() for position in positions]
        return all_idxs, pdbs, prompt_tokens, positions

def generate(
        model: DistributedDataParallel, 
        voc_encoder: VocEncoder,
        result_dir: str, 
        max_new_token: int,
        coord_range: int,
        lig_h_atom: bool,
        lig_h_coord: bool,
        device: torch.device,
        size_recorder: IterateRecorder,

        step: int,
        prompt_tokens,
        positions,
        do_save: bool,
) -> tuple[Tensor, Tensor, Tensor, int, list[Chem.Mol|None], list[str|None]]:
    rank = dist.get_rank()

    model.eval()
    streamers = ligand_streamers = [LigandStreamer(f"{result_dir}/generation/{step}/{rank}_{idx}/new_sdf.sdf" if do_save else None, coord_range, voc_encoder, False, lig_h_atom, lig_h_coord, None) for idx in range(len(prompt_tokens))]
    streamers = token_streamers = [TokenSaveStreamer(streamer) for streamer in streamers]
    streamers = position_streamers = [PositionSaveStreamer(streamer) for streamer in streamers]
    if step < 5:
        streamers = [TimeLogStreamer(streamer, str(b), 10.0) for b, streamer in enumerate(streamers)]
    with torch.inference_mode(), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        model.module.model.generate2(prompt_tokens, positions, streamers, max_new_token)

    tokens = pad_sequence([
        torch.tensor(token_streamer.prompt_tokens+token_streamer.new_tokens)
        for token_streamer in token_streamers
    ], padding_value=voc_encoder.pad_token).to(device=device, dtype=torch.long)
    input = tokens[:-1] # [L, B]
    output = tokens[1:]

    weight = torch.zeros_like(output, dtype=torch.float) # [L, B]
    for b, token_streamer in enumerate(token_streamers):
        prompt_size, new_size = len(token_streamer.prompt_tokens), len(token_streamer.new_tokens)
        weight[prompt_size-1:prompt_size+new_size-1, b] = 1.0
    all_exp = int(reduce_float(torch.sum(weight).item(), device))
    position = pad_sequence([
        torch.tensor(position+position_streamer.new_positions) for position, position_streamer
        in zip(positions, position_streamers)
    ], padding_value=0).to(device=device, dtype=torch.long)[:-2]
    size_recorder.record(prompt=[len(s.prompt_tokens) for s in token_streamers], output=[len(s.new_tokens) for s in token_streamers])
    ligs, errors = zip(*[(streamer.mol, streamer.error) for streamer in ligand_streamers])
    return input, output, position, weight, all_exp, ligs, errors

def all_gather(values: Tensor, dim=0):
    world_size = dist.get_world_size()
    all_values = [torch.zeros_like(values) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    return torch.cat(all_values, dim=dim)

def get_sample_stat(values: Tensor, all_idxs: Tensor) -> tuple[Tensor, Tensor]:
    rank = dist.get_rank()
    all_values = all_gather(values)
    B, = values.shape

    sample_means = torch.full_like(values, math.nan)
    sample_stds = torch.full_like(values, math.nan)
    idxs = all_idxs[rank*B:(rank+1)*B]
    for uidx in torch.unique(idxs):
        uidx_all_values = all_values[all_idxs == uidx]
        if torch.all(torch.isnan(uidx_all_values)):
            continue
        mean = torch.nanmean(uidx_all_values)
        std = torch.nanmean((uidx_all_values-mean)**2).sqrt()
        sample_means[idxs == uidx] = mean
        sample_stds[idxs == uidx] = std
    return sample_means, sample_stds

def get_all_stat(values: Tensor) -> tuple[float, float]:
    all_values = all_gather(values)
    mean = torch.nanmean(all_values).item()
    std = torch.nanmean((all_values-mean)**2).sqrt().item()
    return mean, std

def whiten_scores(scores: Tensor, all_idxs: Tensor, sample_whiten: list[Literal['mean', 'std']], all_whiten: list[Literal['mean', 'std']]):
    world_size = dist.get_world_size()
    # all_gather scores
    all_scores = [torch.zeros_like(scores) for _ in range(world_size)]
    dist.all_gather(all_scores, scores)
    all_scores = torch.cat(all_scores)
    
    # sample whiten
    sample_means, sample_stds = get_sample_stat(scores, all_idxs)
    sample_stds+=1e-5
    sample_mean = 'mean' in sample_whiten
    sample_std = 'std' in sample_whiten
    if sample_mean or sample_std:
        if sample_mean and sample_std:
            scores = (scores-sample_means)/sample_stds
        elif sample_mean:
            scores = scores-sample_means
        elif sample_std:
            scores = (scores-sample_means)/sample_stds+sample_mean

    # all whiten
    all_mean = 'mean' in all_whiten
    all_std = 'std' in all_whiten
    mean, std = get_all_stat(scores)
    if (all_mean or all_std) and not torch.all(torch.isnan(scores)):
        if all_mean and all_std:
            scores = (scores - mean) / (std+1e-5)
        elif sample_mean:
            scores = scores - mean
        elif sample_std:
            scores = (scores - mean) / (std+1e-5) + mean
    return scores

def get_scores(
        pdbs: list[str],
        ligs: list[Chem.Mol|None],
        errors: list[str|None],
        all_idxs: Tensor,

        out_dir: str,
        error_recorder: IterateRecorder,
        score_recorder: IterateRecorder,
        device: torch.device,

        min_valid_score: float,
        gen_error_score: float,
        vina_error_score: float,
        gen_error_sample_deviation: float,
        vina_error_sample_deviation: float,
        sample_whiten: list[Literal['mean', 'std']],
        all_whiten: list[Literal['mean', 'std']],
        gen_error_white_score: float, 
        vina_error_white_score: float,
        sample_rewhiten: list[Literal['mean', 'std']],
        all_rewhiten: list[Literal['mean', 'std']],

        target: str,
        num_score_workers: int,
        cpu: int,
        print_prepare: bool
) -> Tensor:
    """
    1. nanでない報酬を(--min-valid-score, default=-math.inf)でclip
    2. generation errorに(--gen-error-score, default=np.nan)を代入
       vina errorに(--vina-error-score, default=np.nan)を代入
    3. gen_error_sample_deviation, vina_error_sample_deviationがnanでない場合これらを代入
        mean+std*gen_error_sample_deviation 等を代入
    4. (--sample-whiten, default=[])に従ってサンプルごとの正規化
        'mean': 平均=0とする
        'std': 分散=1とする
    5. (--all-whiten, default=[])に従って全体の正規化
    6. (--gen-error-white-score, default=np.nan)がnp.nanでない場合, generationerrorに代入
       (--vina-error-white-score, default=np.nan)がnp.nanでない場合, vina errorに代入
    7. (--sample-rewhiten, default=[])に従ってサンプルごとの正規化
    8. (--all-rewhiten, default=[])に従って全体の正規化
    9. np.nanに0.0を代入
    """
    rank = dist.get_rank()

    if num_score_workers >= 2:
        with cf.ProcessPoolExecutor(num_score_workers) as e:
            futures = [None if errors[idx] is not None else e.submit(get_score, target=target, lig_rdmol=lig, rec_pdb=pdbs[idx], out_dir=f"{out_dir}/{rank}_{idx}", cpu=cpu, print_prepare=print_prepare) for idx, lig in enumerate(ligs)]
            scores = [np.nan if f is None else f.result() for f in futures]
            errors = ['VINA' if error is None and np.isnan(score) else error for error, score in zip(errors, scores)]
    else:
        scores = []
        for idx, lig in enumerate(ligs):
            if errors[idx] is None:
                score = get_score(target, lig, pdbs[idx], f"{out_dir}/{rank}_{idx}", cpu, print_prepare)
                if np.isnan(score):
                    errors[idx] = 'VINA'
                scores.append(score)
            else:
                scores.append(np.nan)
    error_recorder.record(**{str(i): error for i, error in enumerate(errors)})
    scores = torch.tensor(scores, dtype=torch.float, device=device)
    raw_scores = scores.detach().clone()
    is_vina_error = torch.tensor([error == 'VINA' for error in errors], dtype=torch.bool, device=device)
    is_gen_error = torch.tensor([error is not None for error in errors], dtype=torch.bool, device=device) & (~is_vina_error)
    
    if math.isfinite(min_valid_score):
        torch.clamp_(scores, min=min_valid_score) # nan is ignored
    scores[is_gen_error] = gen_error_score
    scores[is_vina_error] = vina_error_score
    sample_mean, sample_std = get_sample_stat(scores, all_idxs) # out of IF for DDP
    if math.isfinite(gen_error_sample_deviation) and torch.any(is_gen_error):
        scores[is_gen_error] = sample_mean[is_gen_error]+sample_std[is_gen_error]*gen_error_sample_deviation

    sample_mean, sample_std = get_sample_stat(scores, all_idxs)
    if math.isfinite(vina_error_sample_deviation) and torch.any(is_vina_error):
        scores[is_vina_error] = sample_mean[is_vina_error]+sample_std[is_vina_error]*vina_error_sample_deviation


    scores = whiten_scores(scores, all_idxs, sample_whiten, all_whiten)
    if math.isfinite(gen_error_white_score):
        scores[is_gen_error] = gen_error_white_score
    if math.isfinite(vina_error_white_score):
        scores[is_vina_error] = vina_error_white_score
    scores = whiten_scores(scores, all_idxs, sample_rewhiten, all_rewhiten)
    scores[torch.isnan(scores)] = 0.0
    score_recorder.record(raw=raw_scores.tolist(), normalized=scores.tolist())
    return scores

def get_velocity(model: ReinforceModel, input: Tensor, position: Tensor, weight: Tensor, scores: Tensor, whiten: list[Literal['mean', 'std']], mbatch_size: int) -> Tensor:
    """
    Parameters
    ----------
    input, position, weight: Tensor(float)[L, B]
    scores: Tensor(float)[B,]

    Returns
    -------
    velocity: Tensor(float)[L, B]
    
    """
    model
    L, B = input.shape
    device = input.device
    velocity = []
    with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for mbatch_start in range(0, B, mbatch_size):
            mslice = slice(mbatch_start, mbatch_start+mbatch_size)
            _, values_m = model(input[:,mslice], position[:,mslice]) # [L,B]
            if model.baseline:
                velocity_m = scores[mslice].unsqueeze(0) - values_m
            else:
                mb = min(B-mbatch_start, mbatch_size)
                velocity_m = scores[mslice].unsqueeze(0).expand(L, mb)
            velocity.append(velocity_m)
    velocity = torch.cat(velocity, dim=1).detach() # [L, B]
    all_adv = reduce_float(torch.sum(velocity*weight), device)
    all_adv2 = reduce_float(torch.sum(velocity**2*weight), device)
    all_weight = reduce_float(torch.sum(weight), device)
    adv_mean = all_adv / all_weight
    adv_std = all_adv2 / all_weight - adv_mean**2
    whiten_mean = 'mean' in whiten
    whiten_std = 'std' in whiten
    if whiten_mean and whiten_std:
        velocity = (velocity-adv_mean)/adv_std
    elif whiten_mean:
        velocity = velocity-adv_mean
    elif whiten_std:
        velocity = (velocity-adv_mean)/adv_std+adv_mean
    return velocity

def scale_loss(exp_loss: Tensor, weight: Tensor, scale: list[Literal['batch_size', 'all_exp', 'episode_exp']], all_exp: int) -> Tensor:
    """
    exp_losses: Tensor[L, B]
    weight: Tensor[L, B]
    
    Returns
    -------
    eposide_losses: [, B]

    """
    _, B = exp_loss.shape
    ns_exp = weight.sum(dim=0)
    world_size = dist.get_world_size()
    exp_loss = exp_loss*weight
    episode_losses = torch.sum(exp_loss*weight, dim=0) # [B, ]
    episode_losses *= world_size # DDPで平均されるのを一旦解消

    if 'batch_size' in scale:
        episode_losses /= world_size*B
    if 'all_exp' in scale:
        episode_losses /= all_exp
    if 'episode_exp' in scale:
        episode_losses /= ns_exp
    return episode_losses

def save_rl_model(model: DistributedDataParallel, optimizer, result_dir, step: int):
    MAIN_RANK, SAVE_RANK, DATA_RANK = get_process_ranks()
    if dist.get_rank() == SAVE_RANK:
        os.makedirs(f"{result_dir}/models", exist_ok=True)
        torch.save(model.module.model.state_dict(), f"{result_dir}/models/{step}.pth")
        if model.module.baseline:
            os.makedirs(f"{result_dir}/value_models", exist_ok=True)
            torch.save(model.module.value_head.state_dict(), f"{result_dir}/value_models/{step}.pth")
        if step > 0:
            cleardir(f"{result_dir}/optimizers")
            torch.save(optimizer.state_dict(), f"{result_dir}/optimizers/{step}.pth")


@record
def main():
    # arguments
    parser = ArgumentParser()
    ## score
    parser.add_argument('--target', choices=['min_vina', 'vina', 'mw_max', 'logp', 'qvina', 'dummy'], required=True)
    parser.add_argument('--max-new-token', type=int, default=1000)
    ### score scale & normalization
    parser.add_argument('--min-valid-score', type=float, default=-math.inf)
    parser.add_argument('--gen-error-score', type=float, default=math.nan)
    parser.add_argument('--vina-error-score', type=float, default=math.nan)
    parser.add_argument('--gen-error-sample-deviation', type=float, default=math.nan)
    parser.add_argument('--vina-error-sample-deviation', type=float, default=math.nan)
    parser.add_argument('--sample-whiten', choices=['mean', 'std'], nargs='*', default=[])
    parser.add_argument('--all-whiten', choices=['mean', 'std'], nargs='*', default=[])
    parser.add_argument('--gen-error-white-score', type=float, default=math.nan)
    parser.add_argument('--vina-error-white-score', type=float, default=math.nan)
    parser.add_argument('--sample-rewhiten', choices=['mean', 'std'], nargs='*', default=[])
    parser.add_argument('--all-rewhiten', choices=['mean', 'std'], nargs='*', default=[])
    parser.add_argument('--adv-whiten', choices=['mean', 'std'], nargs='*', default=[])
    parser.add_argument('--no-baseline', action='store_false', dest='baseline')
    ## Data
    parser.add_argument('--max-prompt-len', type=int)
    parser.add_argument('--generate-per-sample', type=int, default=1)
    parser.add_argument('--train-sample', type=float, default=1.0)
    parser.add_argument('--valid-sample', type=float, default=1.0)
    ## training
    parser.add_argument('--studyname', required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-opt', type=int, default=10000)
    ## optimizer
    parser.add_argument('--weight-decay', type=float, default=0.0) # same as BindGPT
    parser.add_argument('--clip-grad-value', type=float, default=None)
    parser.add_argument('--clip-grad-norm', type=float, default=1.0) # same as BindGPT
    parser.add_argument('--loss-scale', choices=['batch_size', 'all_exp', 'episode_exp'], nargs='*', default=[])
    parser.add_argument('--kl-factor', type=float, default=0.05) # --alpha in old code. same as BindGPT
    parser.add_argument('--value-factor', type=float, default=0.05)
    ## lr/scheduler
    parser.add_argument('--lr', type=float, default=1.4e-5) # same as BindGPT
    parser.add_argument('--scheduler', default='constant') # same as BindGPT
    parser.add_argument("--schedule-free", action='store_true')
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    ## finetune
    parser.add_argument('--finetune-name', required=True)
    parser.add_argument('--finetune-opt', type=int, default=None)
    parser.add_argument('--finetune-patience-val', type=int)
    ## environment
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--num-score-workers', type=int, default=1)
    parser.add_argument('--gc', action='store_true')
    parser.add_argument('--gpu-size-gb', type=float)
    parser.add_argument('--mbatch-size', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--commit', action='store_false', dest='no_commit')
    ## verbosity
    parser.add_argument('--tqdm-generate', action='store_true')
    parser.add_argument('--record-opt', type=int)
    ## test
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument("--deterministic", action='store_true')
    parser.add_argument('--use-categorical', action='store_true')
    parser.add_argument('--all-autocast', action='store_true')
    parser.add_argument('--fix-pocket', action='store_true')
    parser.add_argument("--weight-decay-all", action='store_true')
    parser.add_argument('--check', nargs='*', default=[], choices=['data_dist', 'optimizer', 'tokens', 'gpu'])
    args = parser.parse_args()
    ## set default args
    if args.test: args.studyname+='_test'
    if args.record_opt is None:
        args.record_opt = 1 if args.test else 500
    if args.mbatch_size is None:
        assert args.gpu_size_gb is not None
        args.gpu_size = args.gpu_size_gb * (2**30)
    
    args.sdp_kernel = 'FLASH'

    logs = []

    # get finetune info
    finetune_dir = f"finetune/results/{args.finetune_name}"
    with open(f"{finetune_dir}/args.yaml") as f:
        fargs = yaml.safe_load(f)
    fargs = Namespace(**fargs)
    pname = fargs.pretrain_name
    pretrain_dir = f"training/results/{pname}"
    with open(f"{pretrain_dir}/args.yaml") as f:
        pargs = Namespace(**yaml.safe_load(f))

    ## check finetuning
    assert fargs.no_score

    ## set finetune opt
    if args.finetune_opt is None:
        if args.finetune_patience_val is not None: # set from early stopping
            logs.append(f"finetune_opt was set from patience_val={args.finetune_patience_val}")
            dfval = pd.read_csv(f"{finetune_dir}/vals/0.csv")
            losses = dfval['mean_loss'].values
            for val in range(len(losses)-args.finetune_patience_val):
                if losses[val] <= losses[val:val+args.finetune_patience_val+1].min():
                    break
            else:
                raise ValueError(f"Early stopping not finished.")
            args.finetune_opt = int(dfval['opt'].values[val])
        else: # set from max_opt
            args.finetune_opt = fargs.max_opt
        logs.append(f"finetune_opt was set to {args.finetune_opt}")
    else:
        assert args.finetune_patience_val is None

    do_save_steps = [0, 1, 100, 200, 400, 700]+list(range(1000, args.max_opt, 1000))

    
    # Environment
    result_dir = f"reinforce/results/{args.studyname}"
    logger, _, rank, device = set_env(result_dir, args, logs, 
            subdirs=['grads/reward', 'grads/kl', 'grads/value'], get_token_logger=False)
    ignore_rdkit_warning()
    ## check generate_per_sample
    logger.info(f"git hash={get_git_hash()}", **NO_DUP)
    MAIN_RANK, SAVE_RANK, DATA_RANK = get_process_ranks()

    # model
    init_state_path = f"{finetune_dir}/models/{args.finetune_opt}.pth"
    init_model, voc_encoder = get_model(pargs, None, init_state_path, device)
    net_model = get_model(pargs, voc_encoder, init_state_path, device)
    reinforce_model = ReinforceModel(net_model, args.baseline)
    reinforce_model.to(device)
    model = DistributedDataParallel(reinforce_model)
    if args.max_prompt_len is not None and args.mbatch_size is not None:
        logger.info(f"Estimated GPU use={net_model.get_gpuuse(args.mbatch_size, args.max_new_token+args.max_prompt_len, True, 'FLASH')/2**30:.03f}")

    # data
    ## vocs from state_dict
    _voc_encoder, raw_data, protein_data, _lig, token_data, position_data, weight_data, center_data, data_log \
            = get_finetune_data(fargs, 'train', 1.0, False, True, set(voc_encoder.i2voc[1:]), 'none')
    protein_pdb_data = Protein2PDBDataset(protein_data)
    logs += data_log
    index_data, token_data = index_dataset(token_data)
    train_data = StackDataset(index_data, protein_pdb_data, token_data, position_data)

    # DataLoader
    data_iter = ReinforceDataIter(train_data, device, args.batch_size, args.generate_per_sample, args.max_prompt_len, args.fix_pocket, args.num_workers, args.seed)

    # optimizer
    optimizer, scheduler = get_optimizer_scheduler(model, args.max_opt, args.weight_decay_all, args.weight_decay, args.schedule_free, args.scheduler, args.lr, args.warmup_ratio, 'optimizer' in args.check)

    ## records
    step_recorder = IterateRecorder(f"{result_dir}/steps/{rank}.csv", args.record_opt)
    error_recorder = IterateRecorder(f"{result_dir}/errors/{rank}.csv", args.record_opt)
    score_recorder = IterateRecorder(f"{result_dir}/scores/{rank}.csv", args.record_opt)
    size_recorder = IterateRecorder(f"{result_dir}/sizes/{rank}.csv", args.record_opt)
    train_looper = Loopers([
        LogLooper(logger, 1000, 5, 'step', 'training'),
        TimeWriteLooper(f"{result_dir}/steps/times/{rank}.csv", 1000),
        TimeLogLooper(logger, 'step', 1000), 
        GPUUseLooper(logger, device, 'step', 5)
    ])
    if 'gpu' in args.check:
        train_looper.append(MemorySnapshotLooper(f"{result_dir}/memory_snapshot.pkl", 1, dump_process=True))

    # save at step 0
    save_rl_model(model, optimizer, result_dir, 0)

    train_looper.start_loops()
    for step in range(args.max_opt):
        is_starting = step < 3
        do_save = step in do_save_steps

        train_looper.put('get_batch')
        all_idxs, pdbs, prompt_tokens, positions = data_iter.get()

        train_looper.put('generate')
        input, output, position, weight, all_exp, ligs, errors = generate(
            model, 
            voc_encoder,
            result_dir,
            args.max_new_token,
            fargs.coord_range,
            not fargs.no_lig_h_atom,
            not fargs.no_lig_h_coord,
            device,
            size_recorder,
            step, 
            prompt_tokens, 
            positions, 
            do_save,
        )

        train_looper.put('get_score')
        scores = get_scores(
            pdbs,
            ligs,
            errors,
            all_idxs,

            f"{result_dir}/generation/{step if do_save else 'tmp'}",
            error_recorder,
            score_recorder,
            device,
            args.min_valid_score,
            args.gen_error_score,
            args.vina_error_score,
            args.gen_error_sample_deviation,
            args.vina_error_sample_deviation,
            args.sample_whiten,
            args.all_whiten,
            args.gen_error_white_score, 
            args.vina_error_white_score,
            args.sample_rewhiten,
            args.all_rewhiten,

            args.target,
            args.num_score_workers,
            args.cpu,
            is_starting
        )
        if is_starting:
            logger.info(f"step {step} scores={scores.cpu().tolist()}")
        
        # get & whiten velocity
        L, B = input.shape
        mbatch_size = args.mbatch_size if args.mbatch_size is not None else \
            solve_increasing_fn_left(lambda bsz: 
            net_model.get_gpuuse(bsz, L, True, 'FLASH')-args.gpu_size, 16)
        if mbatch_size == 0:
            raise ValueError(f"Input was too large")
        train_looper.put('velocity')
        velocity = get_velocity(model.module, input, position, weight, scores, args.adv_whiten, mbatch_size)

        # Forward Get prob & reward loss
        train_looper.put('forward_backward')
        model.train()
        optimizer.zero_grad()
        save_grad = step in [0, 100]
        with model.join():
            reward_loss = 0.0
            kl_loss = 0.0
            value_loss = 0.0
            if do_save:
                values = []
                log_probs = []
            if save_grad:
                term2grads = {term: defaultdict(float) for term in ['reward', 'kl', 'value']}
            for mbatch_start in range(0, B, mbatch_size):
                mslice = slice(mbatch_start, mbatch_start+mbatch_size)
                input_m = input[:, mslice]
                output_m = output[:, mslice]
                position_m = position[:,mslice]
                weight_m = weight[:, mslice]
                rewards_m = scores[mslice]
                with torch.autocast('cuda', torch.bfloat16), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    with torch.inference_mode():
                        init_logits = init_model(input_m, position_m) # [L, B, N]
                        init_log_probs_all = F.log_softmax(init_logits, dim=-1).detach() # [L, B, N]
                    logits_m, values_m = model(input_m, position_m) # [L, B, T], [L, B]
                    log_probs_all = F.log_softmax(logits_m, dim=-1) # [L, B, N]
                    log_probs_m = torch.gather(log_probs_all, dim=-1, index=output_m.unsqueeze(-1)).squeeze(-1) # [L, B]
                    reward_loss_m = -velocity[:, mslice]*log_probs_m
                    reward_loss_m = scale_loss(reward_loss_m, weight_m, args.loss_scale, all_exp)

                    ## KL loss
                    log_probs_all = F.log_softmax(logits_m, dim=-1)
                    kl_loss_m = F.kl_div(input=log_probs_all, target=init_log_probs_all, reduction='none', 
                        log_target=True) # [Lo-1, B, N]
                    kl_loss_m = kl_loss_m.sum(dim=-1) # [Lo-1, B]
                    kl_loss_m = scale_loss(kl_loss_m, weight_m, args.loss_scale, all_exp) * args.kl_factor

                    if args.baseline:
                        value_loss_m = (rewards_m-values_m)**2*0.5
                        value_loss_m = scale_loss(value_loss_m, weight_m, args.loss_scale, all_exp) * args.value_factor
                    else:
                        value_loss_m = torch.zeros_like(velocity, requires_grad=True) # [L, B]
                        values_m = torch.zeros_like(velocity) # [L, B]
                                        
                if save_grad:
                    train_looper.put('save_grad')
                    terms = {'reward': reward_loss_m.sum(), 'kl': kl_loss_m.sum(), 'value': value_loss_m.sum()}
                    for term_name, term in terms.items():
                        grads = torch.autograd.grad(term, model.parameters(), retain_graph=True, allow_unused=True)
                        for (param_name, _), grad in zip(model.named_parameters(), grads):
                            if grad is not None:
                                term2grads[term_name][param_name] += grad
                    train_looper.put('forward_backward')
                loss = reward_loss_m.sum()+kl_loss_m.sum()+value_loss_m.sum()
                loss.backward()
                reward_loss += reward_loss_m.sum().item()
                kl_loss += kl_loss_m.sum().item()
                value_loss += value_loss_m.sum().item()
                if do_save:
                    log_probs.append(log_probs_m.detach())
                    values.append(values_m.detach())
        if do_save:
            log_probs10 = torch.cat(log_probs, dim=1) / math.log(10)
            values = torch.cat(values, dim=1)
            os.makedirs(f"{result_dir}/batches/{step}", exist_ok=True)
            for b in (range(B) if step in [0, 100] else [0]):
                pd.DataFrame(dict(
                    input=[voc_encoder.i2voc[i] for i in input[:,b]], 
                    output=[voc_encoder.i2voc[i] for i in output[:,b]], 
                    position=position[:,b].float().cpu(),
                    weight=weight[:,b].float().cpu(),
                    log_prob=log_probs10[:,b].float().cpu(),
                    value=values[:,b].float().cpu(), 
                    velocity=velocity[:,b].float().cpu())
                ).to_csv(f"{result_dir}/batches/{step}/{rank}_{b}.csv")
        if save_grad:
            for term, grads in term2grads.items():
                for param_name, grad in grads.items():
                    dist.reduce(grad, dst=SAVE_RANK, op=dist.ReduceOp.AVG)
            if rank == SAVE_RANK:
                for term, grads in term2grads.items():
                    torch.save(dict(grads), f"{result_dir}/grads/{term}/step{step}.pth")

        # Optimizer's step
        train_looper.put('optim')
        if args.clip_grad_value is not None:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_value)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()
        step += 1
        step_recorder.record(lr=scheduler.get_last_lr()[0], memory=psutil.virtual_memory().used/(2**30), 
                reward_loss=reward_loss, kl_loss=kl_loss, value_loss=value_loss)
        scheduler.step()
        
        if step % args.record_opt == 0:
            save_rl_model(model, optimizer, result_dir, step)

        if step == 3:
            logger.info("RDKit logger will be disabled from now on.")
            getLogger('rdkit').propagate = False

        train_looper.end_loop()
    train_looper.end_loops()
    step_recorder.flush()
    error_recorder.flush()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
