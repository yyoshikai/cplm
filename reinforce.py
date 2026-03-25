import os, yaml, psutil, math, random, copy
import itertools as itr
import concurrent.futures as cf
from argparse import ArgumentParser, Namespace
from logging import getLogger
from typing import Literal
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
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
from src.utils.rdkit import ignore_rdkit_warning
from src.utils.logger import NO_DUP
from src.chem import rdmol2obmol, pdb2obmol
from src.evaluate import eval_vina, eval_qvina
from src.data.protein import Protein2PDBDataset, AtomRepr
from src.data.tokenizer import VocEncoder
from src.model import Model
from src.train import set_env, get_model, get_process_ranks
from src.train.data import get_finetune_data
from src.train.looper import Loopers, TimeWriteLooper, LogLooper, TimeLogLooper, GPUUseLooper, MemorySnapshotLooper
from src.generate.streamer import WrapperStreamer, LigandStreamer, TokenSaveStreamer, PositionSaveStreamer, TimeLogStreamer, TqdmStreamer
from src.train.reinforce import ReinforceTrainer, DPOTrainer, GRPOTrainer, SaveBatchTrainer, SaveStepTrainer, GetMemoryTrainer, get_sample_stat, whiten_scores
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

class GetScoreStreamer(WrapperStreamer):
    def __init__(self, streamer: LigandStreamer, e: cf.ProcessPoolExecutor, target, rec_pdb: str, out_dir: str, cpu: int, print_prepare: bool):
        super().__init__(streamer)
        self.e = e
        self.kwargs = dict(target=target, rec_pdb=rec_pdb, out_dir=out_dir, cpu=cpu, print_prepare=print_prepare)
        self.future = None
        self.out = np.nan
    def put(self, tokens):
        is_remain, position, token_range = self.streamer.put(tokens)
        if not is_remain and self.streamer.error is None:
            self.future = self.e.submit(get_score, lig_rdmol=self.streamer.mol, **self.kwargs)
        return is_remain, position, token_range
    def result(self):
        if self.future is not None:
            self.out = self.future.result()
            if np.isnan(self.out):
                self.streamer.error = 'VINA'


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
                item = self.train_fixed_item if self.fix_pocket else self.train_iter.__next__()
                all_items += [item] * self.generate_per_sample
            batched_items = [all_items[r*self.batch_size:(r+1)*self.batch_size] for r in range(self.ddp_size)]
        else:
            batched_items = None
        items_box = [None]
        dist.scatter_object_list(items_box, batched_items, src=self.data_rank)
        idxs, pdbs, prompt_tokens, positions = zip(*items_box[0])
        prompt_tokens = [token.to(self.device) for token in prompt_tokens]
        positions = [position.tolist() for position in positions]
        idxs = torch.tensor(idxs, dtype=torch.long, device=self.device)
        return idxs, pdbs, prompt_tokens, positions

def generate(
        model: Model, 
        voc_encoder: VocEncoder,
        result_dir: str, 
        max_new_token: int,
        coord_range: int,
        lig_h: AtomRepr,
        device: torch.device,
        size_recorder: IterateRecorder,

        step: int,
        prompt_tokens,
        positions,
        do_save: bool,

        num_score_workers: int,
        target: str,
        cpu: int,
        pdbs: list[str],
        error_recorder: IterateRecorder,
) -> tuple[Tensor, Tensor, Tensor, Tensor, list[float], list[str|None]]:
    rank = dist.get_rank()

    model.eval()
    ligand_streamers = [LigandStreamer(f"{result_dir}/generation/{step}/{rank}_{idx}/new_sdf.sdf" if do_save else None, coord_range, voc_encoder, False, lig_h, None) for idx in range(len(prompt_tokens))]
    with cf.ProcessPoolExecutor(num_score_workers) as e:
        score_streamers = [
            GetScoreStreamer(streamer, e, target, pdb, f"{result_dir}/generation/{step if do_save else 'tmp'}/{rank}_{idx}", cpu=cpu, print_prepare=step < 3) for idx, (streamer, pdb) in enumerate(zip(ligand_streamers, pdbs))
        ]
        token_streamers = [TokenSaveStreamer(streamer) for streamer in score_streamers]
        streamers = position_streamers = [PositionSaveStreamer(streamer) for streamer in token_streamers]
        if step < 5:
            streamers = [TimeLogStreamer(streamer, str(b), 10.0) for b, streamer in enumerate(streamers)]
            # streamers[0] = TqdmStreamer(streamers[0], total=max_new_token, desc="generate")
        with torch.inference_mode(), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            model.generate2(prompt_tokens, positions, streamers, max_new_token)
        for streamer in score_streamers:
            streamer.result()

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
    position = pad_sequence([
        torch.tensor(position+position_streamer.new_positions) for position, position_streamer
        in zip(positions, position_streamers)
    ], padding_value=0).to(device=device, dtype=torch.long)[:-2]
    size_recorder.record(prompt=[len(s.prompt_tokens) for s in token_streamers], output=[len(s.new_tokens) for s in token_streamers])
    
    errors = [streamer.error for streamer in ligand_streamers]
    scores = [streamer.out for streamer in score_streamers]
    error_recorder.record(**{str(i): error for i, error in enumerate(errors)})
    return input, output, position, weight, scores, errors

def get_scores(
        errors: list[str|None],
        idxs: Tensor,
        scores: list[float],

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

    scores = torch.tensor(scores, dtype=torch.float, device=device)
    raw_scores = scores.detach().clone()
    is_vina_error = torch.tensor([error == 'VINA' for error in errors], dtype=torch.bool, device=device)
    is_gen_error = torch.tensor([error is not None for error in errors], dtype=torch.bool, device=device) & (~is_vina_error)
    
    if math.isfinite(min_valid_score):
        torch.clamp_(scores, min=min_valid_score) # nan is ignored
    scores[is_gen_error] = gen_error_score
    scores[is_vina_error] = vina_error_score
    sample_mean, sample_std = get_sample_stat(scores, idxs) # out of IF for DDP
    if math.isfinite(gen_error_sample_deviation) and torch.any(is_gen_error):
        scores[is_gen_error] = sample_mean[is_gen_error]+sample_std[is_gen_error]*gen_error_sample_deviation

    sample_mean, sample_std = get_sample_stat(scores, idxs)
    if math.isfinite(vina_error_sample_deviation) and torch.any(is_vina_error):
        scores[is_vina_error] = sample_mean[is_vina_error]+sample_std[is_vina_error]*vina_error_sample_deviation


    scores = whiten_scores(scores, idxs, sample_whiten, all_whiten)
    if math.isfinite(gen_error_white_score):
        scores[is_gen_error] = gen_error_white_score
    if math.isfinite(vina_error_white_score):
        scores[is_vina_error] = vina_error_white_score
    scores = whiten_scores(scores, idxs, sample_rewhiten, all_rewhiten)
    scores[torch.isnan(scores)] = 0.0
    score_recorder.record(raw=raw_scores.tolist(), normalized=scores.tolist())
    return scores

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
    parser.add_argument('--adv-all-whiten', choices=['mean', 'std'], nargs='*', default=[])
    parser.add_argument('--adv-sample-whiten', choices=['mean', 'std'], nargs='*', default=[])
    parser.add_argument('--trainer', choices=['reinforce', 'reinforce_no_baseline', 'dpo', 'grpo', 'dapo'], required=True)
    ## Trainer
    ### PPO
    parser.add_argument('--n-ppo-step', type=int, default=5)
    parser.add_argument('--ppo-clip-eps', type=float, default=0.2)
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
    parser.add_argument('--record-opt', type=int, default=10)
    parser.add_argument('--save-opt', type=int)
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
    if args.save_opt is None:
        args.save_opt = 1 if args.test else 500
    if args.mbatch_size is None:
        assert args.gpu_size_gb is not None
        args.gpu_size = args.gpu_size_gb * (2**30)
    else:
        assert args.gpu_size_gb is None
        args.gpu_size = None
    
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
    logger, rank, device = set_env(result_dir, args, logs, subdirs=['grads/reward', 'grads/kl', 'grads/value'])
    ignore_rdkit_warning()
    ## check generate_per_sample
    logger.info(f"git hash={get_git_hash()}", **NO_DUP)

    # model
    init_state_path = f"{finetune_dir}/models/{args.finetune_opt}.pth"
    model_, voc_encoder = get_model(pargs, None, init_state_path, device)

    # data
    ## vocs from state_dict
    _voc_encoder, _raw_data, protein_data, _lig, token_data, position_data, _weight_data, _center_data, data_log \
            = get_finetune_data(fargs, 'train', 1.0, False, True, set(voc_encoder.i2voc[1:]), 'none')
    protein_pdb_data = Protein2PDBDataset(protein_data)
    logs += data_log
    index_data, token_data = index_dataset(token_data)
    train_data = StackDataset(index_data, protein_pdb_data, token_data, position_data)

    # DataLoader
    data_iter = ReinforceDataIter(train_data, device, args.batch_size, args.generate_per_sample, args.max_prompt_len, args.fix_pocket, args.num_workers, args.seed)

    ## records
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
    log_optimizer = 'optimizer' in args.check
    if args.trainer in ['reinforce', 'reinforce_no_baseline']:
        baseline = args.trainer == 'reinforce'
        trainer = ReinforceTrainer(model_, baseline, args.max_opt, args.weight_decay_all, args.weight_decay, args.schedule_free, args.scheduler, args.lr, args.warmup_ratio, log_optimizer, args.mbatch_size, args.gpu_size, args.adv_sample_whiten, args.adv_all_whiten, args.loss_scale, args.kl_factor, args.value_factor, train_looper, args.clip_grad_value, args.clip_grad_norm, result_dir)
    elif args.trainer == 'dpo':
        trainer = DPOTrainer(model_, args.max_opt, args.weight_decay_all, args.weight_decay, args.schedule_free, args.scheduler, args.lr, args.warmup_ratio, log_optimizer, args.mbatch_size, args.gpu_size, args.kl_factor, train_looper, args.clip_grad_value, args.clip_grad_norm)
    elif args.trainer == 'grpo':
        trainer = GRPOTrainer(model_, args.max_opt, args.weight_decay_all, args.weight_decay, args.schedule_free, args.scheduler, args.lr, args.warmup_ratio, log_optimizer, args.mbatch_size, args.gpu_size, args.kl_factor, args.clip_grad_value, args.clip_grad_norm, args.n_ppo_step, args.ppo_clip_eps)
    trainer = GetMemoryTrainer(trainer, device)
    trainer = SaveBatchTrainer(trainer, result_dir, do_save_steps, voc_encoder)
    trainer = SaveStepTrainer(trainer, result_dir, args.record_opt, args.max_opt)
    trainer.save(result_dir)

    train_looper.start_loops()
    for step in range(args.max_opt):
        is_starting = step < 3
        do_save = step in do_save_steps

        train_looper.put('get_batch')
        idxs, pdbs, prompt_tokens, positions = data_iter.get()

        train_looper.put('generate')
        input, output, position, weight, scores, errors = generate(
            trainer.policy_model(), 
            voc_encoder,
            result_dir,
            args.max_new_token,
            fargs.coord_range,
            fargs.lig_h,
            device,
            size_recorder,
            step, 
            prompt_tokens, 
            positions, 
            do_save,
            args.num_score_workers,
            args.target,
            args.cpu, 
            pdbs,
            error_recorder,
        )
        if is_starting:
            logger.info(f"step {step} raw scores={scores}")

        train_looper.put('get_score')
        scores = get_scores(
            errors,
            idxs,
            scores,

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

        )

        # Forward Get prob & reward loss
        train_looper.put('train')
        trainer.train(input, output, position, weight, scores, errors, idxs)
        
        if step % args.save_opt == 0:
            train_looper.put('save')
            trainer.save(result_dir)

        if step == 3:
            logger.info("RDKit logger will be disabled from now on.")
            getLogger('rdkit').propagate = False

        train_looper.end_loop()
    train_looper.end_loops()
    error_recorder.flush()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
