import os, yaml, psutil, gc, math, random
import itertools as itr
import concurrent.futures as cf
from argparse import ArgumentParser, Namespace
from logging import getLogger
import numpy as np, pandas as pd
from contextlib import nullcontext
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from rdkit.Chem import rdMolDescriptors
from torch.utils.data import DataLoader, StackDataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.distributions import Categorical
from torch.distributed.elastic.multiprocessing.errors import record
from rdkit import Chem
from src.data._sampler import InfiniteRandomSampler
from src.data import index_dataset
from src.utils import IterateRecorder, get_git_hash
from src.utils.path import cleardir, make_pardir
from src.utils.time import TimerTqdm
from src.utils.rdkit import ignore_rdkit_warning
from src.utils.ddp import dist_all_gather
from src.evaluate import rdmol2obmol, pdb2obmol, eval_vina, eval_qvina
from src.data.protein import Protein2PDBDataset
from src.train import set_env, get_model, get_optimizer_scheduler, get_process_ranks
from src.model import Model
from src.finetune import get_finetune_data
from src.data.collator import solve_increasing_fn_left
from src.generate.streamer import LigandStreamer, TokenWriteStreamer, TokenSaveStreamer, PositionSaveStreamer, TimeLogStreamer
WORKDIR = os.environ.get('WORKDIR', os.path.abspath('..'))

## Scoring function 最大化したいものとする

def get_score(target, lig_rdmol: Chem.Mol, rec_pdb: str, out_dir: str, cpu: int):
    match target:
        case 'min_vina':
            score, min_score = eval_vina(rdmol2obmol(lig_rdmol), pdb2obmol(rec_pdb), f"{out_dir}/protein_h.pdbqt")
            return -min_score if min_score is not None else np.nan
        case 'vina':
            score, min_score = eval_vina(rdmol2obmol(lig_rdmol), pdb2obmol(rec_pdb), f"{out_dir}/protein_h.pdbqt")
            return -score if min_score is not None else np.nan
        case 'mw_max':
            return rdMolDescriptors.CalcExactMolWt(lig_rdmol)
        case 'qvina':
            with open(f"{out_dir}/rec_input.pdb", 'w') as f:
                f.write(rec_pdb)
            score = eval_qvina(lig_rdmol, f"{out_dir}/rec_input.pdb", out_dir, timeout=60, cpu=cpu)[0]
            return -score if score is not None else np.nan
        case 'dummy':
            return random.random()

@record
def main():
    # arguments
    parser = ArgumentParser()
    ## reward
    parser.add_argument('--target', choices=['min_vina', 'vina', 'mw_max', 'logp', 'qvina', 'dummy'], required=True)
    parser.add_argument('--error-score', type=float, default=None)
    parser.add_argument('--ignore-error', action='store_true')
    parser.add_argument('--min-score', type=float, default=-math.inf)
    parser.add_argument('--max-new-token', type=int, default=1000)
    parser.add_argument('--reward-scale', choices=['none', 'all_mean', 'sample_mean', 'rank_mean', 'rank_mean_std'], default='none')
    parser.add_argument('--alpha', type=float, default=0.05) # same as BindGPT
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
    parser.add_argument('--loss-scale')
    ## scheduler
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
    parser.add_argument('--prefetch-factor', type=int)
    parser.add_argument('--num-score-workers', type=int, default=1)
    parser.add_argument('--reset-nan-grad', action='store_true')
    parser.add_argument('--gc', action='store_true')
    parser.add_argument('--gpu-size-gb', type=float, required=True)
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
    parser.add_argument('--check', nargs='*', default=[], choices=['data_dist', 'optimizer'])
    args = parser.parse_args()
    ## set default args
    if args.test: args.studyname+='_test'
    if args.record_opt is None:
        args.record_opt = 1 if args.test else 500
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

    log_sample_step = 3
    do_save_steps = [0, 1, 2, 3, 4, 50, 100]+list(range(200, 1000, 200)) \
            +list(range(1000, args.max_opt, 1000))

    
    # Environment
    result_dir = f"reinforce/results/{args.studyname}"
    logger, token_logger, rank, device = set_env(result_dir, args, logs, 
            subdirs=['models'])
    MAIN_RANK, SAVE_RANK, DATA_RANK = get_process_ranks()
    ignore_rdkit_warning()
    ## check generate_per_sample
    ddp_size = dist.get_world_size()
    if rank == MAIN_RANK:
        logger.info(f"git hash={get_git_hash()}")

    # model
    init_state_path = f"{finetune_dir}/models/{args.finetune_opt}.pth"
    init_model, voc_encoder = get_model(pargs, None, init_state_path, device)
    net_model = get_model(pargs, voc_encoder, init_state_path, device)
    if args.all_autocast:
        init_model.to(torch.bfloat16)
        net_model.to(torch.bfloat16)
    model = DistributedDataParallel(net_model)
    def get_gpuuse(batch_size: int, length: int):
        if isinstance(model.module, Model):
            return model.module.get_gpuuse(batch_size, length, True, 'FLASH')
        else:
            return model.module.get_gpuuse(batch_size, length, True)
    if args.max_prompt_len is not None:
        logger.info(f"Estimated GPU use={get_gpuuse(args.batch_size, args.max_new_token+args.max_prompt_len)/2**30:.03f}")

    # data
    ## vocs from state_dict
    _voc_encoder, raw_data, protein_data, token_data, position_data, weight_data, center_data, data_log \
            = get_finetune_data(fargs, 'train', False, True, set(voc_encoder.i2voc[1:]), 'none')
    protein_pdb_data = Protein2PDBDataset(protein_data)
    logs += data_log
    index_data, token_data = index_dataset(token_data)
    train_data = StackDataset(index_data, protein_pdb_data, token_data, position_data)

    ## Scoring function 最大化したいものとする
    error_score = { 'min_vina': -50.0, 'vina': -50.0, 'mw_max': 0.0, 'qvina': -50.0, 'dummy': 0.0}[args.target]
    if args.error_score is not None:
        error_score = args.error_score

    # DataLoader
    assert args.batch_size * ddp_size % args.generate_per_sample == 0
    if rank == DATA_RANK['train']:
        loader = DataLoader(train_data, batch_size=None, 
            sampler=InfiniteRandomSampler(train_data, generator=torch.Generator().manual_seed(args.seed)),
            num_workers=args.num_workers, pin_memory=True, prefetch_factor=args.prefetch_factor)
        train_iter = loader.__iter__()
        if args.max_prompt_len is not None:
            train_iter = itr.filterfalse(lambda x: len(x[2]) > args.max_prompt_len, train_iter)
        if args.fix_pocket:
            train_fixed_item = train_iter.__next__()
            del train_iter

    # optimizer
    optimizer, scheduler = get_optimizer_scheduler(model, args.max_opt, args.weight_decay_all, args.weight_decay, args.schedule_free, args.scheduler, args.lr, args.warmup_ratio, 'optimizer' in args.check)
    optimizer.zero_grad()
    match args.loss_scale:
        case None:
            loss_scale = 1.0
        case _:
            loss_scale = float(args.loss_scale)

    ## records
    step_recorder = IterateRecorder(f"{result_dir}/steps/{rank}.csv", args.record_opt)
    error_recorder = IterateRecorder(f"{result_dir}/errors/{rank}.csv", args.record_opt)
    score_recorder = IterateRecorder(f"{result_dir}/scores/{rank}.csv", args.record_opt)
    size_recorder = IterateRecorder(f"{result_dir}/size/{rank}.csv",  args.record_opt)
    step_timer = TimerTqdm(range(args.max_opt), time_path=f"{result_dir}/steps/times/{rank}.csv", file_interval=100, log_interval=100, disable_bar=True)
    nan_grad_step_saved = False

    # save at step 0
    if rank == SAVE_RANK:
        torch.save(model.state_dict(), f"{result_dir}/models/0.pth")
        cleardir(f"{result_dir}/optimizers")
        torch.save(optimizer.state_dict(), f"{result_dir}/optimizers/0.pth")

    logger.info("Training started.")
    for step in step_timer:
        is_starting = step < 5
        do_save = step in do_save_steps

        # get batch
        step_timer.start('get_batch')
        if is_starting:
            logger.debug("get_batch started.")
        if rank == DATA_RANK['train']:
            all_items = []
            for si in range(args.batch_size*ddp_size // args.generate_per_sample):
                all_items += [train_fixed_item if args.fix_pocket else train_iter.__next__()] * args.generate_per_sample
            batched_items = [all_items[r*args.batch_size:(r+1)*args.batch_size] for r in range(ddp_size)]
        else:
            batched_items = None
        items_box = [None]
        if is_starting:
            logger.debug("scatter_object_list started.")
        dist.scatter_object_list(items_box, batched_items, src=DATA_RANK['train'])
        if is_starting:
            logger.debug("scatter_object_list ended.")
        idxs, pdbs, prompt_tokens, positions = zip(*items_box[0])
        all_idxs = torch.cat(dist_all_gather(torch.tensor(idxs, dtype=torch.int, device=device)))
        prompt_tokens = [token.to(device) for token in prompt_tokens]
        positions = [position.tolist() for position in positions]
        prompt_sizes = [len(token) for token in prompt_tokens]
        B = len(prompt_tokens)

        step_timer.start('generate')
        if is_starting: logger.debug(f"step[{step}] generate started.")
        with torch.autocast('cuda', torch.bfloat16) if args.all_autocast else nullcontext():
            # generate sample
            model.eval()
            streamers = ligand_streamers = [LigandStreamer(f"{result_dir}/generation/{step}/{rank}_{idx}/new_sdf.sdf" if do_save else None, fargs.coord_range, voc_encoder, False, not fargs.no_lig_h_atom, not fargs.no_lig_h_coord, None) for idx in range(B)]
            streamers = token_streamers = [TokenSaveStreamer(streamer) for streamer in streamers]
            streamers = position_streamers = [PositionSaveStreamer(streamer) for streamer in streamers]
            if False:
                streamers = [TokenWriteStreamer(streamer, 
                        prompt_token_path=f"{result_dir}/generation/{step}/{rank}_{idx}/prompt_token.txt", 
                        new_token_path=f"{result_dir}/generation/{step}/{rank}_{idx}/new_token.txt", 
                        voc_encoder=voc_encoder
                    ) for idx, streamer in enumerate(streamers)]
            if is_starting:
                streamers = [TimeLogStreamer(streamer, str(b), 5.0) for b, streamer in enumerate(streamers)]
            with torch.inference_mode():
                net_model.generate2(prompt_tokens, positions, streamers, args.max_new_token)

            out_batch = pad_sequence([
                torch.tensor(token_streamer.prompt_tokens+token_streamer.new_tokens)
                for token_streamer in token_streamers
            ], padding_value=voc_encoder.pad_token).to(device=device, dtype=torch.long)
            weight = torch.zeros_like(out_batch, dtype=torch.float)[:-1] # [Lo-1, B]
            sizes = {}
            for b, token_streamer in enumerate(token_streamers):
                prompt_size, new_size = len(token_streamer.prompt_tokens), len(token_streamer.new_tokens)
                weight[prompt_size-1:prompt_size+new_size-1, b] = 1.0
                sizes[f'prompt {b}'] = prompt_size; sizes[f'output {b}'] = new_size
            position_batch = pad_sequence([
                torch.tensor(position+position_streamer.new_positions) for position, position_streamer
                in zip(positions, position_streamers)
            ], padding_value=0).to(device=device, dtype=torch.long)
            size_recorder.record(**sizes)
                    
            # Get score
            step_timer.start('get_score')
            errors = []
            with cf.ProcessPoolExecutor(args.num_score_workers) if (args.num_score_workers >= 2) else nullcontext() as e:
                futures = []
                valid_scores = []
                for idx, ligand_streamer in enumerate(ligand_streamers):

                    lig_mol = ligand_streamer.mol
                    errors.append('' if ligand_streamer.error is None else ligand_streamer.error)
                    if ligand_streamer.error is not None: 
                        continue
                    out_dir = f"{result_dir}/generation/{step if do_save else 'tmp'}/{rank}_{idx}"
                    os.makedirs(out_dir, exist_ok=True)
                    if args.num_score_workers >= 2:
                        futures.append(e.submit(get_score, target=args.target, lig_rdmol=lig_mol, rec_pdb=pdbs[idx], out_dir=out_dir, cpu=args.cpu))
                    else:
                        valid_scores.append(get_score(args.target, lig_mol, pdbs[idx], out_dir, args.cpu))
                if args.num_score_workers >= 2:
                    valid_scores = np.array([f.result() for f in futures])
                else:
                    valid_scores = np.array(valid_scores)
            errors = np.array(errors)
            scores = np.full(len(errors), np.nan)
            scores[errors == ""] = valid_scores
            errors[errors == ""][np.isnan(valid_scores)] = 'VINA'

            error_recorder.record(**{str(i): errors[i] for i in range(args.batch_size)})
            scores = torch.tensor(scores, device=device, dtype=torch.float)
            if not args.ignore_error:
                scores[torch.isnan(scores)] = error_score
            torch.clamp_(scores, min=args.min_score)
            score_recorder.record(**{str(i): score for i, score in enumerate(scores.tolist())})

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

            if is_starting:
                logger.info(f"step {step} scores={scores.cpu().tolist()}")

            # Forward Get prob & reward loss
            logger.debug(f"step[{step}] forward & backward started.")
            step_timer.start('forward_backward')
            model.train()
            La, B = out_batch.shape
            mbatch_size = solve_increasing_fn_left(lambda bsz: get_gpuuse(bsz, La)-args.gpu_size, 16)
            if mbatch_size == 0:
                raise ValueError(f"Input was too large and {mbatch_size=}")
            if is_starting:
                logger.info(f"{mbatch_size=}")
            reward_loss = 0
            kl_loss = 0
            with model.join():
                for mbatch_start in range(0, B, mbatch_size):
                    mslice = slice(mbatch_start, mbatch_start+mbatch_size)
                    out_mbatch = out_batch[:, mslice]
                    position_mbatch = position_batch[:,mslice]
                    weight_mbatch = weight[:, mslice]
                    scores_mbatch = scores[mslice]
                    with (nullcontext() if args.all_autocast else torch.autocast('cuda', torch.bfloat16)), \
                            sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        with torch.inference_mode():
                            init_logits = init_model(out_mbatch[:-1], position_mbatch[:-2]) # [L, B, N]
                            init_log_probs_all = F.log_softmax(init_logits, dim=-1).detach() # [Lo-1, B, N]
                        logits = model(out_mbatch[:-1], position_mbatch[:-2]) # [Lo-1, B, T]
                        if args.use_categorical:
                            cat = Categorical(logits=logits) # ~[Lo-1, B]
                            log_probs = cat.log_prob(out_mbatch[1:]) # [Lo-1, B]
                        else:
                            log_probs_all = F.log_softmax(logits, dim=-1) # [Lo-1, B, N]
                            log_probs = torch.gather(log_probs_all, dim=-1, index=out_mbatch[1:].unsqueeze(-1)).squeeze(-1) # [Lo-1, B]
                        reward_loss_mbatch = torch.sum(-scores_mbatch*(log_probs*weight_mbatch).sum(dim=0)/weight_mbatch.sum(dim=0))

                        ## KL loss
                        log_probs_all = F.log_softmax(logits, dim=-1)
                        kl_loss_mbatch = F.kl_div(input=log_probs_all, target=init_log_probs_all, reduction='none', 
                            log_target=True) # [Lo-1, B, N]
                        kl_loss_mbatch = kl_loss_mbatch.sum(dim=-1) # [Lo-1, B]
                        kl_loss_mbatch = torch.sum(kl_loss_mbatch*weight_mbatch)
                        
                        loss = (reward_loss_mbatch + kl_loss_mbatch * args.alpha) * loss_scale
                    loss.backward()
                    reward_loss += reward_loss_mbatch.item()
                    kl_loss += kl_loss_mbatch.item()

        # check nan
        step_timer.start('check_nan')
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

        # Optimizer's step
        logger.debug(f"step[{step}] optim started")
        step_timer.start('optim')
        if args.clip_grad_value is not None:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_value)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        step_recorder.record(batch_size=B, max_len=max(prompt_sizes), lr=scheduler.get_last_lr()[0], 
                memory=psutil.virtual_memory().used/(2**30), 
                reward_loss=reward_loss, kl_loss=kl_loss)
        scheduler.step()
        
        if step % args.record_opt == 0:
            if rank == SAVE_RANK:
                torch.save(model.state_dict(), f"{result_dir}/models/{step}.pth")
                cleardir(f"{result_dir}/optimizers")
                torch.save(optimizer.state_dict(), f"{result_dir}/optimizers/{step}.pth")
            if args.gc:
                gc.collect()
        if is_starting:
            logger.info(f"{step=} finished.")

        if is_starting:
            logger.debug(f"Actual GPU use={torch.cuda.max_memory_allocated(device)/2**30:.03f}")
            torch.cuda.reset_peak_memory_stats()

        if step == log_sample_step:
            logger.info("RDKit logger will be disabled from now on.")
            getLogger('rdkit').propagate = False

    step_recorder.flush()
    error_recorder.flush()

    logger.info("Training finished!")

    dist.destroy_process_group()

if __name__ == '__main__':
    main()