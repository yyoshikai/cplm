import os, psutil, math, copy
from collections import defaultdict, Counter
from typing import Literal, Any
from contextlib import nullcontext
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from src.utils import IterateRecorder, wraps
from src.utils.path import cleardir
from src.utils.ddp import reduce_float, all_gather
from src.train.collator import solve_increasing_fn_left
from src.data.tokenizer import VocEncoder
from src.model import Model
from src.train import get_optimizer_scheduler, get_process_ranks
from src.train.looper import Looper
from .reinforce import ReinforceModel, get_all_stat, get_sample_stat, all_gather_counter, scale_loss, get_mbatch_size

def get_velocity(
        model: ReinforceModel, 
        input: Tensor, 
        position: Tensor, 
        weight: Tensor, 
        scores: list[Tensor], 
        idxs: Tensor, 
        sample_whiten: list[Literal['mean', 'std']], 
        all_whiten: list[Literal['mean', 'std']], 
        mbatch_size: int
) -> Tensor:
    """
    Parameters
    ----------
    input, position, weight: Tensor(float)[L, B]
    scores: list[Tensor(float)[L+1,]]

    Returns
    -------
    velocity: Tensor(float)[L, B]
    
    """
    model
    L, B = input.shape
    velocity = []
    if model.baseline:
        values = []
        with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for mbatch_start in range(0, B, mbatch_size):
                mslice = slice(mbatch_start, mbatch_start+mbatch_size)
                _, values_m = model(input[:,mslice], position[:,mslice]) # [L,B]
                values += list(values_m.T)
            advantage = [scores[b][1:] - values[b] for b in range(B)]
            velocity = [torch.cumsum(adv[::-1])[::-1] for adv in advantage]
    else:
        velocity = [torch.cumsum(score[:1:-1])[::-1] for score in scores]
    velocity = pad_sequence(velocity) # [L, B]

    # sample-wise whiten
    ## get sample-wise stat
    idx2n = {}
    idx2s = {}
    idx2s2 = {}
    for idx in torch.unique(idxs).tolist():
        idx_weight = weight[:, idxs == idx]
        idx_velocity = velocity[:, idxs == idx]
        idx2n[idx] = idx_weight.sum().item()
        idx2s[idx] = (idx_velocity*idx_weight).sum().item()
        idx2s2[idx] = (idx_velocity**2*idx_weight).sum().item()
    idx2n = all_gather_counter(idx2n)
    idx2s = all_gather_counter(idx2s)
    idx2s2 = all_gather_counter(idx2s2)

    means = torch.zeros_like(scores)
    stds = torch.zeros_like(scores)
    for idx in torch.unique(idxs).tolist():
        mean = idx2s[idx] / idx2n[idx]
        std = math.sqrt((idx2s2[idx] / idx2n[idx]) - mean**2)+1e-5
        means[idxs == idx] = mean
        stds[idxs == idx] = std
    ## whiten
    sample_whiten_mean = 'mean' in sample_whiten
    sample_whiten_std = 'std' in sample_whiten
    if sample_whiten_mean and sample_whiten_std:
        velocity = (velocity-means)/stds
    elif sample_whiten_mean:
        velocity = (velocity-means)
    elif sample_whiten_std:
        velocity = (velocity-means)/stds+means

    # all whiten
    all_n = sum(idx2n.values())
    all_s = sum(idx2s.values())
    all_s2 = sum(idx2s2.values())
    all_mean = all_s / all_n
    all_std = math.sqrt((all_s2/all_n)-all_mean**2)+1e-5
    all_s
    all_whiten_mean = 'mean' in all_whiten
    all_whiten_std = 'std' in all_whiten
    if all_whiten_mean and all_whiten_std:
        velocity = (velocity-all_mean)/all_std
    elif all_whiten_mean:
        velocity = velocity-all_mean
    elif all_whiten_std:
        velocity = (velocity-all_mean)/all_std+all_mean
    return velocity

class Trainer:
    def train(self, input: Tensor, output: Tensor, position: Tensor, weight: Tensor, scores: Tensor, errors: list[str], idxs: Tensor):
        raise NotImplementedError
    def save(self, result_dir: str):
        raise NotImplementedError
    def policy_model(self) -> Model:
        raise NotImplementedError
    def batch_info(self) -> dict[str, Tensor]:
        raise NotImplementedError
    def step_info(self) -> dict[str, Any]:
        raise NotImplementedError


class ReinforceTrainer(Trainer):
    def __init__(self, model: Model, baseline: bool, max_opt, weight_decay_all, weight_decay, schedule_free, scheduler, lr, warmup_ratio, log_optimizer, mbatch_size, gpu_size, adv_sample_whiten, adv_all_whiten, loss_scale, kl_factor: float, value_factor: float, train_looper: Looper, clip_grad_value: float|None, clip_grad_norm: float, result_dir: str):
        self.step = 0
        self._batch_info = {}
        self._step_info = {}

        device = next(model.parameters()).device
        self.rank = dist.get_rank()
        _, self.save_rank, _ = get_process_ranks()

        self.init_model = copy.deepcopy(model)
        self.reinforce_model = ReinforceModel(model, baseline)
        self.reinforce_model.to(device)
        self.model = DistributedDataParallel(self.reinforce_model)

        # optimizer
        self.optimizer, self.scheduler = get_optimizer_scheduler(self.model, max_opt, weight_decay_all, weight_decay, schedule_free, scheduler, lr, warmup_ratio, log_optimizer)

        # train args
        self.mbatch_size = mbatch_size
        self.gpu_size = gpu_size
        self.adv_sample_whiten = adv_sample_whiten
        self.adv_all_whiten = adv_all_whiten
        self.loss_scale = loss_scale
        self.kl_factor = kl_factor
        self.value_factor = value_factor
        self.train_looper = train_looper
        self.clip_grad_value = clip_grad_value
        self.clip_grad_norm = clip_grad_norm
        self.result_dir = result_dir

    def train(self, input: Tensor, output: Tensor, position: Tensor, weight: Tensor, scores: list[Tensor], errors: list[str], idxs: Tensor):
        # get & whiten velocity
        L, B = input.shape
        mbatch_size = get_mbatch_size(self.mbatch_size, L, self.gpu_size, self.reinforce_model.model)
        self.train_looper.put('velocity')
        velocity = get_velocity(self.reinforce_model, input, position, weight, scores, idxs, self.adv_sample_whiten, self.adv_all_whiten, mbatch_size)

        self.train_looper.put('forward_backward')
        self.model.train()
        self.optimizer.zero_grad()
        save_grad = self.step in [0, 100]
        with self.model.join():
            reward_loss = 0.0
            kl_loss = 0.0
            value_loss = 0.0
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
                        init_logits = self.init_model(input_m, position_m) # [L, B, N]
                        init_log_probs_all = F.log_softmax(init_logits, dim=-1).detach() # [L, B, N]
                    logits_m, values_m = self.model(input_m, position_m) # [L, B, T], [L, B]

                    log_probs_all = F.log_softmax(logits_m, dim=-1) # [L, B, N]
                    log_probs_m = torch.gather(log_probs_all, dim=-1, index=output_m.unsqueeze(-1)).squeeze(-1) # [L, B]
                    reward_loss_m = -velocity[:, mslice]*log_probs_m
                    reward_loss_m = scale_loss(reward_loss_m, weight_m, self.loss_scale)

                    ## KL loss
                    log_probs_all = F.log_softmax(logits_m, dim=-1)
                    kl_loss_m = F.kl_div(input=log_probs_all, target=init_log_probs_all, reduction='none', 
                        log_target=True) # [Lo-1, B, N]
                    kl_loss_m = kl_loss_m.sum(dim=-1) # [Lo-1, B]
                    kl_loss_m = scale_loss(kl_loss_m, weight_m, self.loss_scale) * self.kl_factor

                    if self.reinforce_model.baseline:
                        value_loss_m = (rewards_m-values_m)**2*0.5
                        value_loss_m = scale_loss(value_loss_m, weight_m, self.loss_scale) * self.value_factor
                    else:
                        value_loss_m = torch.zeros_like(velocity, requires_grad=True) # [L, B]
                        values_m = torch.zeros_like(velocity) # [L, B]
                                        
                if save_grad:
                    self.train_looper.put('save_grad')
                    terms = {'reward': reward_loss_m.sum(), 'kl': kl_loss_m.sum(), 'value': value_loss_m.sum()}
                    for term_name, term in terms.items():
                        grads = torch.autograd.grad(term, self.model.parameters(), retain_graph=True, allow_unused=True)
                        for (param_name, _), grad in zip(self.model.named_parameters(), grads):
                            if grad is not None:
                                term2grads[term_name][param_name] += grad
                    self.train_looper.put('forward_backward')
                loss = reward_loss_m.sum()+kl_loss_m.sum()+value_loss_m.sum()
                loss.backward()
                reward_loss += reward_loss_m.sum().item()
                kl_loss += kl_loss_m.sum().item()
                value_loss += value_loss_m.sum().item()
                log_probs.append(log_probs_m.detach())
                values.append(values_m.detach())
        self._batch_info = {
            'log_prob': (torch.cat(log_probs, dim=1) / math.log(10)).float().cpu(), 
            'velocity': velocity.float().cpu(),
            'value': torch.cat(values, dim=1).float().cpu()
        }
        self._step_info = dict(
            lr=self.scheduler.get_last_lr()[0], 
            reward_loss=reward_loss, 
            kl_loss=kl_loss, 
            value_loss=value_loss, 
            mbatch_size=mbatch_size,
            pred_gpu_use=self.reinforce_model.model.get_gpuuse(mbatch_size, L, True, 'FLASH')
        )
        if save_grad:
            for term, grads in term2grads.items():
                for param_name, grad in grads.items():
                    dist.reduce(grad, dst=self.save_rank, op=dist.ReduceOp.AVG)
            if self.rank == self.save_rank:
                for term, grads in term2grads.items():
                    torch.save(dict(grads), f"{self.result_dir}/grads/{term}/step{self.step}.pth")

        # Optimizer's step
        self.train_looper.put('optim')
        if self.clip_grad_value is not None:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad_value)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1

    def save(self, result_dir):
        if self.rank == self.save_rank:
            os.makedirs(f"{result_dir}/models", exist_ok=True)
            torch.save(self.policy_model().state_dict(), f"{result_dir}/models/{self.step}.pth")
            if self.reinforce_model.baseline:
                os.makedirs(f"{result_dir}/value_models", exist_ok=True)
                torch.save(self.reinforce_model.value_head.state_dict(), f"{result_dir}/value_models/{self.step}.pth")
            if self.step > 0:
                cleardir(f"{result_dir}/optimizers")
                torch.save(self.optimizer.state_dict(), f"{result_dir}/optimizers/{self.step}.pth")
    def policy_model(self):
        return self.reinforce_model.model
    def batch_info(self):
        return self._batch_info
    def step_info(self):
        return self._step_info

def get_log_prob(logits: Tensor, output: Tensor):
    """
    Args:
        logits (Tensor(float)[L, B, T]): logits (before softmax)
        output (Tensor(long)[L, B]): output indices
    Returns:
        Tensor(float)[L, B]: log prob of output token
    """
    log_probs_all = F.log_softmax(logits, dim=-1) # [L, B, N]
    log_probs = torch.gather(log_probs_all, dim=-1, index=output.unsqueeze(-1)).squeeze(-1) # [L, B]
    return log_probs


class WrapTrainer(Trainer):
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
    def train(self, input: Tensor, output: Tensor, position: Tensor, weight: Tensor, scores: Tensor, errors: list[str], idxs: Tensor):
        return self.trainer.train(input, output, position, weight, scores, errors, idxs)
    def save(self, result_dir: str):
        self.trainer.save(result_dir)
    def policy_model(self):
        return self.trainer.policy_model()
    def batch_info(self):
        return self.trainer.batch_info()
    def step_info(self):
        return self.trainer.step_info()

class SaveBatchTrainer(WrapTrainer):
    def __init__(self, trainer: Trainer, result_dir: str, do_save_steps: list[int], voc_encoder: VocEncoder):
        super().__init__(trainer)
        self.step = 0
        self.rank = dist.get_rank()
        self.result_dir = result_dir
        self.do_save_steps = do_save_steps
        self.voc_encoder = voc_encoder

    def train(self, input: Tensor, output: Tensor, position: Tensor, weight: Tensor, scores: Tensor, errors: list[str], idxs: Tensor):
        L, B = input.shape
        self.trainer.train(input, output, position, weight, scores, errors, idxs)

        if self.step in self.do_save_steps:
            
            os.makedirs(f"{self.result_dir}/batches/{self.step}", exist_ok=True)
            for b in (range(B) if self.step in [0, 100] else [0]):
                df = pd.DataFrame(dict(
                    input=[self.voc_encoder.i2voc[i] for i in input[:,b]], 
                    output=[self.voc_encoder.i2voc[i] for i in output[:,b]], 
                    position=position[:,b].float().cpu(),
                    weight=weight[:,b].float().cpu(),
                    **{k: v[:, b] for k, v in self.batch_info().items()})
                )
                df.to_csv(f"{self.result_dir}/batches/{self.step}/{self.rank}_{b}.csv")
                with open(f"{self.result_dir}/batches/{self.step}/{self.rank}_{b}.txt", 'w') as f:
                    f.write(df.to_string())
                
        self.step += 1

class GetMemoryTrainer(WrapTrainer):
    def __init__(self, trainer: Trainer, device: torch.device):
        super().__init__(trainer)
        self.device = device

    def step_info(self):
        step_info = self.trainer.step_info()
        step_info.update({
            'memory_gb': psutil.virtual_memory().used/(2**30), 
            'gpu_gb': torch.cuda.max_memory_allocated(self.device)/2**30
        })
        return step_info

class SaveStepTrainer(WrapTrainer):
    def __init__(self, trainer: Trainer, result_dir: str, record_opt: int, max_opt: int):
        super().__init__(trainer)
        self.step = 0
        rank = dist.get_rank()
        self.step_recorder = IterateRecorder(f"{result_dir}/steps/{rank}.csv", record_opt)
        self.max_opt = max_opt

    def train(self, input: Tensor, output: Tensor, position: Tensor, weight: Tensor, scores: Tensor, errors: list[str], idxs: Tensor):
        self.trainer.train(input, output, position, weight, scores, errors, idxs)
        self.step_recorder.record(**self.step_info())
        if self.step == self.max_opt:
            self.step_recorder.flush()
