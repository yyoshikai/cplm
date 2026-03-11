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
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from src.utils import IterateRecorder
from src.utils.path import cleardir
from src.utils.ddp import reduce_float, all_gather
from src.train.collator import solve_increasing_fn_left
from src.data.tokenizer import VocEncoder
from src.model import Model
from src.train import get_optimizer_scheduler, get_process_ranks
from src.train.looper import Looper

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
    
def all_gather_counter(c: dict) -> dict:
    cs = [None]*dist.get_world_size()
    dist.all_gather_object(cs, c)
    all_c = {}
    for c in cs:
        for k, n in c.items():
            if k in all_c:
                all_c[k] += n
            else:
                all_c[k] = n
    return all_c

def get_velocity(
        model: ReinforceModel, 
        input: Tensor, 
        position: Tensor, 
        weight: Tensor, 
        scores: Tensor, 
        idxs: Tensor, 
        sample_whiten: list[Literal['mean', 'std']], 
        all_whiten: list[Literal['mean', 'std']], 
        mbatch_size: int
) -> Tensor:
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
    velocity = []
    with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
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

def scale_loss(exp_loss: Tensor, weight: Tensor, scale: list[Literal['batch_size', 'all_exp', 'episode_exp']]) -> Tensor:
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
        all_exp = reduce_float(weight.sum().item(), weight.device)
        episode_losses /= all_exp
    if 'episode_exp' in scale:
        episode_losses /= ns_exp
    return episode_losses

def get_mbatch_size(mbatch_size: int|None, L: int, gpu_size: float, model: Model, bfloat16: bool=True) -> int:
    
    if mbatch_size is None:
        mbatch_size = solve_increasing_fn_left(lambda bsz: model.get_gpuuse(bsz, L, bfloat16, 'FLASH')-gpu_size, 16)
    if mbatch_size == 0:
        raise ValueError(f"Input was too large")
    return mbatch_size

class Trainer:
    def train(self, input, output, position, weight, scores, idxs):
        raise NotImplementedError
    def save(self):
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

    def train(self, input, output, position, weight, scores, idxs):
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

def get_log_ratio(model: Model, init_model: Model, input: Tensor, output: Tensor, position: Tensor, weight: Tensor, mbatch_size: int|None, gpu_size: float):

    L, B = input.shape
    mbatch_size = get_mbatch_size(mbatch_size, L, gpu_size, init_model)
    rs = [] # [B, ]
    with torch.autocast('cuda', torch.bfloat16), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for mb in range(0, B, mbatch_size):
            mslice = slice(mb, mb+mbatch_size)
            with torch.inference_mode():
                init_logits_m = init_model(input[:,mslice], position[:,mslice]).detach() # [L, Bm]
            logits_m = model(input[:,mslice], position[:,mslice]) # [L, Bm]
            init_logits_m = torch.sum(init_logits_m*weight[:,mslice], dim=0)
            logits_m = torch.sum(logits_m*weight[:,mslice], dim=0)
            rs.append((logits_m-init_logits_m))
    return torch.cat(rs)

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


class DPOTrainer(Trainer):
    def __init__(self, model: Model, max_opt, weight_decay_all, weight_decay, schedule_free, scheduler, lr, warmup_ratio, log_optimizer, mbatch_size, gpu_size, kl_factor: float, train_looper: Looper, clip_grad_value: float|None, clip_grad_norm: float):
        """
        DPO Trainer with Placket-Luce model
        """
        self.step = 0
        self._batch_info = self._step_info = {}
        self.rank = dist.get_rank()
        
        self.init_model = copy.deepcopy(model)
        self.model = DistributedDataParallel(model)
        def grad_hook(name):
            def hook(grad):
                if torch.isnan(grad).any():
                    print(f"NaN detected in gradient of {name}")
            return hook
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.register_hook(grad_hook(name))

        self.optimizer, self.scheduler = get_optimizer_scheduler(self.model, max_opt, weight_decay_all, weight_decay, schedule_free, scheduler, lr, warmup_ratio, log_optimizer)

        self.mbatch_size = mbatch_size
        self.gpu_size = gpu_size
        self.kl_factor = kl_factor
        self.clip_grad_value = clip_grad_value
        self.clip_grad_norm = clip_grad_norm
        self.train_looper = train_looper

        _, self.save_rank, _ = get_process_ranks()

    def train(self, input: Tensor, output: Tensor, position: Tensor, weight: Tensor, scores: Tensor, idxs: Tensor):
        
        self.model.train()
        self.optimizer.zero_grad()
        L, B = input.shape
        mbatch_size = get_mbatch_size(self.mbatch_size, L, self.gpu_size, self.init_model)
        rs = [] # [B, ]
        with torch.autocast('cuda', torch.bfloat16), \
            sdpa_kernel(SDPBackend.FLASH_ATTENTION), \
            torch.inference_mode():
            for mb in range(0, B, mbatch_size):
                mslice = slice(mb, mb+mbatch_size)
                init_logits_m = self.init_model(input[:,mslice], position[:,mslice]) # [L, Bm, D]
                init_log_probs_m = get_log_prob(init_logits_m, output[:,mslice]) # [L, Bm]
                logits_m = self.model(input[:,mslice], position[:,mslice]) # [L, Bm, D]
                log_probs_m = get_log_prob(logits_m, output[:,mslice])
                rs_m = torch.sum((log_probs_m-init_log_probs_m)*weight[:,mslice], dim=0)*self.kl_factor
                rs.append(rs_m)
        rs = torch.cat(rs)
        all_rs = all_gather(rs)
        all_rs.requires_grad_()
        all_idxs = all_gather(idxs)
        all_scores = all_gather(scores)
        loss = 0
        for idx in torch.unique(all_idxs).tolist():
            irs = all_rs[all_idxs == idx]
            iscores = all_scores[all_idxs == idx] # ex. [1.0, 0.5, 1.5, 0.0]
            irs = irs[torch.argsort(-iscores, stable=False)]
            Bi = len(irs)
            for i in range(Bi):
                loss += torch.log_softmax(irs[i:], dim=0)[0]
        loss.backward()
        r_grads = all_rs.grad[B*self.rank:B*(self.rank+1)]*0.001

        self.model.train()
        log_probs = []
        mbatch_size = get_mbatch_size(self.mbatch_size, L, self.gpu_size, self.model.module, True)
        with torch.autocast('cuda', torch.bfloat16), \
                sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for mb in range(0, B, mbatch_size):
                mslice = slice(mb, mb+mbatch_size)
                logits_m = self.model(input[:,mslice], position[:,mslice]) # [L, Bm, D]
                log_probs_m = get_log_prob(logits_m, output[:,mslice])
                log_probs.append(log_probs_m.detach())
                rs_m = torch.sum(log_probs_m*weight[:,mslice], dim=0)
                rs_m.backward(gradient=r_grads[mslice])

        self._batch_info = {
            'log_prob': (torch.cat(log_probs, dim=1)/math.log(10)).float().cpu(),
            'velocity': r_grads.float().cpu().unsqueeze(0).expand_as(input)
        }
        self._step_info = dict(lr=self.scheduler.get_last_lr()[0], loss=loss.item())
        self._step_info.update({f'r_grad_{i}': r_grads[i].item() for i in range(B)})

        # Optimizer's step
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
            if self.step > 0:
                cleardir(f"{result_dir}/optimizers")
                torch.save(self.optimizer.state_dict(), f"{result_dir}/optimizers/{self.step}.pth")
    def policy_model(self):
        return self.model.module
    def batch_info(self):
        return self._batch_info
    def step_info(self):
        return self._step_info
    
class GRPOTrainer(Trainer):
    def __init__(self, model: Model, baseline: bool, max_opt, weight_decay_all, weight_decay, schedule_free, scheduler, lr, warmup_ratio, log_optimizer, mbatch_size, gpu_size, adv_sample_whiten, adv_all_whiten, loss_scale, kl_factor: float, value_factor: float, train_looper: Looper, clip_grad_value: float|None, clip_grad_norm: float):
        self.step = 0
        self._batch_info = self._step_info = {}

        self.rank = dist.get_rank()

        self.init_model = copy.deepcopy(model)
        self.model = DistributedDataParallel(model)
        self.optimizer, self.scheduler = get_optimizer_scheduler(self.model, max_opt, weight_decay_all, weight_decay, schedule_free, scheduler, lr, warmup_ratio, log_optimizer)

class WrapTrainer(Trainer):
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
    def train(self, input, output, position, weight, scores, idxs):
        return self.trainer.train(input, output, position, weight, scores, idxs)
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

    def train(self, input: Tensor, output: Tensor, position: Tensor, weight: Tensor, scores: Tensor, idxs):
        L, B = input.shape
        self.trainer.train(input, output, position, weight, scores, idxs)

        if self.step in self.do_save_steps:
            
            os.makedirs(f"{self.result_dir}/batches/{self.step}", exist_ok=True)
            for b in (range(B) if self.step in [0, 100] else [0]):
                pd.DataFrame(dict(
                    input=[self.voc_encoder.i2voc[i] for i in input[:,b]], 
                    output=[self.voc_encoder.i2voc[i] for i in output[:,b]], 
                    position=position[:,b].float().cpu(),
                    weight=weight[:,b].float().cpu(),
                    **{k: v[:, b] for k, v in self.batch_info().items()})
                ).to_csv(f"{self.result_dir}/batches/{self.step}/{self.rank}_{b}.csv")
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

    def train(self, input, output, position, weight, scores, idxs):
        self.trainer.train(input, output, position, weight, scores, idxs)
        self.step_recorder.record(**self.step_info())
        if self.step == self.max_opt:
            self.step_recorder.flush()
