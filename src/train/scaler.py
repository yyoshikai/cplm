import math
from collections.abc import Callable
import torch
import torch.distributed as dist
from torch import Tensor

def all_gather(values: Tensor, dim=0):
    world_size = dist.get_world_size()
    all_values = [torch.zeros_like(values) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    return torch.cat(all_values, dim=dim)

def get_sample_stat(values: Tensor, idxs: Tensor) -> tuple[Tensor, Tensor]:
    rank = dist.get_rank()
    all_values = all_gather(values)
    all_idxs = all_gather(idxs)
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

def whiten_scores(scores: Tensor, idxs: Tensor, sample_mean: bool, sample_std: bool, all_mean: bool, all_std: bool):
    world_size = dist.get_world_size()
    # all_gather scores
    all_scores = [torch.zeros_like(scores) for _ in range(world_size)]
    dist.all_gather(all_scores, scores)
    all_scores = torch.cat(all_scores)
    
    # sample whiten
    sample_means, sample_stds = get_sample_stat(scores, idxs)
    sample_stds+=1e-5
    if sample_mean or sample_std:
        if sample_mean and sample_std:
            scores = (scores-sample_means)/sample_stds
        elif sample_mean:
            scores = scores-sample_means
        elif sample_std:
            scores = (scores-sample_means)/sample_stds+sample_mean

    # all whiten
    mean, std = get_all_stat(scores)
    std += 1e-5
    if (all_mean or all_std) and not torch.all(torch.isnan(scores)):
        if all_mean and all_std:
            scores = (scores - mean) / std
        elif sample_mean:
            scores = scores - mean
        elif sample_std:
            scores = (scores - mean) / std + mean
    return scores

SCORE_SCALERS: dict[str, Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]] = {}
def register_score_scaler(name):
    def _register(func):
        SCORE_SCALERS[name] = func
        return func
    return _register

ADV_SCALERS = {}
def register_adv_scaler(name): 
    def _register(func):
        ADV_SCALERS[name] = func
        return func
    return _register

# 260224 旧 optionを何も指定しない場合
@register_score_scaler('none')
def default_score_scaler(scores: Tensor, idxs: Tensor, is_vina_error: Tensor, is_gen_error: Tensor):
    scores[torch.isnan(scores)] = 0.0
    return scores

# 260224 no_baselineの実装
@register_score_scaler('no_baseline')
def no_baseline_score_scaler(scores: Tensor, idxs: Tensor, is_vina_error: Tensor, is_gen_error: Tensor):
    scores[torch.isnan]