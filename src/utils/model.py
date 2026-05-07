import itertools as itr
from collections import OrderedDict
from typing import Any
import torch
import torch.nn as nn

def get_substate(state_dict: OrderedDict[str, Any], prefix: str):
    return OrderedDict({key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)})

def get_num_params(model: nn.Module) -> int:
    return sum(param.numel() for param in itr.chain(model.parameters(), model.buffers()))
def get_model_size(model: nn.Module) -> int:
    return sum(param.numel() * param.element_size() for param in itr.chain(model.parameters(), model.buffers()))