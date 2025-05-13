from collections import OrderedDict
from typing import Any
import torch
import torch.nn as nn

def get_substate(state_dict: OrderedDict[str, Any], prefix: str):
    return OrderedDict({key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)})
