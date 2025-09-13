from typing import Optional, TypeVar

import torch
import torch.distributed as dist
from torch import Tensor
T = TypeVar('T')

def dist_send_tensor(tensor: Tensor, dst: int) -> Tensor:
    dist.send_object_list([tensor.shape, tensor.dtype], dst=dst)
    dist.send(tensor, dst=dst)

def dist_recv_tensor(src: int, recv_device: torch.device) -> Tensor:
    info = [None, None]
    dist.recv_object_list(info, src=src)
    shape, dtype = info
    tensor = torch.zeros(shape, dtype=dtype, device=recv_device)
    dist.recv(tensor, src=src)
    return tensor

def dist_broadcast_object(obj: Optional[T], src: int) -> T:
    if dist.get_rank() != src:
        obj = None
    obj_box = [obj]
    dist.broadcast_object_list(obj_box, src=src)
    return obj_box[0]

def dist_broadcast_tensor(t: Tensor|None, device: torch.device, src: int,
            shape: torch.Size|None=None, dtype: torch.dtype|None=None) -> Tensor:
    is_src = dist.get_rank() == src
    
    # Check tensor at src rank
    if is_src: 
        assert t is not None
        if shape is not None: assert tuple(t.shape) == tuple(shape)
        if dtype is not None: assert t.dtype == dtype

    # Check tensor info & collect missing info
    info_to_send = []
    if shape is None:
        info_to_send.append(t.shape if is_src else None)        
    if dtype is None:
        info_to_send.append(t.dtype if is_src else None)
    if len(info_to_send) > 0:
        dist.broadcast_object_list(info_to_send, src=src, device=device)
    if shape is None: shape = info_to_send.pop(0)
    if dtype is None: dtype = info_to_send.pop(0)

    # Send tensor
    if not is_src:
        t = torch.zeros(shape, dtype=dtype, device=device)
    dist.broadcast(t, src=src)
    return t


def reduce_float(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, dtype=torch.float, device=device)
    dist.all_reduce(tensor)
    return tensor.item()