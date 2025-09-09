import os, subprocess, math, gc
from collections import defaultdict

import torch

def get_mems(device, offset = None, mem_path = None):
    torch.cuda.empty_cache()
    use = torch.cuda.memory_allocated(device)
    mx = torch.cuda.max_memory_allocated(device)
    out = subprocess.run('nvidia-smi --format=csv --query-gpu=memory.used', shell=True, 
            capture_output=True)
    nv = float(out.stdout.decode('ascii').split('\n')[1].split(' ')[0]) * 2**20
    
    gcm = 0
    shape_dtype2n = defaultdict(int)
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor):
            if obj.device == device:
                gcm += math.ceil(obj.numel()*obj.dtype.itemsize / 512) * 512
                shape_dtype2n[tuple(obj.shape), obj.dtype, obj.dtype.itemsize] += 1
        """
        else:
            try:
                has_data = hasattr(obj, 'data')
            except Exception:
                has_data = False
            if has_data:
                if isinstance(obj.data, torch.Tensor):
                    raise ValueError
        """
    if mem_path is not None:
        os.makedirs(os.path.dirname(mem_path), exist_ok=True)
        with open(mem_path, 'w') as f:
            f.write("shape,dtype,itemsize,n\n")
            for (shape, dtype, itemsize), n in shape_dtype2n.items():
                f.write(f"{' '.join(str(s) for s in shape)},{dtype},{itemsize},{n}\n")

    output = [use, mx, nv, gcm]
    if offset is not None:
        output = [out - off for out, off in zip(output, offset)]
    return output