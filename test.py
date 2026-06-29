import torch.distributed as dist

dist.init_process_group()

c = {11830145: 246268280.046875}
cs = [None]*dist.get_world_size()

dist.all_gather_object(cs, c)

dist.destroy_process_group()

