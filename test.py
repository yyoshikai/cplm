

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchvision import models

device = torch.device('cuda')

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18()
        self.head1 = nn.Linear(1000, 2)
    def forward(self, x):
        return self.model(x)

dist.init_process_group()

model = Model().to(device)
model = DistributedDataParallel(model)
optimizer = torch.optim.SGD(model.parameters())
x = torch.randn(5, 3, 224, 224, device=device)
y1 = model(x)
for name, param in model.named_parameters():
    print(f"{name}: {param.grad}")
grads = torch.autograd.grad(torch.sum(y1), model.module.parameters(), retain_graph=True, allow_unused=True)
grads = torch.autograd.grad(torch.sum(y1), model.module.parameters(), retain_graph=True, allow_unused=True)
grads = torch.autograd.grad(torch.sum(y1), model.module.parameters(), retain_graph=True, allow_unused=True)
for name, param in model.named_parameters():
    print(f"{name}: {param.grad}")
torch.sum(y1).backward()
for name, param in model.named_parameters():
    print(f"{name}: {param.grad}")

dist.destroy_process_group()


