import sys, os
from argparse import ArgumentParser, Namespace
from glob import glob
import yaml
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

sys.path.append(f"{os.environ.get('WORKDIR', '/workspace')}/cplm")
from src.train.data import get_train_data
from src.train import init_ddp, get_model, validate, CrossEntropyLoss
from src.utils.logger import get_logger, add_file_handler

parser = ArgumentParser()
parser.add_argument('--studyname', required=True)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument("--sdp-kernel", choices=['FLASH', 'EFFICIENT'], default='FLASH')
parser.add_argument("--gpu-size-gb", type=float, required=True)
parser.add_argument("--val-sample", type=float)
args = parser.parse_args()
args.gpu_size = args.gpu_size_gb * (2**30)


dfdir = f"downstream/finetune/results/{args.studyname}"
with open(f"{dfdir}/args.yaml") as f:
    dfargs = Namespace(**yaml.safe_load(f))
tdir = f"training/results/{dfargs.pretrain_name}"
targs = Namespace(**yaml.safe_load(open(f"{tdir}/args.yaml")))

# env
rank, size, device = init_ddp()
logger = get_logger(stream=True)
add_file_handler(logger, f"{dfdir}/logs/eval_label/{rank}.log")

# data
targs.PDBUniMolRandom = targs.UniMolPocket = 0.0
if args.val_sample is not None:
    targs.UniMolLigand_val_sample = targs.UniMolLigandNoMolNet_val_sample = args.val_sample
datas, voc_encoder, data_names, data_log = get_train_data(targs, 'valid', 
        'reg' if dfargs.reg else 'cls', lig_weight=0.0, score_weight=1.0)
for msg, level in data_log:
    logger.log(level, msg)

# model
net_model = get_model(targs, voc_encoder, None, device)
criterion = CrossEntropyLoss(reduction='none', ignore_index=voc_encoder.pad_token)

opts = sorted(int(path.split('/')[-1].split('.')[0]) for path in glob(f"{dfdir}/models/*.pth"))

losses = []
for opt in opts:
    net_model.load_state_dict(torch.load(f"{dfdir}/models/{opt}.pth", weights_only=True))

    model = DistributedDataParallel(net_model)
    model.eval()

    _, _, total_weights, total_losses = validate(datas, data_names, voc_encoder, model, criterion, args.num_workers, False, args.sdp_kernel, args.gpu_size, f"[{opt}]", False)
    assert len(total_weights) == 1
    loss = total_losses[0].item() / total_weights[0].item()

    losses.append(loss)
    logger.info(f"{loss=}")
if rank == 0:
    pd.DataFrame({'opt': opts, 'loss': losses}).to_csv(f"{dfdir}/val_label.csv", index=False)
dist.destroy_process_group()
