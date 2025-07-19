import sys, os, argparse, yaml, math
from logging import getLogger
from addict import Dict
from glob import glob
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import transformers.utils.logging

from src.model import MambaModel2
from src.data.tokenizer import VocEncoder
from src.train import sync_train_dir, set_sdp_kernel, get_train_logger
from src.utils.path import timestamp
from src.utils import set_random_seed
WORKDIR = os.environ.get('WORKDIR', os.path.abspath('..'))

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--studyname', required=True)
## trainings
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--max-len', type=int, default=2500)
parser.add_argument('--reward-scale', choices=['none', 'all_mean', 'sample_mean', 
        'rank_mean', 'rank_mean_std'], default='none')
parser.add_argument('--ignore-invalid', action='store_true')
parser.add_argument('--fix-pocket', action='store_true')

## optimizer
parser.add_argument('--weight-decay', type=float, default=0.0) # same as BindGPT
parser.add_argument('--clip-grad-norm', type=float, default=1.0) # same as BindGPT
parser.add_argument('--clip-grad-value', type=float, default=None)
parser.add_argument('--scheduler', default='constant') # same as BindGPT
parser.add_argument('--lr', type=float, default=1.4e-5) # same as BindGPT
parser.add_argument('--alpha', type=float, default=0.05) # same as BindGPT
parser.add_argument('--loss-scale')
parser.add_argument('--max-step', type=int, default=1000)
## data
parser.add_argument('--finetune-save-dir', required=True)
parser.add_argument('--pocket-coord-heavy', action='store_true')
parser.add_argument('--target', choices=['min_vina', 'vina', 'mw_max', 'logp', 'qvina', 'dummy'], default='min_vina')
parser.add_argument('--generate-per-sample', type=int, default=1)
## finetune
parser.add_argument('--finetune-name', required=True)
parser.add_argument('--finetune-step', type=int)
## environment
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--num-score-workers', type=int, default=1)
parser.add_argument('--pin-memory', action='store_true')
parser.add_argument('--prefetch-factor', type=int)
parser.add_argument('--sdp-kernel', choices=['FLASH', 'CUDNN', 'MATH', 'EFFICIENT'])
parser.add_argument('--reset-nan-grad', action='store_true')
parser.add_argument('--gc', action='store_true')
parser.add_argument('--use-categorical', action='store_true')
parser.add_argument('--tqdm-generate', action='store_true')
## record
parser.add_argument('--record-opt-step', type=int)
parser.add_argument('--tokenizer-log-interval', type=int)
## not classified
parser.add_argument('--error-score', type=float, default=None)
parser.add_argument('--min-score', type=float, default=-math.inf)
parser.add_argument('--ignore-error', action='store_true')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

# get finetune info
finetune_dir = f"finetune/results/{args.finetune_name}"
fargs = Dict(yaml.safe_load(open(f"{finetune_dir}/config.yaml")))
pname = fargs.pretrain_name
pretrain_dir = f"training/results/{pname}"
pargs = Dict(yaml.safe_load(open(f"{pretrain_dir}/config.yaml")))
coord_follow_atom = pargs.get('coord_follow_atom', False)

## get last finetune step
auto_finetune_step = False
if args.finetune_step is None:
    steps = [os.path.splitext(os.path.basename(step))[0] for step in glob(f"{finetune_dir}/models/*")]
    steps = [int(step) for step in steps if step.isdigit()]
    args.finetune_step = max(steps)
    auto_finetune_step = True

# set default args
if args.test: args.studyname+='_test'
if args.record_opt_step is None:
    args.record_opt_step = 1 if args.test else 100
if args.tokenizer_log_interval is None:
    args.tokenizer_log_interval = 10000 if args.test else int(1e7)
batch_first = False
log_sample_step = 3
do_save_steps = [0, 1, 2, 3, 4, 50, 100]+list(range(200, 1000, 200)) \
        +list(range(1000, args.max_step, 1000))

# DDP
dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo')
rank = dist.get_rank()
dist_size = dist.get_world_size()
torch.cuda.set_device(rank % torch.cuda.device_count())
device = torch.device('cuda', index=rank % torch.cuda.device_count()) \
    if torch.cuda.is_available() else torch.device('cpu')

## check generate_per_sample
assert (dist_size * args.batch_size) % args.generate_per_sample == 0
## make&sync result dir
result_dir = sync_train_dir(f"reinforce/results/{timestamp()}_{args.studyname}")
## logger
logger = get_train_logger(result_dir)
logger.info(f"num_workers={args.num_workers}")
if auto_finetune_step:
    logger.info(f"finetune_step was set to {args.finetune_step}")
log_step = 1
getLogger('rdkit').propagate = False

### logging on other libraries
transformers.utils.logging.enable_propagation()
transformers.utils.logging.disable_default_handler()

# load state dict(for vocs)
state_dict = torch.load(f"{finetune_dir}/models/{args.finetune_step}.pth", 
    map_location=device, weights_only=True)
# modlfy from MambaModel to MambaModel2
new_state = {}
for key, value in state_dict.items():
    assert key.startswith('module.')
    key = key[7:]
    if key == 'vocs':
        pass
    else:
        key = f"model.{key}"
    new_state[key] = value
state_dict = new_state
vocs = state_dict['vocs']
voc_encoder = VocEncoder(vocs[1:]) # remove '[PAD]'

# model
net_model = MambaModel2(voc_encoder.i2voc, voc_encoder.pad_token, '[END]')
init_model = MambaModel2(voc_encoder.i2voc, voc_encoder.pad_token, '[END]')
net_model.to(torch.bfloat16)
init_model.to(torch.bfloat16)
net_model.to(device)
init_model.to(device)
## Load state dict
net_model.load_state_dict(state_dict)
init_model.load_state_dict(net_model.state_dict())
model = DistributedDataParallel(net_model)

# training

# Environment
## fix seed
set_random_seed(args.seed)
if args.test:
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

## scaled dot product attention kernel
set_sdp_kernel(args.sdp_kernel)

## optimizer
if args.weight_decay == 0.0:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
else:
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
optimizer.zero_grad()

batch = torch.load("/workspace/cplm/reinforce/results/250719_test0/batches/0.pt") \
        .to(device)
with torch.autocast('cuda', dtype=torch.bfloat16):
    model.eval()
    with torch.inference_mode():
        outputs = net_model.generate2(batch, '[END]', args.max_len, voc_encoder.pad_token, 10, args.tqdm_generate) # [B, L]
dist.destroy_process_group()
