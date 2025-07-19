import sys, os, argparse, yaml, shutil, psutil, gc, math, random
from logging import getLogger
from collections import defaultdict
import numpy as np, pandas as pd
from addict import Dict
from glob import glob
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from rdkit import Chem, RDLogger
from torch.utils.data import Dataset, DataLoader, StackDataset
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical
import transformers.utils.logging

from src.data.sampler import InfiniteRandomSampler
from src.model import MambaModel2
from src.data.finetune import CDDataset
from src.data import untuple_dataset, index_dataset
from src.data.tokenizer import ProteinAtomTokenizer, FloatTokenizer, TokenizeDataset, ArrayTokenizeDataset, VocEncoder, TokenEncodeDataset, SentenceDataset
from src.data.pretrain.protein import CoordFollowDataset
from src.train import MAIN_RANK, sync_train_dir, set_sdp_kernel, get_train_logger, get_scheduler
from src.utils.path import timestamp
from src.utils import set_random_seed
from src.utils.time import FileWatch
from src.utils.logger import INFO_WORKER
from src.evaluate import parse_mol_tokens, parse_mol
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
is_main = rank == MAIN_RANK
torch.cuda.memory._record_memory_history()
## check generate_per_sample
assert (dist_size * args.batch_size) % args.generate_per_sample == 0
## make&sync result dir
result_dir = sync_train_dir(f"reinforce/results/{timestamp()}_{args.studyname}")
if is_main:
    os.makedirs(f"{result_dir}/steps", exist_ok=True)
    os.makedirs(f"{result_dir}/errors", exist_ok=True)
    os.makedirs(f"{result_dir}/scores", exist_ok=True)
    os.makedirs(f"{result_dir}/cplm", exist_ok=True)
    shutil.copy2('reinforce.py', f"{result_dir}/cplm/reinforce.py")
    shutil.copytree('src', f"{result_dir}/cplm/src")
os.makedirs(f"{result_dir}/generated/{rank}", exist_ok=True)
## logger
logger = get_train_logger(result_dir)
logger.info(f"num_workers={args.num_workers}")
if auto_finetune_step:
    logger.info(f"finetune_step was set to {args.finetune_step}")
log_step = 1
getLogger('rdkit').propagate = False

### logging on other libraries
RDLogger.DisableLog("rdApp.*")
### logging on other libraries
transformers.utils.logging.enable_propagation()
transformers.utils.logging.disable_default_handler()

## vina evaluation
for i in range(args.batch_size):
    os.makedirs(f"{result_dir}/eval_vina_tmp/{rank}/{i}", exist_ok=True)

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

# data
cddata = CDDataset(args.finetune_save_dir, args.seed, mol_atom_h=True,
        mol_coord_h=True, pocket_coord_heavy=args.pocket_coord_heavy)
pocket_atom_data, pocket_coord_data, _lig_smi, _lig_coord, _score, center_data, \
    rotation_data = untuple_dataset(cddata, 7)
pocket_atom_data = TokenizeDataset(pocket_atom_data, ProteinAtomTokenizer())
pocket_coord_data = ArrayTokenizeDataset(pocket_coord_data, FloatTokenizer(-fargs.coord_range, fargs.coord_range))
if coord_follow_atom:
    train_data = SentenceDataset('[POCKET]', CoordFollowDataset(pocket_atom_data, pocket_coord_data))
else:
    train_data = SentenceDataset('[POCKET]', pocket_atom_data, '[XYZ]', pocket_coord_data, '[LIGAND]')

assert train_data.vocs() < set(vocs)
train_data = TokenEncodeDataset(train_data, voc_encoder)
index_data, token_data = index_dataset(train_data)
train_data = StackDataset(index_data, token_data, center_data, rotation_data)
if not is_main:
    del train_data
    train_data = None

## Make dataloader
class ReinforceIter:
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, dataset: Dataset, num_workers:int, pin_memory: bool, prefetch_factor: int, 
            batch_size: int, batch_first: bool, padding_value: int, repeat_per_sample: int,
            fix_pocket: bool, device: torch.device, main_rank: int=0):
        self.size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.main_rank = main_rank
        self.batch_size = batch_size
        self.repeat_per_sample = repeat_per_sample
        self.batch_first = batch_first
        self.padding_value = padding_value
        assert self.batch_size * self.size % self.repeat_per_sample == 0
        self.fix_pocket = fix_pocket

        self.device = device

        if self.rank == self.main_rank:
            loader = DataLoader(dataset, batch_size=None, 
                sampler=InfiniteRandomSampler(dataset),
                num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
            self.logger.info(f"Loading filenames.csv.gz ...")
            self.df_file = pd.read_csv(f"{args.finetune_save_dir}/filenames.csv.gz", index_col=0)
            self.logger.info("Loaded.")
            self.iter = loader.__iter__()
            self.next_item = None
            self.step = 0

            if self.fix_pocket:
                self.idx, self.data, self.center, self.rotation = self.iter.__next__()
                del self.iter

    def __next__(self) -> tuple[Tensor, pd.DataFrame, Tensor, Tensor]:
        if self.rank == self.main_rank:
            all_idxs = []
            all_datas = []
            all_centers = []
            all_rotations = []
            for _ in range(self.batch_size*self.size // self.repeat_per_sample):
                if self.fix_pocket:
                    idx, data, center, rotation = self.idx, self.data, self.center, self.rotation
                else:
                    idx, data, center, rotation = self.iter.__next__()
                all_idxs.append(idx)
                all_datas += [data]*self.repeat_per_sample
                all_centers.append(center)
                all_rotations.append(rotation)
            all_idxs = np.array(all_idxs).repeat(self.repeat_per_sample)
            all_files = [self.df_file.loc[all_idxs[rank*self.batch_size:(rank+1)*self.batch_size]]
                    for rank in range(self.size)]
            all_idxs = torch.tensor(all_idxs, dtype=torch.int, device=self.device)
            all_centers = torch.stack(all_centers).to(torch.float).to(self.device).repeat_interleave(self.repeat_per_sample, dim=0)
            all_rotations = torch.stack(all_rotations).to(torch.float).to(self.device).repeat_interleave(self.repeat_per_sample, dim=0)
            all_centers = list(torch.chunk(all_centers, chunks=self.size, dim=0))
            all_rotations = list(torch.chunk(all_rotations, chunks=self.size, dim=0))
            all_datas = [all_datas[rank*self.batch_size:(rank+1)*self.batch_size]
                    for rank in range(self.size)]
        else:
            all_idxs = torch.zeros(self.batch_size*self.size, dtype=torch.int, device=self.device)
            all_files = all_centers = all_rotations = all_datas = None

        dist.broadcast(all_idxs, src=self.main_rank)
        files = [None]
        dist.scatter_object_list(files, all_files, src=self.main_rank)
        files = files[0]
        centers = torch.zeros((self.batch_size, 3), dtype=torch.float, device=self.device)
        dist.scatter(centers, all_centers, src=self.main_rank)
        rotations = torch.zeros((self.batch_size, 3, 3), dtype=torch.float, device=self.device)
        dist.scatter(rotations, all_rotations, src=self.main_rank)
        datas = [None]
        dist.scatter_object_list(datas, all_datas, src=self.main_rank)
        datas = datas[0]
        batch = pad_sequence(datas, self.batch_first, self.padding_value).to(torch.long).to(self.device) 
        return all_idxs, files, centers, rotations, batch

train_iter = ReinforceIter(train_data, args.num_workers, 
    args.pin_memory, args.prefetch_factor, args.batch_size, batch_first, 
    voc_encoder.pad_token, args.generate_per_sample, args.fix_pocket, device, MAIN_RANK)

error_score = args.error_score or 0


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
## rank
rank = dist.get_rank()
is_main = rank == MAIN_RANK

## save args
if is_main:
    with open(f"{result_dir}/config.yaml", 'w') as f:
        yaml.dump(vars(args), f)

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
match args.loss_scale:
    case None:
        loss_scale = 1.0
    case 'token_per_batch':
        loss_scale = 1/args.token_per_step
    case _:
        loss_scale = float(args.loss_scale)

## scheduler
scheduler = get_scheduler(optimizer, args.scheduler, 55000)

## records
os.makedirs(f"{result_dir}/times", exist_ok=True)
watch = FileWatch(f"{result_dir}/times/{rank}.csv", 10000)
steps = defaultdict(list)
errorss = []
scoress = []
nan_grad_step_saved = False
step = 0

logger.info("Training started.")
for step in range(args.max_step): 
    logger.log(INFO_WORKER,f"step {step}")

    # get batch
    batch = torch.load("/workspace/cplm/reinforce/results/250719_test0/batches/0.pt") \
            .to(device)
    # forward
    with torch.autocast('cuda', dtype=torch.bfloat16):

        ## generate sample
        logger.log(INFO_WORKER, "generate sample")
        with watch.hold('generate'):
            model.eval()
            with torch.inference_mode():
                outputs = net_model.generate2(batch, '[END]', args.max_len, voc_encoder.pad_token, 10, args.tqdm_generate) # [B, L]
dist.destroy_process_group()
