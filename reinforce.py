import sys, os, argparse, yaml, shutil, psutil, gc, math, random
import itertools as itr
import concurrent.futures as cf
from logging import getLogger
from collections import defaultdict
import numpy as np, pandas as pd
from addict import Dict
from glob import glob
from contextlib import nullcontext
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from torch.utils.data import Dataset, DataLoader, StackDataset
from torch.nn.utils.rnn import pad_sequence
from torch.distributions import Categorical
import transformers.utils.logging

from src.data._sampler import InfiniteRandomSampler
from src.model import Model, MambaModel2
from src.data.datasets.crossdocked import CDDataset
from src.data.protein import ProteinProcessDataset, CoordFollowDataset
from src.data.molecule import MolProcessDataset
from src.data.coord import CoordTransformDataset
from src.data import untuple, index_dataset
from src.data.tokenizer import ProteinAtomTokenizer, FloatTokenizer, TokenizeDataset, ArrayTokenizeDataset, VocEncoder, TokenEncodeDataset, SentenceDataset
from src.train import MAIN_RANK, sync_train_dir, set_sdp_kernel, get_train_logger, get_scheduler
from src.utils.path import timestamp, cleardir
from src.utils import set_random_seed
from src.utils.time import FileWatch
from src.utils.logger import INFO_WORKER
from src.evaluate import parse_mol_tokens, parse_mol
from src.evaluate import eval_vina, eval_qvina3
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
parser.add_argument('--path-to-qvina', default=f"{WORKDIR}/github/qvina/bin/qvina02")
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
## check generate_per_sample
assert (dist_size * args.batch_size) % args.generate_per_sample == 0
## make&sync result dir
result_dir = sync_train_dir(f"reinforce/results/{timestamp()}_{args.studyname}")
if is_main:
    os.makedirs(f"{result_dir}/steps", exist_ok=True)
    os.makedirs(f"{result_dir}/errors", exist_ok=True)
    os.makedirs(f"{result_dir}/scores", exist_ok=True)
    os.makedirs(f"{result_dir}/cplm", exist_ok=True)
    os.makedirs(f"{result_dir}/batches", exist_ok=True)
    shutil.copy2('reinforce.py', f"{result_dir}/cplm/reinforce.py")
    shutil.copytree('src', f"{result_dir}/cplm/src")
os.makedirs(f"{result_dir}/generated/{rank}", exist_ok=True)
## logger
logger = get_train_logger(result_dir)
logger.info(f"num_workers={args.num_workers}")
if auto_finetune_step:
    logger.info(f"finetune_step was set to {args.finetune_step}")
log_step = 1

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
if pargs.mamba:
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
else:
    # modlfy from MambaModel to MambaModel2
    new_state = {}
    for key, value in state_dict.items():
        assert key.startswith('module.')
        key = key[7:]
        new_state[key] = value
    state_dict = new_state


vocs = state_dict['vocs']
voc_encoder = VocEncoder(vocs[1:]) # remove '[PAD]'

# data

rstate = np.random.RandomState(args.seed)
protein, lig, score = untuple(CDDataset(args.finetune_save_dir), 3)
lig_smi, lig_coord = untuple(MolProcessDataset(lig, rstate, h_atom=True, h_coord=True, randomize=True), 2)
protein_atoms, protein_coord, coord_position = untuple(ProteinProcessDataset(protein, heavy_coord=args.pocket_coord_heavy), 2)

coords = CoordTransformDataset(lig_coord, protein_coord, rstate=rstate, normalize_coord=True, random_rotate=True)
lig_coord, protein_coord, center, rotation_matrix = untuple(coords, 4)

protein_atoms = TokenizeDataset(protein_atoms, ProteinAtomTokenizer())
protein_coord = ArrayTokenizeDataset(protein_coord, FloatTokenizer(-fargs.coord_range, fargs.coord_range))
if coord_follow_atom:
    train_data = SentenceDataset('[POCKET]', CoordFollowDataset(protein_atoms, protein_coord, coord_position))
else:
    train_data = SentenceDataset('[POCKET]', protein_atoms, '[XYZ]', protein_coord, '[LIGAND]')

assert train_data.vocs() < set(vocs)
train_data = TokenEncodeDataset(train_data, voc_encoder)
index_data, token_data = index_dataset(train_data)
train_data = StackDataset(index_data, token_data, center, rotation_matrix)
if not is_main:
    del train_data

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
                sampler=InfiniteRandomSampler(dataset, generator=torch.Generator().manual_seed(args.seed)),
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
                logger.info(f"{data=}")
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

## Scoring function 最大化したいものとする
match args.target:
    case 'min_vina':
        def get_score(lig_path: str, rec_path: str, out_dir: str):
            score, min_score = eval_vina(lig_path, rec_path, out_dir)
            if min_score is None:
                return np.nan
            return -min_score
        error_score = -50
    case 'vina':
        def get_score(lig_path: str, rec_path: str, out_dir: str):
            score, min_score = eval_vina(lig_path, rec_path, out_dir)
            if score is None:
                return np.nan
            return -score
        error_score = -50
    case 'mw_max':
        def get_score(lig_path: str, rec_path: str, out_dir: str):
            mol = Chem.SDMolSupplier(lig_path).__next__()
            if mol is None: 
                return np.nan
            return rdMolDescriptors.CalcExactMolWt(mol)
        error_score = 0
    case 'qvina':
        def get_score(lig_path: str, rec_path: str, out_dir: str):
            score = eval_qvina3(lig_path, rec_path, out_dir, timeout=60, path_to_qvina=args.path_to_qvina)
            if score is None:
                return np.nan
            return -score
        error_score = -50
    case 'dummy':
        def get_score(lig_path: str, rec_path: str, out_dir: str):
            return random.random()
        error_score = 0

if args.error_score is not None:
    error_score = args.error_score


# model
if pargs.mamba:
    net_model = MambaModel2(voc_encoder.i2voc, voc_encoder.pad_token, '[END]')
    init_model = MambaModel2(voc_encoder.i2voc, voc_encoder.pad_token, '[END]')
else:
    net_model = Model(12, 768, 12, 4, 0.0, 'gelu', True, voc_encoder.i2voc, voc_encoder.pad_token)
    init_model = Model(12, 768, 12, 4, 0.0, 'gelu', True, voc_encoder.i2voc, voc_encoder.pad_token)
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

# save at step 0
if is_main:
    torch.save(model.state_dict(), f"{result_dir}/models/{step}.pth")
    cleardir(f"{result_dir}/optimizers")
    torch.save(optimizer.state_dict(), f"{result_dir}/optimizers/{step}.pth")

logger.info("Training started.")
for step in range(args.max_step): 

    # get batch
    all_idxs, files, centers, rotations, batch = train_iter.__next__()
    batch = batch.to(device)
    centers = centers.cpu().numpy()
    rotations = rotations.cpu().numpy()
    steps['batch_size'].append(batch.shape[1])
    steps['max_len'].append(batch.shape[0])
    torch.save(batch.cpu(), f"{result_dir}/batches/{step}.pt") # TODO: remove batch

    # forward
    with torch.autocast('cuda', dtype=torch.bfloat16):

        ## generate sample
        model.eval()
        with torch.inference_mode():
            outputs = net_model.generate2(batch, '[END]', args.max_len, voc_encoder.pad_token, 10, args.tqdm_generate) # [B, L]

        out_batch = pad_sequence(outputs, batch_first, padding_value=voc_encoder.pad_token) # [L, B]
        Lo, B = out_batch.shape
        dtype = torch.float
        weight = torch.zeros((Lo-1, B), device=device, dtype=dtype) # [Lo-1, B]
        lig_count = torch.cumsum(out_batch == voc_encoder.voc2i['[LIGAND]'], dim=0) # [L, B]
        weight[lig_count[:-1] > 0] = 1.0
        end_count  = torch.cumsum(out_batch == voc_encoder.voc2i['[END]'], dim=0)
        weight[end_count[:-1] > 0] = 0.0

        ## Log output
        if step < log_sample_step:
            for idx in np.arange(B):
                context = voc_encoder.decode(batch[:,idx])
                output = voc_encoder.decode(outputs[idx])
                logger.debug(f"step {step}[{idx}]weight={weight[:,idx].tolist()}")
            ## check distribution
            if rank == dist_size-1:
                logger.log(INFO_WORKER, f"step {step}{all_idxs=}")
                logger.log(INFO_WORKER, f"step {step}{files=}")
                logger.log(INFO_WORKER, f"step {step}{centers=}")
                

        ## Get score
        do_save = step in do_save_steps
        errors = []
        with cf.ProcessPoolExecutor(args.num_score_workers) if (args.num_score_workers >= 2) else nullcontext() as e:
            futures = []
            valid_scores = []
            for idx in range(len(outputs)):

                center = centers[idx]
                rotation = rotations[idx]
                if do_save:
                    eval_dir = f"{result_dir}/eval_vina/{step}/{rank}/{idx}"
                    os.makedirs(eval_dir, exist_ok=True)
                else:
                    eval_dir = f"{result_dir}/eval_vina_tmp/{rank}/{idx}"

                score = np.nan
                out_tokens = voc_encoder.decode(outputs[idx].tolist())

                if do_save:
                    info = {**files.iloc[idx].to_dict(), 'idx': idx, 
                            'center': center.tolist(), 'rotation': rotation.tolist()}
                    with open(f"{eval_dir}/info.yaml", 'w') as f:
                        yaml.dump(info, f)
                    with open(f"{eval_dir}/tokens.txt", 'w') as f:
                        f.write(','.join(out_tokens)+'\n')
                error, smiles, coords = parse_mol_tokens(out_tokens)
                
                if error != '':
                    errors.append(error)
                    continue
                
                rotation_inv = np.linalg.inv(rotation)
                coords = np.matmul(coords, rotation_inv) + center
                error, mol = parse_mol(smiles, coords)
                errors.append(error)

                if error != '':
                    continue
                
                with open(f"{eval_dir}/lig.sdf", 'w') as f:
                    f.write(Chem.MolToMolBlock(mol))
                dname, lig_name, protein_name, sdf_idx = files.iloc[idx].tolist()
                
                lig_path = f"{eval_dir}/lig.sdf"
                rec_path = f"{WORKDIR}/cheminfodata/crossdocked/CrossDocked2020/{dname}/{protein_name}"
                if args.num_score_workers >= 2:
                    futures.append(e.submit(get_score, 
                            lig_path=lig_path, rec_path=rec_path, out_dir=eval_dir))
                else:
                    valid_scores.append(get_score(lig_path=lig_path, rec_path=rec_path, out_dir=eval_dir))
            if args.num_score_workers >= 2:
                valid_scores = np.array([f.result() for f in futures])
            else:
                valid_scores = np.array(valid_scores)

        errors = np.array(errors)
        scores = np.full(len(errors), np.nan)
        scores[errors == ""] = valid_scores
        errors[errors == ""][np.isnan(valid_scores)] = 'VINA'

        errorss.append(errors)
        scores = torch.tensor(scores, device=device, dtype=torch.float)
        if not args.ignore_error:
            scores[torch.isnan(scores)] = error_score
        torch.clamp_(scores, min=args.min_score)
        scoress.append(scores.cpu().tolist())

        ## gather & normalize score
        all_scores = [torch.zeros(args.batch_size, dtype=torch.float, device=device)
                for _ in range(dist_size)]
        dist.all_gather(all_scores, scores)
        all_scores = torch.cat(all_scores)
        
        match args.reward_scale:
            case 'none': 
                pass
            case 'all_mean':
                if torch.any(torch.isfinite(all_scores)):
                    scores = scores - torch.nanmean(all_scores)
            case 'sample_mean':
                idxs = all_idxs[rank*args.batch_size:(rank+1)*args.batch_size]
                unique_idxs = idxs.unique()
                for uidx in unique_idxs:
                    if torch.any(torch.isfinite(all_scores[all_idxs == uidx])):
                        scores[idxs == uidx] -= torch.nanmean(all_scores[all_idxs == uidx])
            case 'rank_mean':
                if torch.any(torch.isfinite(scores)):
                    scores = scores - torch.nanmean(scores)
            case 'rank_mean_std':
                if torch.any(torch.isfinite(scores)):
                    scores = scores - torch.nanmean(scores)
                    if torch.sum(torch.isfinite(scores)) >= 2:
                        scores = scores / (torch.std(scores[torch.isfinite(scores)])+1.0e-8)
        if args.ignore_error:
            scores[torch.isnan(scores)] = 0.0

        if step < 5:
            logger.info(f"step {step} scores={scores.cpu().tolist()}")

        ## Get prob & reward loss
        model.train()
        logits = model(out_batch[:-1]) # [Lo-1, B, T]
        if args.use_categorical:
            cat = Categorical(logits=logits) # ~[Lo-1, B]
            log_probs = cat.log_prob(out_batch[1:]) # [Lo-1, B]
        else:
            log_probs_all = F.log_softmax(logits, dim=-1) # [Lo-1, B, N]
            log_probs = torch.gather(log_probs_all, dim=-1, index=out_batch[1:].unsqueeze(-1)).squeeze(-1) # [Lo-1, B]
        reward_loss = torch.sum(-scores*(log_probs*weight).sum(dim=0)/weight.sum(dim=0))

        ## KL loss
        log_probs_all = F.log_softmax(logits, dim=-1)
        with torch.inference_mode():
            init_logits = init_model(out_batch[:-1]) # [L, B, N]
            init_log_probs_all = F.log_softmax(init_logits, dim=-1).detach() # [Lo-1, B, N]
        kl_loss = F.kl_div(input=log_probs_all, target=init_log_probs_all, reduction='none', 
            log_target=True) # [Lo-1, B, N]
        kl_loss = kl_loss.sum(dim=-1) # [Lo-1, B]
        kl_loss = torch.sum(kl_loss*weight)
        
        loss = (reward_loss + kl_loss * args.alpha) * loss_scale

    loss.backward()
    steps['reward_loss'].append(reward_loss.item())
    steps['kl_loss'].append(kl_loss.item())

    # Save step data
    if (step+1) % args.record_opt_step == 0:
        sdir = f"{result_dir}/step_data/{step}"
        os.makedirs(sdir, exist_ok=True)
        torch.save(log_probs_all.cpu(), f"{sdir}/log_probs_all.pt")
        torch.save(log_probs.cpu(), f"{sdir}/log_probs.pt")

    # check nan
    if args.reset_nan_grad:
        grad_is_finite = np.all([torch.all(torch.isfinite(param.grad)).item() for param in model.parameters()])
        if not grad_is_finite:
            if is_main:
                logger.warning("nan or infinite value in gradient. Gradient is reset.")
                for name, param in model.module.named_parameters():
                    n_nan = torch.sum(torch.isnan(param.grad)).item()
                    n_inf = torch.sum(torch.isinf(param.grad)).item()
                    if n_nan > 0 or n_inf > 0:
                        logger.warning(f"{name}: {n_nan=}, {n_inf=}")

            ## save situation
            if is_main and not nan_grad_step_saved:
                nan_dir = f"{result_dir}/nan_step_{step}/{rank}"
                os.makedirs(nan_dir, exist_ok=True)
                torch.save(batch.detach().cpu(), f"{nan_dir}/batch.pt")
                torch.save(model.state_dict(), f"{nan_dir}/model.pth")
                nan_grad_step_saved = True
            
            ## reset grad
            optimizer.zero_grad()

    if args.clip_grad_value is not None:
        torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad_value)
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    step += 1
    steps['lr'].append(scheduler.get_last_lr()[0])
    steps['memory'].append(psutil.virtual_memory().used/(2**30))
    scheduler.step()
    
    if step % args.record_opt_step == 0:
        pd.DataFrame(steps).to_csv(f"{result_dir}/steps/{rank}.csv")
        pd.DataFrame(scoress).to_csv(f"{result_dir}/scores/{rank}.csv")
        pd.DataFrame(errorss).to_csv(f"{result_dir}/errors/{rank}.csv")
        watch.flush()
        if is_main:
            torch.save(model.state_dict(), f"{result_dir}/models/{step}.pth")
            cleardir(f"{result_dir}/optimizers")
            torch.save(optimizer.state_dict(), f"{result_dir}/optimizers/{step}.pth")
        if args.gc:
            gc.collect()

    if step % log_step == 0:
        logger.info(f"{step=} finished.")

    if step == log_sample_step:
        getLogger('rdkit').propagate = False
    
logger.info("Training finished!")

dist.destroy_process_group()
