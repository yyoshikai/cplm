# 241025 若干argumentを変更した。
import sys, os
import argparse

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset
from torch.nn.parallel import DistributedDataParallel
import transformers.utils.logging
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path.append(WORKDIR)

from src.data import RepeatDataset
from src.data.collator import DDPStringCollateLoader
from src.data.tokenizer import StringTokenizer, FloatTokenizer, \
    ProteinAtomTokenizer, TokenizeDataset, ArrayTokenizeDataset, \
    SentenceDataset, VocEncoder, TokenEncodeDataset
from src.data.coord import CoordTransformDataset
from src.data.datasets.unimol import UniMolLigandDataset, UniMolPocketDataset
from src.data.molecule import MolProcessDataset
from src.data.protein import ProteinProcessDataset, CoordFollowDataset
from src.data import untuple
from src.model import Model
from src.model.mamba import MambaModel
from src.utils import set_logtime
from src.utils.path import timestamp
from src.train import CELoss, train, add_train_args, get_train_logger, sync_train_dir, MAIN_RANK
from src.utils.logger import INFO_WORKER

# arguments
parser = argparse.ArgumentParser()

## settings
parser.add_argument("--studyname", default='default')
parser.add_argument("--test", action='store_true')

## training
add_train_args(parser)

## model
parser.add_argument("--n-layer", type=int, default=8)

## data
# bool系は何も指定しない場合BindGPTの設定になるようにしている
parser.add_argument("--coord-range", type=int, default=200)
parser.add_argument("--mol-repeat", type=int, default=1)
parser.add_argument("--mol-data", default=f"{WORKDIR}/cheminfodata/unimol/ligands/train.lmdb")
parser.add_argument("--no-lig-coord-h", action='store_true')
parser.add_argument("--no-lig-atom-h", action='store_true')
parser.add_argument("--lig-randomize", action='store_true')
parser.add_argument("--pocket-repeat", type=int, default=1)
parser.add_argument("--pocket-data", default=f"{WORKDIR}/cheminfodata/unimol/pockets/train.lmdb")
parser.add_argument("--frag-repeat", type=int, default=1)
parser.add_argument("--frag-data")
parser.add_argument("--no-pocket-atom-heavy", action='store_true')
parser.add_argument("--pocket-coord-heavy", action='store_true')
parser.add_argument("--pocket-atom-h", action='store_true')
parser.add_argument("--pocket-coord-h", action='store_true')
parser.add_argument("--coord-follow-atom", action='store_true')
parser.add_argument("--frag-type", default='1')
parser.add_argument('--mamba', action='store_true')

args = parser.parse_args()

# defaults in test
if args.test: args.studyname+='_test'
if args.record_opt_step is None:
    args.record_opt_step = 1 if args.test else 1000
if args.tokenizer_log_interval is None:
    args.tokenizer_log_interval = 10000 if args.test else int(1e7)
log_step = 1 if args.test else 10000

batch_first = False
set_logtime(args.logtime)

## DDP
dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo', )
rank = dist.get_rank()
size = dist.get_world_size()
torch.cuda.set_device(rank % torch.cuda.device_count())
device = torch.device('cuda', index=rank % torch.cuda.device_count()) \
    if torch.cuda.is_available() else torch.device('cpu')
is_main = rank == MAIN_RANK

## make result dir
result_dir = sync_train_dir(f"training/results/{timestamp()}_{args.studyname}")

## logger
logger = get_train_logger(result_dir)
logger.info(f"num_workers={args.num_workers}")
logger.log(INFO_WORKER, f"{device=}, {torch.cuda.device_count()=}")

### logging on other libraries
transformers.utils.logging.enable_propagation()
transformers.utils.logging.disable_default_handler()

# data
datas = []
vocs = set()
smiles_tokenizer = StringTokenizer(open("src/data/smiles_tokens.txt").read().splitlines())
coord_tokenizer = FloatTokenizer(-args.coord_range, args.coord_range, log_interval=args.tokenizer_log_interval)
protein_atom_tokenizer = ProteinAtomTokenizer(log_interval=args.tokenizer_log_interval)
## mol data
if args.mol_repeat > 0:
    sample_save_dir = f"{result_dir}/ligand_sample" if args.test else None
    mol = UniMolLigandDataset(sample_save_dir)
    mol = MolProcessDataset(mol, np.random.default_rng(args.seed), 
        h_atom=not args.no_lig_atom_h, h_coord=not args.no_lig_coord_h, randomize=args.lig_randomize, sample_save_dir=sample_save_dir)
    smi, coord = untuple(mol, 2)
    coords = CoordTransformDataset(coord, rstate=args.seed, normalize_coord=True, random_rotate=True, coord_noise_std=args.coord_noise_std)
    coord = untuple(coords, 1)[0]

    smi = TokenizeDataset(smi, smiles_tokenizer)
    coord = ArrayTokenizeDataset(coord, coord_tokenizer)
    mol_data = SentenceDataset('[LIGAND]', smi, '[XYZ]', coord, '[END]')
    
    vocs |= mol_data.vocs()
    mol_data = RepeatDataset(mol_data, args.mol_repeat)
    logger.info(f"mol data: {len(mol_data)}")
    datas.append(mol_data)

## pocket data
if args.pocket_repeat > 0:
    pocket = UniMolPocketDataset()

    pocket = ProteinProcessDataset(pocket, heavy_atom=not args.no_pocket_atom_heavy, heavy_coord=args.pocket_coord_heavy, h_atom=args.pocket_atom_h, h_coord=args.pocket_coord_h)
    atoms, coord, coord_position = untuple(pocket, 3)
    
    coords = CoordTransformDataset(coord, rstate=np.random.default_rng(args.seed), normalize_coord=True, random_rotate=True, coord_noise_std=args.coord_noise_std)
    coord = untuple(coords, 1)[0]

    atoms = TokenizeDataset(atoms, protein_atom_tokenizer)
    coord = ArrayTokenizeDataset(coord, coord_tokenizer)
    if args.coord_follow_atom:
        pocket_data = CoordFollowDataset(atoms, coord, coord_position)
        pocket_data = SentenceDataset('[POCKET]', pocket_data, '[END]')
    else:
        pocket_data = SentenceDataset('[POCKET]', atoms, '[XYZ]', coord, '[END]')

    vocs |= pocket_data.vocs()
    pocket_data = RepeatDataset(pocket_data, args.pocket_repeat)
    logger.info(f"pocket data: {len(pocket_data)}")
    datas.append(pocket_data)

## fragment data
if args.frag_repeat > 0:
    raise NotImplementedError
    """
    match args.frag_type:
        case '1':
            frag_data = PDBFragmentDataset(args.frag_data)
        case '2':
            frag_data = PDBFragment2Dataset(args.frag_data)
        case '3':
            frag_data = LMDBDataset(args.frag_data, idx_to_key='str')
        case _:
            raise ValueError(f'Unsupported args.frag_class: {args.frag_type}')
    frag_data = ProteinDataset(frag_data, protein_atom_tokenizer, coord_tokenizer, coord_transform, 
        atom_heavy=not args.no_pocket_atom_heavy, coord_heavy=args.pocket_coord_heavy, 
        atom_h=args.pocket_atom_h, coord_h=args.pocket_coord_h, )
    vocs |= frag_data.vocs()
    frag_data = RepeatDataset(frag_data, args.frag_repeat)
    logger.info(f"frag data: {len(frag_data)}")
    datas.append(frag_data)
    """
train_data = ConcatDataset(datas)

voc_encoder = VocEncoder(vocs)
train_data = TokenEncodeDataset(train_data, voc_encoder)

if not is_main:
    del train_data

# Make dataset
train_loader = DDPStringCollateLoader(train_data, args.num_workers, args.pin_memory, args.prefetch_factor, 
    args.token_per_batch, batch_first, voc_encoder.pad_token, device, MAIN_RANK, seed=args.seed)

# model
if args.mamba:
    model = MambaModel(voc_encoder.i2voc, voc_encoder.pad_token, '[END]')
else:
    model = Model(args.n_layer, 768, 12, 4, 0.1, 'gelu', True, voc_encoder.i2voc, voc_encoder.pad_token)
model.to(torch.bfloat16)
model.to(device)
model = DistributedDataParallel(model)

criterion = CELoss(voc_encoder, args.seed)

train(args, train_loader, model, criterion, result_dir, is_main, device, log_step)

dist.destroy_process_group()
