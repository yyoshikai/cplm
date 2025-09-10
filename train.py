# 241025 若干argumentを変更した。
# 250901 argumentを変更
import sys, os
import argparse
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, StackDataset
from torch.nn.parallel import DistributedDataParallel
import transformers.utils.logging
WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path.append(WORKDIR)

from src.data import RepeatDataset
from src.data.collator import DDPStringCollateLoader
from src.data.tokenizer import StringTokenizer, FloatTokenizer, \
    ProteinAtomTokenizer, TokenizeDataset, ArrayTokenizeDataset, \
    SentenceDataset, VocEncoder, TokenEncodeDataset, TokenWeightDataset, RemoveLastDataset
from src.data.coord import CoordTransformDataset
from src.data.datasets.unimol import UniMolLigandDataset, UniMolLigandNoMolNetDataset, UniMolPocketDataset, UniMolPocketNoTDTestDataset
from src.data.datasets.pdb import PDBDataset, PDBNoTDTestDataset
from src.data.molecule import MolProcessDataset
from src.data.protein import ProteinProcessDataset, CoordFollowDataset
from src.data import untuple, CacheDataset
from src.model import Model
from src.model.mamba import MambaModel2
from src.utils import set_logtime
from src.utils.path import timestamp
from src.train import train, add_train_args, get_train_logger, sync_train_dir, MAIN_RANK
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
parser.add_argument("--coord-range", type=int, default=200)

# bool系は何も指定しない場合BindGPTの設定になるようにしている
## dataset
parser.add_argument("--UniMolLigand", type=int, default=0)
parser.add_argument('--UniMolLigandNoMolNet', type=int, default=0)
parser.add_argument('--UniMolPocket', type=int, default=0)
parser.add_argument('--UniMolPocketNoTDTest', type=int, default=0)
parser.add_argument('--PDB', type=int, default=0)
parser.add_argument('--PDBNoTDTest', type=int, default=0)

## process
parser.add_argument("--lig-randomize", action='store_true')
parser.add_argument("--no-lig-coord-h", action='store_true')
parser.add_argument("--no-lig-atom-h", action='store_true')
parser.add_argument("--no-pocket-atom-heavy", action='store_true')
parser.add_argument("--pocket-coord-heavy", action='store_true')
parser.add_argument("--pocket-atom-h", action='store_true')
parser.add_argument("--pocket-coord-h", action='store_true')

parser.add_argument("--coord-follow-atom", action='store_true')
parser.add_argument('--mamba', action='store_true')

args = parser.parse_args()

# defaults in test
if args.test: args.studyname+='_test'
if args.record_opt_step is None:
    args.record_opt_step = 1 if args.test else 1000
if args.tokenizer_log_interval is None:
    args.tokenizer_log_interval = 1000000 if args.test else int(1e7)
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
logger, data_logger = get_train_logger(result_dir)
logger.debug(f"num_workers={args.num_workers}")
logger.log(INFO_WORKER, f"{device=}, {torch.cuda.device_count()=}")


### logging on other libraries
transformers.utils.logging.enable_propagation()
transformers.utils.logging.disable_default_handler()

# data
smiles_tokenizer = StringTokenizer(open("src/data/smiles_tokens.txt").read().splitlines())
coord_tokenizer = FloatTokenizer(-args.coord_range, args.coord_range, log_interval=args.tokenizer_log_interval)
protein_atom_tokenizer = ProteinAtomTokenizer(log_interval=args.tokenizer_log_interval)
sample_save_dir = f"{result_dir}/ligand_sample" if args.test else None

if is_main:
    ## datasets
    print('datasets started.', flush=True)
    ### Molecule
    mol_data = []
    for cls in [UniMolLigandDataset, UniMolLigandNoMolNetDataset]:
        repeat = getattr(args, cls.__name__.removesuffix('Dataset'))
        if repeat > 0:
            data = cls(sample_save_dir=sample_save_dir)
            logger.info(f"{cls.__name__}: {len(data)}*{repeat}={len(data)*repeat}")
            mol_data.append(RepeatDataset(data, repeat))

    ### Protein
    protein_data = []
    for cls in [UniMolPocketDataset, UniMolPocketNoTDTestDataset, PDBDataset, PDBNoTDTestDataset]:
        repeat = getattr(args, cls.__name__.removesuffix('Dataset'))
        if repeat > 0:
            data = cls()
            logger.info(f"{cls.__name__}: {len(data)}*{repeat}={len(data)*repeat}")
            protein_data.append(RepeatDataset(data, repeat))

    ## process
    print('Process started.', flush=True)
    vocs = set()
    datas = []
    weight_datas = []
    ### Process Molecules
    if len(mol_data) > 0:
        mol = ConcatDataset(mol_data)
        mol = MolProcessDataset(mol, np.random.default_rng(args.seed), 
            h_atom=not args.no_lig_atom_h, h_coord=not args.no_lig_coord_h, randomize=args.lig_randomize, sample_save_dir=sample_save_dir)
        smi, coord = untuple(mol, 2)
        coords = CoordTransformDataset(coord, rstate=args.seed, normalize_coord=True, random_rotate=True, coord_noise_std=args.coord_noise_std)
        coord = untuple(coords, 1)[0]

        smi = TokenizeDataset(smi, smiles_tokenizer)
        coord = ArrayTokenizeDataset(coord, coord_tokenizer)
        mol_data = SentenceDataset('[LIGAND]', smi, '[XYZ]', coord, '[END]')
        vocs |= mol_data.vocs()
        
        ### weight
        mol_data = CacheDataset(mol_data)
        separates = {'[LIGAND]', '[END]'}
        separates2weight = {('[LIGAND]',): 1.0, ('[LIGAND]', '[END]'): 0.0}
        mol_weight_data = RemoveLastDataset(TokenWeightDataset(mol_data, separates, separates2weight))

        logger.debug(f"mol data: {len(mol_data)}")
        datas.append(mol_data)
        weight_datas.append(mol_weight_data)
    ### Process proteins
    print('Process protein started', flush=True)
    if len(protein_data) > 0:
        pocket = ConcatDataset(protein_data)
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
        pocket_data = CacheDataset(pocket_data)

        #### weight
        separates = {'[POCKET]', '[END]'}
        separates2weight = { ('[POCKET]',): 1.0, ('[POCKET]', '[END]'): 0.0 }
        pocket_weight_data = RemoveLastDataset(TokenWeightDataset(pocket_data, separates, separates2weight))

        logger.debug(f"pocket data: {len(pocket_data)}")
        datas.append(pocket_data)
        weight_datas.append(pocket_weight_data)
    print('Process ended', flush=True)

    voc_encoder = VocEncoder(vocs)
    print(1, flush=True)
    datas = ConcatDataset(datas)
    print(2, flush=True)
    datas = TokenEncodeDataset(datas, voc_encoder)
    print(3, flush=True)
    weight_datas = ConcatDataset(weight_datas)
    print(4, flush=True)
    print(len(datas), len(weight_datas), flush=True)
    train_data = StackDataset(datas, weight_datas)
    print(5, flush=True)
    voc_encoder_d = [voc_encoder]
    print(6, flush=True)
else:
    voc_encoder_d = [None]
print('broadcast started.', flush=True)
dist.broadcast_object_list(voc_encoder_d, src=MAIN_RANK)
print('broadcast ended', flush=True)
voc_encoder = voc_encoder_d[0]

# model
if args.mamba:
    model = MambaModel2(voc_encoder.i2voc, voc_encoder.pad_token, '[END]')
else:
    model = Model(args.n_layer, 768, 12, 4, 0.1, 'gelu', True, voc_encoder.i2voc, voc_encoder.pad_token)
model.to(torch.bfloat16)
model.to(device)
model = DistributedDataParallel(model)

# DataLoader
print('train_loader', flush=True)
train_loader = DDPStringCollateLoader(train_data, voc_encoder, model.module, args.num_workers, args.pin_memory, args.prefetch_factor, 
    args.gpu_size_gb*(2**30), batch_first, voc_encoder.pad_token, True, args.sdp_kernel, device,  main_rank=MAIN_RANK, seed=args.seed)
print('train started.', flush=True)
train(args, train_loader, voc_encoder, model, result_dir, device, log_step, args.seed)

dist.destroy_process_group()
