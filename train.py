# 241025 若干argumentを変更した。
# 250901 argumentを変更
import sys, os
import argparse
from functools import partial
from logging import Logger

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
from src.data.datasets.pdb import PDBDataset
from src.data.molecule import MolProcessDataset
from src.data.protein import ProteinProcessDataset, CoordFollowDataset
from src.data import CacheDataset
from src.model import Model
from src.model.mamba import MambaModel2
from src.utils import set_logtime
from src.utils.path import timestamp
from src.train import train, add_train_args, set_default_args, get_train_logger, sync_train_dir, log_dataset, MAIN_RANK
from src.utils.logger import INFO_WORKER

# arguments
parser = argparse.ArgumentParser()
## settings
parser.add_argument("--studyname", default='default')
parser.add_argument("--test", action='store_true')

## training
add_train_args(parser)

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
parser.add_argument("--coord-range", type=int, default=200)
parser.add_argument("--coord-follow-atom", action='store_true')

## model
parser.add_argument('--mamba', action='store_true')

args = parser.parse_args()
set_default_args(args)
if args.test: args.studyname+='_test'
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

# data
smiles_tokenizer = StringTokenizer(open("src/data/smiles_tokens.txt").read().splitlines())
coord_tokenizer = FloatTokenizer(-args.coord_range, args.coord_range, log_interval=args.tokenizer_log_interval)
protein_atom_tokenizer = ProteinAtomTokenizer(log_interval=args.tokenizer_log_interval)
sample_save_dir = f"{result_dir}/ligand_sample" if args.test else None

# datasets
if is_main:
    split2sizes = {}
    seed_a = 0
    vocs = set()
    for split in ['valid', 'train']:
        datas = []
        weight_datas = []
        datas_to_log = []
        ## Molecule
        for cls in [UniMolLigandDataset, UniMolLigandNoMolNetDataset]:
            repeat = getattr(args, cls.__name__.removesuffix('Dataset'))
            if repeat == 0: continue
            
            mol = cls(split=split, sample_save_dir=sample_save_dir)
            
            ### repeat
            if split == 'train' and repeat != 1:
                mol = RepeatDataset(mol, repeat)

            ### log data at this point
            datas_to_log.append(mol)
            
            ### process
            smi, coord = MolProcessDataset(mol, np.random.default_rng(args.seed+seed_a), 
                h_atom=not args.no_lig_atom_h, h_coord=not args.no_lig_coord_h, randomize=args.lig_randomize, sample_save_dir=sample_save_dir).untuple()
            coord = CoordTransformDataset(coord, rstate=args.seed+seed_a, normalize_coord=True, random_rotate=True, coord_noise_std=args.coord_noise_std).untuple()[0]

            ### tokenize
            smi = TokenizeDataset(smi, smiles_tokenizer)
            coord = ArrayTokenizeDataset(coord, coord_tokenizer)
            mol_data = SentenceDataset('[LIGAND]', smi, '[XYZ]', coord, '[END]')
            vocs |= mol_data.vocs()
            
            ### weight
            mol_data = CacheDataset(mol_data)
            separates = {'[LIGAND]', '[END]'}
            separates2weight = {('[LIGAND]',): 1.0, ('[LIGAND]', '[END]'): 0.0}
            mol_weight_data = RemoveLastDataset(TokenWeightDataset(mol_data, separates, separates2weight))

            datas.append(mol_data)
            weight_datas.append(mol_weight_data)
            seed_a += 1

        ## Protein
        protein_data = []
        for cls in [UniMolPocketDataset, UniMolPocketNoTDTestDataset, PDBDataset]:
            repeat = getattr(args, cls.__name__.removesuffix('Dataset'))
            if repeat == 0: continue

            pocket = cls(split=split)
            
            ### repeat
            if split == 'train' and repeat != 1:
                pocket = RepeatDataset(pocket, repeat)

            ### log at this point
            datas_to_log.append(pocket)

            ### process
            atoms, coord, coord_position = ProteinProcessDataset(pocket, heavy_atom=not args.no_pocket_atom_heavy, heavy_coord=args.pocket_coord_heavy, h_atom=args.pocket_atom_h, h_coord=args.pocket_coord_h).untuple()
            coords = CoordTransformDataset(coord, rstate=np.random.default_rng(args.seed+seed_a), normalize_coord=True, random_rotate=True, coord_noise_std=args.coord_noise_std).untuple()[0]

            ### tokenize
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

            datas.append(pocket_data)
            weight_datas.append(pocket_weight_data)
            seed_a += 1
        log_dataset(logger, split, datas_to_log)

        ### encode words
        voc_encoder = VocEncoder(vocs)
        datas = [TokenEncodeDataset(data, voc_encoder) for data in datas]

        ### merge weight
        datas = [StackDataset(data, weight_data) for data, weight_data in zip(datas, weight_datas)]

        split2sizes[split] = [len(data) for data in datas]
        if split == 'train':
            train_data = ConcatDataset(datas)
        else:
            valid_datas = datas
    sharings = [voc_encoder, split2sizes]
else:
    train_data, valid_datas = None, None
    sharings = [None, None]
dist.broadcast_object_list(sharings, src=MAIN_RANK)
voc_encoder, split2sizes = sharings

# model
if args.mamba:
    model = MambaModel2(voc_encoder.i2voc, voc_encoder.pad_token, '[END]')
else:
    model = Model(12, 768, 12, 4, 0.0, 'gelu', True, voc_encoder.i2voc, voc_encoder.pad_token)
model.to(torch.bfloat16)
model.to(device)
model = DistributedDataParallel(model)

# DataLoader
train_loader = DDPStringCollateLoader(train_data, model.module, args.num_workers, args.pin_memory, args.prefetch_factor, 
    args.gpu_size_gb*(2**30), batch_first, voc_encoder.pad_token, True, args.sdp_kernel, device,  main_rank=MAIN_RANK, seed=args.seed)
train(args, train_loader, voc_encoder, model, result_dir, device, log_step, args.seed)

dist.destroy_process_group()
