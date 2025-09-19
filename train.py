# 241025 若干argumentを変更した。
# 250901 argumentを変更
import sys, os
import argparse
import itertools as itr

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, StackDataset, Subset
WORKDIR = os.environ.get('WORKDIR', __file__.split('/cplm/')[0])
sys.path.append(WORKDIR)

from src.data import RepeatDataset
from src.data.tokenizer import StringTokenizer, FloatTokenizer, \
    ProteinAtomTokenizer, TokenizeDataset, ArrayTokenizeDataset, \
    SentenceDataset, VocEncoder, TokenEncodeDataset, TokenWeightDataset, RemoveLastDataset
from src.data.coord import CoordTransformDataset
from src.data.datasets.unimol import UniMolLigandNoMolNetDataset, UniMolPocketDataset
from src.data.datasets.pdb import PDBUniMolRandomDataset
from src.data.molecule import MolProcessDataset
from src.data.protein import ProteinProcessDataset, CoordFollowDataset
from src.data import CacheDataset
from src.model import Model
from src.model.mamba import MambaModel2
from src.utils import ddp_set_random_seed, set_deterministic
from src.utils.path import timestamp
from src.utils.ddp import dist_broadcast_tensor, dist_broadcast_object
from src.train import train, add_train_args, set_default_args, get_train_logger, sync_train_dir, log_dataset, DATA_RANK

# arguments
parser = argparse.ArgumentParser()
## settings
parser.add_argument("--studyname", default='default')

## training
add_train_args(parser)

# bool系は何も指定しない場合BindGPTの設定になるようにしている
# pocket-heavy-coordはデフォルトで入れるようにした。
## dataset
for cls in [UniMolLigandNoMolNetDataset, UniMolPocketDataset, PDBUniMolRandomDataset]:
    dname = cls.__name__.removesuffix('Dataset')
    parser.add_argument(f'--{dname}', type=int, default=0)
    parser.add_argument(f'--{dname}-val-sample', type=float, default=1.0)

## process
parser.add_argument("--lig-randomize", action='store_true')
parser.add_argument("--no-lig-h-atom", action='store_true')
parser.add_argument("--no-lig-h-coord", action='store_true')
parser.add_argument("--no-pocket-heavy-atom", action='store_true')
parser.add_argument("--no-pocket-heavy-coord", action='store_true')
parser.add_argument("--pocket-h-atom", action='store_true') # Datasetによっては無効？
parser.add_argument("--pocket-h-coord", action='store_true') # Datasetによっては無効？
parser.add_argument("--coord-range", type=int, default=250)
parser.add_argument("--coord-follow-atom", action='store_true')

## model
parser.add_argument('--mamba', action='store_true')
parser.add_argument('--n-layer', type=int)
### Transformer
parser.add_argument('--pos-buffer-len', type=int, default=1600)

args = parser.parse_args()
set_default_args(args)
if args.test: args.studyname+='_test'

# DDP
dist.init_process_group('nccl' if torch.cuda.is_available() else 'gloo', )
rank = dist.get_rank()
size = dist.get_world_size()
torch.cuda.set_device(rank % torch.cuda.device_count())
device = torch.device('cuda', index=rank % torch.cuda.device_count()) \
    if torch.cuda.is_available() else torch.device('cpu')
DATA_RANK = {k: r % size for k, r in DATA_RANK.items()}

# make result dir
result_dir = os.path.join('training', 'results', args.studyname)
## Add timestamp
head, tail = os.path.split(result_dir)
tail = f"{timestamp()}_{tail}"
result_dir = os.path.join(head, tail)
## sync
result_dir = sync_train_dir(result_dir)

# Fix seed (must be done before model initialization)
ddp_set_random_seed(args.seed)
if args.test or args.deterministic: set_deterministic()


# logger
root_logger, process_logger, data_logger = get_train_logger(result_dir)
root_logger.debug(f"{device=}, {torch.cuda.device_count()=}")

# data
smiles_tokenizer = StringTokenizer(open("src/data/smiles_tokens.txt").read().splitlines())
coord_tokenizer = FloatTokenizer('ligand', -args.coord_range, args.coord_range, log_interval=args.tokenizer_log_interval)
protein_coord_tokenizer = FloatTokenizer('pocket', -args.coord_range, args.coord_range, log_interval=args.tokenizer_log_interval)

protein_atom_tokenizer = ProteinAtomTokenizer(log_interval=args.tokenizer_log_interval)
sample_save_dir = f"{result_dir}/ligand_sample" if args.test else None

# datasets
vocs = set()
voc_encoder = train_data = valid_datas = train_data_sizes = valid_data_sizes = data_names = None
for split in ['valid', 'train']:
    if rank % size == DATA_RANK[split]:
        root_logger.debug(f"num_workers={args.num_workers}")
        
        datas = []
        weight_datas = []
        datas_to_log = []
        data_names = []
        ## Molecule
        for d_seed, cls in enumerate([UniMolLigandNoMolNetDataset, UniMolPocketDataset, PDBUniMolRandomDataset]):

            dname = cls.__name__.removesuffix('Dataset')
            repeat = getattr(args, dname)
            if repeat == 0: continue
            data_names.append(cls.__name__)
            
            raw = cls(split='valid' if 'data_epoch' in args.check else split)
            
            ## repeat / sample
            if split == 'train' and repeat != 1:
                raw = RepeatDataset(raw, repeat)
            sample = getattr(args, dname+'_val_sample')
            if (split == 'valid' or 'data_epoch' in args.check) and sample < 1.0:
                rng = np.random.default_rng(args.seed+d_seed)
                idxs = rng.choice(len(raw), size=round(len(raw)*sample))
                raw = Subset(raw, idxs)
                assert len(raw) > 0

            ## log data at this point
            datas_to_log.append(raw)
            
            ## process
            ### Molecules
            if cls in [UniMolLigandNoMolNetDataset]:
                mol = raw
                smi, coord = MolProcessDataset(mol, args.seed+d_seed, 
                    h_atom=not args.no_lig_h_atom, h_coord=not args.no_lig_h_coord, 
                    randomize=args.lig_randomize, sample_save_dir=None).untuple()
                coord = CoordTransformDataset(coord, base_seed=args.seed+d_seed, 
                    normalize_coord=True, random_rotate=True, coord_noise_std=args.coord_noise_std).untuple()[0]

                ### tokenize
                smi = TokenizeDataset(smi, smiles_tokenizer)
                coord = ArrayTokenizeDataset(coord, coord_tokenizer)
                mol_data = SentenceDataset('[LIGAND]', smi, '[XYZ]', coord, '[END]')
                vocs |= mol_data.vocs()
                
                ### weight
                data = CacheDataset(mol_data)
                separates = {'[LIGAND]', '[END]'}
                separates2weight = {('[LIGAND]',): 1.0, ('[LIGAND]', '[END]'): 0.0}
                weight_data = RemoveLastDataset(TokenWeightDataset(mol_data, separates, separates2weight))
            ### Pockets
            else:
                pocket = raw
                atoms, coord, coord_position = ProteinProcessDataset(pocket, heavy_atom=not args.no_pocket_heavy_atom, heavy_coord=not args.no_pocket_heavy_coord, h_atom=args.pocket_h_atom, h_coord=args.pocket_h_coord).untuple()

                coords = CoordTransformDataset(coord, base_seed=args.seed+d_seed, normalize_coord=True, random_rotate=True, coord_noise_std=args.coord_noise_std).untuple()[0]

                ### tokenize
                atoms = TokenizeDataset(atoms, protein_atom_tokenizer)
                coord = ArrayTokenizeDataset(coord, protein_coord_tokenizer)
                if args.coord_follow_atom:
                    pocket_data = CoordFollowDataset(atoms, coord, coord_position)
                    pocket_data = SentenceDataset('[POCKET]', pocket_data, '[END]')
                else:
                    pocket_data = SentenceDataset('[POCKET]', atoms, '[XYZ]', coord, '[END]')
                vocs |= pocket_data.vocs()
                data = CacheDataset(pocket_data)

                #### weight
                separates = {'[POCKET]', '[END]'}
                separates2weight = { ('[POCKET]',): 1.0, ('[POCKET]', '[END]'): 0.0 }
                weight_data = RemoveLastDataset(TokenWeightDataset(pocket_data, separates, separates2weight))

            datas.append(data)
            weight_datas.append(weight_data)
            d_seed += 1
            
        log_dataset(root_logger, split, datas_to_log)

        ### encode words
        voc_encoder = VocEncoder(vocs)
        datas = [TokenEncodeDataset(data, voc_encoder) for data in datas]

        ### merge weight
        datas = [StackDataset(data, weight_data) for data, weight_data in zip(datas, weight_datas)]

        if split == 'train':
            train_data_sizes = torch.tensor([len(data) for data in datas], device=device)
            train_data = ConcatDataset(datas)
        else:
            valid_data_sizes = torch.tensor([len(data) for data in datas], device=device)
            valid_datas = datas

# Broadcast
train_data_sizes = dist_broadcast_tensor(train_data_sizes, device, src=DATA_RANK['train'])
valid_data_sizes = dist_broadcast_tensor(valid_data_sizes, device, src=DATA_RANK['valid'])
voc_encoder, data_names = dist_broadcast_object((voc_encoder, data_names), DATA_RANK['train'])
train2valid_r = train_data_sizes.to(torch.float) / valid_data_sizes.to(torch.float)

# model
if args.mamba:
    kwargs = {}
    if args.n_layer is not None: kwargs['num_hidden_layers'] = args.n_layer
    model = MambaModel2(voc_encoder.i2voc, voc_encoder.pad_token, '[END]', **kwargs)
else:
    num_layers = args.n_layer or 12
    model = Model(num_layers, 768, 12, 4, 0.0, 'gelu', True, voc_encoder.i2voc, voc_encoder.pad_token, args.pos_buffer_len)
train(args, train_data, valid_datas, data_names, train2valid_r, voc_encoder, model, result_dir, device)

dist.destroy_process_group()
