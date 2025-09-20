import sys, os
import argparse

import numpy as np
from torch.utils.data import StackDataset, Subset
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
from src.train import train, add_train_args, set_default_args

# arguments
parser = argparse.ArgumentParser()
## settings
add_train_args(parser)
## dataset
for cls in [UniMolLigandNoMolNetDataset, UniMolPocketDataset, PDBUniMolRandomDataset]:
    dname = cls.__name__.removesuffix('Dataset')
    parser.add_argument(f'--{dname}', type=int, default=0)
    parser.add_argument(f'--{dname}-val-sample', type=float, default=1.0)
args = parser.parse_args()
set_default_args(args)

# data
smiles_tokenizer = StringTokenizer(open("src/data/smiles_tokens.txt").read().splitlines())
mol_coord_tokenizer = FloatTokenizer('ligand', -args.coord_range, args.coord_range, log_interval=args.tokenizer_log_interval)
pocket_coord_tokenizer = FloatTokenizer('pocket', -args.coord_range, args.coord_range, log_interval=args.tokenizer_log_interval)
pocket_atom_tokenizer = ProteinAtomTokenizer(log_interval=args.tokenizer_log_interval)


# datasets
vocs = set()
split2datas = {}
split2datas_to_log = {split: [] for split in ['valid', 'train']}
for split in ['valid', 'train']:
    
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
        split2datas_to_log[split].append(raw)
        
        ## process
        ### Molecules
        if cls in [UniMolLigandNoMolNetDataset]:
            mol = raw
            smi, coord = MolProcessDataset(mol, args.seed+d_seed, 
                h_atom=not args.no_lig_h_atom, h_coord=not args.no_lig_h_coord, 
                randomize=args.lig_randomize).untuple()
            coord = CoordTransformDataset(coord, base_seed=args.seed+d_seed, 
                normalize_coord=True, random_rotate=True, coord_noise_std=args.coord_noise_std).untuple()[0]

            ### sentence
            smi = TokenizeDataset(smi, smiles_tokenizer)
            coord = ArrayTokenizeDataset(coord, mol_coord_tokenizer)
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

            ### setntence
            atoms = TokenizeDataset(atoms, pocket_atom_tokenizer)
            coord = ArrayTokenizeDataset(coord, pocket_coord_tokenizer)
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

    ### encode words
    voc_encoder = VocEncoder(vocs)
    datas = [TokenEncodeDataset(data, voc_encoder) for data in datas]

    ### merge weight
    datas = [StackDataset(data, weight_data) for data, weight_data in zip(datas, weight_datas)]
    split2datas[split] = datas

train('training', args, split2datas['train'], split2datas['valid'], split2datas_to_log['train'], split2datas_to_log['valid'], voc_encoder)

