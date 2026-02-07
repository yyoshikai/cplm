import logging
from argparse import Namespace
from typing import Literal
from copy import deepcopy
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from openbabel.openbabel import OBMol
from ..data import RepeatDataset, Subset, StackDataset, TensorDataset, untuple_dataset
from ..data.tokenizer import FloatTokenizer, TokenizeDataset, SentenceDataset, VocEncoder, BinaryClassTokenizer, TokenEncodeDataset, TokenWeightDataset, RemoveLastDataset
from ..data.datasets.targetdiff import TargetDiffScafCDDataset, TargetDiffScafCDProteinDataset
from ..data.datasets.unimol import UniMolLigandDataset, UniMolLigandNoMolNetDataset, UniMolPocketDataset
from ..data.datasets.crossdocked import CDDataset, CDProteinDataset
from ..data.datasets.pdb import PDBUniMolRandomDataset
from ..data.protein import Pocket, PocketTokenizeDataset, ProteinTokenizeDataset
from ..data.molecule import MolProcessDataset, MolTokenizeDataset, RandomScoreDataset, RandomClassDataset
from ..data.coord import CoordTransformDataset

def get_train_data(args: Namespace, split, score: Literal['none', 'cls', 'reg'], pocket_weight: float=1.0, lig_weight: float=1.0, score_weight: float=5.0):
    logs = []

    # compatibility
    assert isinstance(args, Namespace) # not Dict
    args = deepcopy(args)
    default_args = { 'lig_coord_follow_atom': False, 'lig_atoms': False, 'pocket_atom_order': False, 'lig_atom_order': False}
    for name, value in default_args.items():
        if not hasattr(args, name):
            logs.append((f"args.{name} was not set and defaults to {value}.", logging.WARNING))
            setattr(args, name, value)

    logs.append((f"{split} data actual_size/total_size=", logging.INFO))
    token_datas = []
    position_datas = []
    weight_datas = []
    dnames = []
    vocs = set()
    ## Molecule
    for d_seed, cls in enumerate([UniMolLigandDataset, UniMolLigandNoMolNetDataset, UniMolPocketDataset, PDBUniMolRandomDataset]):

        dname = cls.__name__.removesuffix('Dataset')
        repeat = getattr(args, dname)
        if repeat == 0: continue
        
        raw = cls(split='valid' if 'data_epoch' in args.check else split)
        ## repeat / sample
        data = raw
        if split == 'train' and repeat != 1:
            data = RepeatDataset(data, repeat)
        sample = getattr(args, dname+'_val_sample')
        if (split == 'valid' or 'data_epoch' in args.check) and sample < 1.0:
            rng = np.random.default_rng(args.seed+d_seed)
            idxs = rng.choice(len(data), size=round(len(data)*sample))
            data = Subset(data, idxs)
            assert len(data) > 0
        
        ## process
        ### Molecules
        if cls in [UniMolLigandDataset, UniMolLigandNoMolNetDataset]:
            mol = data
            mol = MolProcessDataset(mol, args.seed+d_seed, not args.no_lig_h_atom, args.lig_randomize)
            mol = CoordTransformDataset(mol, base_seed=args.seed+d_seed, 
                normalize_coord=True, random_rotate=True, coord_noise_std=args.coord_noise_std).untuple()[0]
            mol = MolTokenizeDataset(mol, coord_follow_atom=args.lig_coord_follow_atom, atoms=args.lig_atoms, atom_order=args.lig_atom_order, coord_range=args.coord_range, no_h_coord=args.no_lig_h_coord)
            
            ### sentence
            sentence = ['[LIGAND]', mol, '[END]']
            separates = {'[LIGAND]', '[SCORE]', '[END]'}
            weights = [None, lig_weight, 0.0, ]
            if score != 'none':
                if score == 'cls':
                    score = RandomClassDataset(len(mol), args.seed+d_seed)
                    score = TokenizeDataset(score, BinaryClassTokenizer())
                elif score == 'reg':
                    score = RandomScoreDataset(-50, 50, len(mol), args.seed+d_seed)
                    score = TokenizeDataset(score, FloatTokenizer('score', -args.coord_range, args.coord_range))
                else:
                    raise ValueError
                sentence += ['[SCORE]', score, '[END]']
                weights += [score_weight, 0.0]
            
            sentence = SentenceDataset(*sentence)
            vocs |= sentence.vocs()
            token, position = sentence.untuple()
            position = TensorDataset(position, torch.long)
            
            ### weight
            weight = RemoveLastDataset(TokenWeightDataset(token, separates, weights, by_n_separate=True))
        ### Proteins
        else:
            protein = data
            protein = CoordTransformDataset(protein, base_seed=args.seed+d_seed, normalize_coord=True, random_rotate=True, coord_noise_std=args.coord_noise_std).untuple()[0]
            if cls == UniMolPocketDataset:
                protein = PocketTokenizeDataset(protein, heavy_atom=not args.no_pocket_heavy_atom, heavy_coord=not args.no_pocket_heavy_coord, h_atom=args.pocket_h_atom, h_coord=args.pocket_h_coord, coord_follow_atom=args.coord_follow_atom, atom_order=args.pocket_atom_order, coord_range=args.coord_range)
            else:
                protein = ProteinTokenizeDataset(protein, heavy_atom=not args.no_pocket_heavy_atom, heavy_coord=not args.no_pocket_heavy_coord, h_atom=args.pocket_h_atom, h_coord=args.pocket_h_coord, coord_follow_atom=args.coord_follow_atom, atom_order=args.pocket_atom_order, coord_range=args.coord_range)
            sentence = SentenceDataset('[POCKET]', protein, '[END]')
            vocs |= sentence.vocs()
            token, position = sentence.untuple()
            position = TensorDataset(position, torch.long)

            #### weight
            separates = {'[POCKET]', '[END]'}
            separates2weight = { ('[POCKET]',): pocket_weight, ('[POCKET]', '[END]'): 0.0 }
            weight = RemoveLastDataset(TokenWeightDataset(token, separates, separates2weight))

        token_datas.append(token)
        position_datas.append(position)
        weight_datas.append(weight)
        dnames.append(dname)
        logs.append((f"    {dname}: {len(data):,}/{len(raw):,}", logging.INFO))
        d_seed += 1
    
    ### encode words
    voc_encoder = VocEncoder(vocs)
    token_datas = [TokenEncodeDataset(token, voc_encoder) for token in token_datas]

    ### merge weight
    datas = [StackDataset(token, position, weight) for token, position, weight
            in zip(token_datas, position_datas, weight_datas)]
    return datas, voc_encoder, dnames, logs

def get_finetune_data(args: Namespace, split: str, sample: float, add_ligand: bool, random_rotate: bool, 
        added_vocs: set[str], prompt_score: Literal['data', 'low', 'none'], raw_data: Dataset[OBMol|Pocket]|None=None, encode: bool=True, tensor_position: bool=True):
    logs = []

    # compatibility
    assert isinstance(args, Namespace) # Dict can't check hasattr
    args = deepcopy(args)
    default_args = { 
        'lig_coord_follow_atom': False, 
        'lig_atoms': False, 
        'targetdiff': True, 
        'pocket_atom_order': False, 
        'lig_atom_order': False
    }
    for name, value in default_args.items():
        if not hasattr(args, name):
            logs.append((f"args.{name} was not set and defaults to {value}.", logging.INFO))
            setattr(args, name, value)

    # raw data
    if raw_data is None:
        if args.targetdiff:
            if args.protein:
                raw_data = TargetDiffScafCDProteinDataset(split)
            else:
                raw_data = TargetDiffScafCDDataset(split)
        else:
            if args.protein:
                raw_data = CDProteinDataset(split)
            else:
                raw_data = CDDataset(split)
    if sample != 1.0:
        assert sample < 1.0
        rng = np.random.default_rng(args.seed)
        idxs = rng.choice(len(raw_data), size=round(len(raw_data)*sample))
        raw_data = Subset(raw_data, idxs)
        assert len(raw_data) > 0
    protein, lig, score = untuple_dataset(raw_data, 3)

    ## rotation
    if args.protein:
        protein, lig, *center_rotation \
            = CoordTransformDataset(protein, lig, base_seed=args.seed, normalize_coord=True, random_rotate=random_rotate).untuple()
    else:
        lig, protein, *center_rotation \
            = CoordTransformDataset(lig, protein, base_seed=args.seed, normalize_coord=True, random_rotate=random_rotate).untuple()
    center = center_rotation[0]


    # sentence
    separates = {'[POCKET]', '[XYZ]', '[SCORE]', '[LIGAND]', '[END]'}
    sentence = []
    weights = []
    ## pocket
    if args.protein:
        protein_tokens = ProteinTokenizeDataset(protein, heavy_atom=not args.no_pocket_heavy_atom, heavy_coord=not args.no_pocket_heavy_coord, h_atom=args.pocket_h_atom, h_coord=args.pocket_h_coord, coord_follow_atom=args.coord_follow_atom, atom_order=args.pocket_atom_order, coord_range=args.coord_range)
    else:
        protein_tokens = PocketTokenizeDataset(protein, heavy_atom=not args.no_pocket_heavy_atom, heavy_coord=not args.no_pocket_heavy_coord, h_atom=args.pocket_h_atom, h_coord=args.pocket_h_coord, coord_follow_atom=args.coord_follow_atom, atom_order=args.pocket_atom_order, coord_range=args.coord_range)
    sentence += ['[POCKET]', protein_tokens, '[END]']
    if args.coord_follow_atom:
        assert args.pocket_atom_weight == args.pocket_coord_weight
        weights += [None, args.pocket_coord_weight, 0.0]
    else:
        weights += [None, args.pocket_atom_weight, args.pocket_coord_weight, 0.0]
    ## score
    assert prompt_score in ['none', 'data', 'low']
    if prompt_score != 'none':
        if prompt_score == 'low':
            score = RandomScoreDataset(-12.0, -10.0, len(protein_tokens), args.seed)
        score_tokenizer = FloatTokenizer('score', -args.coord_range, args.coord_range)
        score = TokenizeDataset(score, score_tokenizer)
        sentence += ['[SCORE]', score, '[END]']
        weights += [0.0, 0.0]
        
    ## ligand
    sentence.append('[LIGAND]')
    weights.append(args.lig_smiles_weight)
    lig = MolProcessDataset(lig, args.seed, not args.no_lig_h_atom, args.lig_randomize)
    if add_ligand:
        lig_tokens = MolTokenizeDataset(lig, coord_follow_atom=args.lig_coord_follow_atom, atoms=args.lig_atoms, atom_order=args.lig_atom_order, coord_range=args.coord_range, no_h_coord=args.no_lig_h_coord)
        sentence += [lig_tokens, '[END]']
        weights += [args.lig_coord_weight, 0.0]
    sentence = SentenceDataset(*sentence)
    vocs = sentence.vocs() | added_vocs
    token, position = sentence.untuple()
    if tensor_position:
        position = TensorDataset(position, torch.long)

    ## weight
    weight = RemoveLastDataset(TokenWeightDataset(token, separates, weights, by_n_separate=True))

    ## encode token
    if encode:
        voc_encoder = VocEncoder(vocs)
        token = TokenEncodeDataset(token, voc_encoder)
    else:
        voc_encoder = None

    return voc_encoder, raw_data, protein, lig, token, position, weight, center, logs

## Define functions for batch collation
def collate(data_list: list[tuple[Tensor, Tensor, Tensor]], pad_token):
    tokens, positions, weights = zip(*data_list)
    tokens = pad_sequence(tokens, padding_value=pad_token)
    positions = pad_sequence(positions, padding_value=0)
    weights = pad_sequence(weights, padding_value=0.0)
    return tokens, positions, weights


