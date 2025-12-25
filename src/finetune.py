import logging
from argparse import Namespace
from typing import Literal
from copy import deepcopy
import numpy as np
from torch.utils.data import Dataset
from openbabel.openbabel import OBMol
from .data import CacheDataset, RepeatDataset, Subset, StackDataset, untuple
from .data.tokenizer import FloatTokenizer, TokenizeDataset, SentenceDataset, VocEncoder, BinaryClassTokenizer, TokenEncodeDataset, TokenWeightDataset, RemoveLastDataset
from .data.datasets.targetdiff import TargetDiffScafCDDataset, TargetDiffScafCDProteinDataset
from .data.datasets.unimol import UniMolLigandDataset, UniMolLigandNoMolNetDataset, UniMolPocketDataset
from .data.datasets.crossdocked import CDDataset, CDProteinDataset
from .data.datasets.pdb import PDBUniMolRandomDataset
from .data.protein import Pocket, PocketTokenizeDataset
from .data.molecule import MolTokenizeDataset, RandomScoreDataset, RandomClassDataset
from .data.coord import CoordTransformDataset

def get_train_data(args: Namespace, split, score: Literal['none', 'cls', 'reg'], pocket_weight: float=1.0, lig_weight: float=1.0, score_weight: float=5.0):
    logs = []
    logs.append((f"{split} data actual_size/total_size=", logging.INFO))

    # compatibility
    assert isinstance(args, Namespace) # not Dict
    args = deepcopy(args)
    default_args = { 'lig_coord_follow_atom': False, 'lig_atoms': False}
    for name, value in default_args.items():
        if not hasattr(args, name):
            logs.append((f"args.{name} was not set and defaults to {value}.", logging.WARNING))
            setattr(args, name, value)

    datas = []
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
            mol = CoordTransformDataset(mol, base_seed=args.seed+d_seed, 
                normalize_coord=True, random_rotate=True, coord_noise_std=args.coord_noise_std).untuple()[0]
            mol = MolTokenizeDataset(mol, args.seed+d_seed, 
                h_atom=not args.no_lig_h_atom, h_coord=not args.no_lig_h_coord, 
                randomize=args.lig_randomize, coord_range=args.coord_range, coord_follow_atom=args.lig_coord_follow_atom, atoms=args.lig_atoms)
            
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
            
            mol_data = SentenceDataset(*sentence)
            vocs |= mol_data.vocs()
            data = CacheDataset(mol_data)
            
            ### weight
            weight_data = RemoveLastDataset(TokenWeightDataset(data, separates, weights, by_n_separate=True))
        ### Proteins
        else:
            protein = data
            protein = CoordTransformDataset(protein, base_seed=args.seed+d_seed, normalize_coord=True, random_rotate=True, coord_noise_std=args.coord_noise_std).untuple()[0]
            protein = PocketTokenizeDataset(protein, not args.no_pocket_heavy_atom, args.pocket_h_atom, not args.no_pocket_heavy_coord, args.pocket_h_coord, args.coord_follow_atom, args.coord_range)
            protein_data = SentenceDataset('[POCKET]', protein, '[END]')
            vocs |= protein_data.vocs()
            data = CacheDataset(protein_data)

            #### weight
            separates = {'[POCKET]', '[END]'}
            separates2weight = { ('[POCKET]',): pocket_weight, ('[POCKET]', '[END]'): 0.0 }
            weight_data = RemoveLastDataset(TokenWeightDataset(protein_data, separates, separates2weight))

        datas.append(data)
        weight_datas.append(weight_data)
        dnames.append(dname)
        logs.append((f"    {dname}: {len(data):,}/{len(raw):,}", logging.INFO))
        d_seed += 1
    
    ### encode words
    voc_encoder = VocEncoder(vocs)
    datas = [TokenEncodeDataset(data, voc_encoder) for data in datas]

    ### merge weight
    datas = [StackDataset(data, weight_data) for data, weight_data in zip(datas, weight_datas)]
    return datas, voc_encoder, dnames, logs
    split2datas[split] = datas

def get_finetune_data(args: Namespace, split: str, add_ligand: bool, random_rotate: bool, 
        added_vocs: set[str], prompt_score: Literal['data', 'low', 'none'], raw_data: Dataset[OBMol|Pocket]|None=None):
    logs = []

    # compatibility
    assert isinstance(args, Namespace)
    args = deepcopy(args)
    default_args = { 
        'lig_coord_follow_atom': False, 
        'lig_atoms': False, 
        'targetdiff': True
    }
    for name, value in default_args.items():
        if not hasattr(args, name):
            logs.append((f"args.{name} was not set and defaults to {value}.", logging.WARNING))
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
    protein, lig, score, protein_filename, ligand_filename = untuple(raw_data, 5)

    ## rotation
    if args.protein:
        protein, lig, *center_rotation \
            = CoordTransformDataset(protein, lig, base_seed=args.seed, normalize_coord=True, random_rotate=random_rotate).untuple()
    else:
        lig, protein, *center_rotation \
            = CoordTransformDataset(lig, protein, base_seed=args.seed, normalize_coord=True, random_rotate=random_rotate).untuple()
    center = center_rotation[0]
    rotation = center_rotation[1] if random_rotate else None


    # sentence
    separates = {'[POCKET]', '[XYZ]', '[SCORE]', '[LIGAND]', '[END]'}
    sentence = []
    weights = []
    ## pocket
    protein = PocketTokenizeDataset(protein, heavy_atom=not args.no_pocket_heavy_atom, heavy_coord=not args.no_pocket_heavy_coord, h_atom=args.pocket_h_atom, h_coord=args.pocket_h_coord, coord_follow=args.coord_follow_atom)
    sentence += ['[POCKET]', protein, '[END]']
    if args.coord_follow_atom:
        assert args.pocket_atom_weight == args.pocket_coord_weight
        weights += [None, args.pocket_coord_weight, 0.0]
    else:
        weights += [None, args.pocket_atom_weight, args.pocket_coord_weight, 0.0]
    ## score
    assert prompt_score in ['none', 'data', 'low']
    if prompt_score != 'none':
        if prompt_score == 'low':
            score = RandomScoreDataset(-12.0, -10.0, len(protein), args.seed)
        score_tokenizer = FloatTokenizer('score', -args.coord_range, args.coord_range)
        score = TokenizeDataset(score, score_tokenizer)
        sentence += ['[SCORE]', score, '[END]']
        weights += [0.0, 0.0]
        
    ## ligand
    sentence.append('[LIGAND]')
    weights.append(args.lig_smiles_weight)
    if add_ligand:
        lig = MolTokenizeDataset(lig, args.seed, h_atom=not args.no_lig_h_atom, h_coord=not args.no_lig_h_coord, randomize=args.lig_randomize, coord_range=args.coord_range, coord_follow_atom=args.lig_coord_follow_atom, atoms=args.lig_atoms)
        sentence += [lig, '[END]']
        weights += [args.lig_coord_weight, 0.0]
    sentence = SentenceDataset(*sentence)
    vocs = sentence.vocs() | added_vocs
    sentence = CacheDataset(sentence)

    ## token
    voc_encoder = VocEncoder(vocs)
    token = TokenEncodeDataset(sentence, voc_encoder)
  
    ## weight
    weight = RemoveLastDataset(TokenWeightDataset(sentence, separates, weights, by_n_separate=True))
    return voc_encoder, raw_data, token, weight, center, rotation, protein_filename, ligand_filename, logs
