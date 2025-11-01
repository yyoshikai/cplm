from argparse import Namespace
from typing import Literal
import numpy as np
from .data import CacheDataset, RepeatDataset, Subset, StackDataset, untuple
from .data.tokenizer import StringTokenizer, FloatTokenizer, \
    ProteinAtomTokenizer, TokenizeDataset, ArrayTokenizeDataset, \
    SentenceDataset, VocEncoder, BinaryClassTokenizer, TokenEncodeDataset, TokenWeightDataset, RemoveLastDataset
from .data.datasets.targetdiff import TargetDiffScafCDDataset, TargetDiffScafCDProteinDataset
from .data.datasets.unimol import UniMolLigandDataset, UniMolLigandNoMolNetDataset, UniMolPocketDataset
from .data.datasets.crossdocked import CDDataset, CDProteinDataset
from .data.datasets.pdb import PDBUniMolRandomDataset
from .data.protein import ProteinProcessDataset
from .data.molecule import MolProcessDataset, RandomScoreDataset, RandomClassDataset
from .data.coord import CoordTransformDataset
from .data.tokenizer import TokenEncodeDataset, VocEncoder, \
        ProteinAtomTokenizer, FloatTokenizer, StringTokenizer
from .data.protein import CoordFollowDataset
from .utils.path import WORKDIR

def get_train_data(args, split, score: Literal['none', 'cls', 'reg'], score_weight: float|None=None):
    logs = []
    logs.append(f"{split} data actual_size/total_size=")

    smiles_tokenizer = StringTokenizer(open("src/data/smiles_tokens.txt").read().splitlines())
    mol_coord_tokenizer = FloatTokenizer('ligand', -args.coord_range, args.coord_range, log_interval=args.tokenizer_log_interval)
    pocket_coord_tokenizer = FloatTokenizer('pocket', -args.coord_range, args.coord_range, log_interval=args.tokenizer_log_interval)
    pocket_atom_tokenizer = ProteinAtomTokenizer(log_interval=args.tokenizer_log_interval)

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
            smi, coord = MolProcessDataset(mol, args.seed+d_seed, 
                h_atom=not args.no_lig_h_atom, h_coord=not args.no_lig_h_coord, 
                randomize=args.lig_randomize).untuple()
            coord = CoordTransformDataset(coord, base_seed=args.seed+d_seed, 
                normalize_coord=True, random_rotate=True, coord_noise_std=args.coord_noise_std).untuple()[0]

            ### sentence
            smi = TokenizeDataset(smi, smiles_tokenizer)
            coord = ArrayTokenizeDataset(coord, mol_coord_tokenizer)
            sentence = ['[LIGAND]', smi, '[XYZ]', coord, '[END]']
            weights = [None, 1.0, 0.0]
            if score == 'none':
                assert score_weight is None
            else:
                assert score_weight is not None
                if score == 'cls':
                    score = RandomClassDataset(len(smi), args.seed+d_seed)
                    score = TokenizeDataset(score, BinaryClassTokenizer())
                elif score == 'reg':
                    score = RandomScoreDataset(-50, 50, len(smi), args.seed+d_seed)
                    score = TokenizeDataset(score, FloatTokenizer('score', -args.coord_range, args.coord_range, 
                            log_interval=args.tokenizer_log_interval))
                else:
                    raise ValueError
                sentence += ['[SCORE]', score, '[END]']
                weights += [score_weight, 0.0]
                
            mol_data = SentenceDataset(*sentence)
            vocs |= mol_data.vocs()
            
            ### weight
            data = CacheDataset(mol_data)
            separates = {'[LIGAND]', '[SCORE]', '[END]'}
            weight_data = RemoveLastDataset(TokenWeightDataset(mol_data, separates, weights, by_n_separate=True))
        ### Pockets
        else:
            pocket = data
            atoms, coord, coord_position = ProteinProcessDataset(pocket, heavy_atom=not args.no_pocket_heavy_atom, heavy_coord=not args.no_pocket_heavy_coord, h_atom=args.pocket_h_atom, h_coord=args.pocket_h_coord).untuple()

            coord = CoordTransformDataset(coord, base_seed=args.seed+d_seed, normalize_coord=True, random_rotate=True, coord_noise_std=args.coord_noise_std).untuple()[0]

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
        dnames.append(dname)
        logs.append(f"    {dname}: {len(data):,}/{len(raw):,}")
        d_seed += 1
    
    ### encode words
    voc_encoder = VocEncoder(vocs)
    datas = [TokenEncodeDataset(data, voc_encoder) for data in datas]

    ### merge weight
    datas = [StackDataset(data, weight_data) for data, weight_data in zip(datas, weight_datas)]
    return datas, voc_encoder, dnames, logs
    split2datas[split] = datas


def get_finetune_data(args: Namespace, split: str, add_ligand: bool, random_rotate: bool, 
        added_vocs: set[str], prompt_score: Literal['data', 'low', 'none']):
    
    # tokenizer
    coord_tokenizer = FloatTokenizer('ligand', -args.coord_range, args.coord_range, log_interval=args.tokenizer_log_interval)
    protein_coord_tokenizer = FloatTokenizer('pocket', -args.coord_range, args.coord_range, log_interval=args.tokenizer_log_interval)
    protein_atom_tokenizer = ProteinAtomTokenizer(log_interval=args.tokenizer_log_interval)

    # raw data
    if getattr(args, 'targetdiff', True):
        if args.protein:
            raw_data = TargetDiffScafCDProteinDataset(split)
        else:
            raw_data = TargetDiffScafCDDataset(split)
    else:
        if args.protein:
            raise NotImplementedError # 実装する
        else:
            raw_data = CDDataset(split)
    protein, lig, score, protein_filename, ligand_filename = untuple(raw_data, 5)

    lig_smi, lig_coord = MolProcessDataset(lig, args.seed, h_atom=not args.no_lig_h_atom, h_coord=not args.no_lig_h_coord, randomize=args.lig_randomize).untuple()
    pocket_atom, pocket_coord, pocket_coord_position = ProteinProcessDataset(protein, heavy_atom=not args.no_pocket_heavy_atom, heavy_coord=not args.no_pocket_heavy_coord, h_atom=args.pocket_h_atom, h_coord=args.pocket_h_coord).untuple()

    if args.protein:
        pocket_coord, lig_coord, center, rotation \
            = CoordTransformDataset(pocket_coord, lig_coord, base_seed=args.seed, normalize_coord=True, random_rotate=random_rotate).untuple()
    else:
        lig_coord, pocket_coord, center, rotation \
            = CoordTransformDataset(lig_coord, pocket_coord, base_seed=args.seed, normalize_coord=True, random_rotate=random_rotate).untuple()

    # sentence
    separates = {'[POCKET]', '[XYZ]', '[SCORE]', '[LIGAND]', '[END]'}
    ## pocket
    pocket_atom = TokenizeDataset(pocket_atom, protein_atom_tokenizer)
    pocket_coord = ArrayTokenizeDataset(pocket_coord, protein_coord_tokenizer)
    if args.coord_follow_atom:
        assert args.pocket_atom_weight == args.pocket_coord_weight
        sentence = ['[POCKET]', CoordFollowDataset(pocket_atom, pocket_coord, pocket_coord_position), '[END]']
        weights = [None, args.pocket_coord_weight, 0.0]
    else:
        sentence = ['[POCKET]', pocket_atom, '[XYZ]', pocket_coord, '[END]']
        weights = [None, args.pocket_atom_weight, args.pocket_coord_weight, 0.0]
    ## score
    assert prompt_score in ['none', 'data', 'low']
    if prompt_score != 'none':
        if prompt_score == 'low':
            score = RandomScoreDataset(-12.0, -10.0, len(pocket_atom), args.seed)
        score_tokenizer = FloatTokenizer('score', -args.coord_range, args.coord_range, log_interval=args.tokenizer_log_interval)
        score = TokenizeDataset(score, score_tokenizer)
        sentence += ['[SCORE]', score, '[END]']
        weights += [0.0, 0.0]
        
    ## ligand
    sentence.append('[LIGAND]')
    weights.append(args.lig_smiles_weight)
    if add_ligand:
        lig_smi = TokenizeDataset(lig_smi, StringTokenizer(open(f"{WORKDIR}/cplm/src/data/smiles_tokens.txt").read().splitlines()))
        lig_coord = ArrayTokenizeDataset(lig_coord, coord_tokenizer)
        sentence += [lig_smi, '[XYZ]', lig_coord, '[END]']
        weights += [args.lig_coord_weight, 0.0]
    sentence = SentenceDataset(*sentence)
    vocs = sentence.vocs() | added_vocs
    sentence = CacheDataset(sentence)

    ## token
    voc_encoder = VocEncoder(vocs)
    token = TokenEncodeDataset(sentence, voc_encoder)
  
    ## weight
    weight = RemoveLastDataset(TokenWeightDataset(sentence, separates, weights, by_n_separate=True))
    return voc_encoder, raw_data, token, weight, center, rotation, protein_filename, ligand_filename
