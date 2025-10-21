from argparse import Namespace
from typing import Literal
from .data import CacheDataset
from .data.tokenizer import StringTokenizer, FloatTokenizer, \
    ProteinAtomTokenizer, TokenizeDataset, ArrayTokenizeDataset, \
    SentenceDataset, VocEncoder, TokenEncodeDataset, TokenWeightDataset, RemoveLastDataset
from .data.datasets.targetdiff import TargetDiffScafCDDataset, TargetDiffScafCDProteinDataset
from .data.protein import ProteinProcessDataset
from .data.molecule import MolProcessDataset, RandomScoreDataset
from .data.coord import CoordTransformDataset
from .data.tokenizer import TokenEncodeDataset, VocEncoder, \
        ProteinAtomTokenizer, FloatTokenizer, StringTokenizer
from .data.protein import CoordFollowDataset
from .utils.path import WORKDIR


def get_finetune_data(args: Namespace, split: str, add_ligand: bool, random_rotate: bool, 
        added_vocs: set[str], prompt_score: Literal['data', 'low', 'none']):
    
    # tokenizer
    coord_tokenizer = FloatTokenizer('ligand', -args.coord_range, args.coord_range, log_interval=args.tokenizer_log_interval)
    protein_coord_tokenizer = FloatTokenizer('pocket', -args.coord_range, args.coord_range, log_interval=args.tokenizer_log_interval)
    protein_atom_tokenizer = ProteinAtomTokenizer(log_interval=args.tokenizer_log_interval)

    # raw data
    if args.protein:
        raw_data = TargetDiffScafCDProteinDataset(split)
    else:
        raw_data = TargetDiffScafCDDataset(split)
    protein, lig, score, protein_filename, ligand_filename = raw_data.untuple()

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
