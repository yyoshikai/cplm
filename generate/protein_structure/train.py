import sys, os, math
import itertools as itr
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from collections.abc import Generator
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, StackDataset
from torch.nn.utils.rnn import pad_sequence
from openbabel.openbabel import OBMol, OBResidueIter, OBResidueAtomIter

from src.utils.logger import get_logger, add_file_handler, set_third_party_logger
from src.utils.rdkit import ignore_rdkit_warning
from src.data import WrapDataset, index_dataset, TensorDataset
from src.data.datasets.pdb import PDBUniMolRandomDataset
from src.data.protein import ProteinTokenizeDataset
from src.data.tokenizer import SentenceDataset, VocEncoder, TokenEncodeDataset, TokenSplitDataset, FloatTokenizer
from src.train import get_model
from src.evaluate import parse_coord_tokens
from src.generate import generate, GeneratorStreamer

def coord_streamer(n_atom: int, next_position: list[int], voc_encoder: VocEncoder, coord_range: float, no_token_range: bool) -> Generator[tuple[bool, list[int], list[int]]]:
    coord_tokenizer = FloatTokenizer('', -coord_range, coord_range)
    if no_token_range:
        int_token_range = frac_token_range = list(range(voc_encoder.voc_size))
    else:
        int_token_range = sorted(voc_encoder.encode(coord_tokenizer.int_vocs()))
        frac_token_range = sorted(voc_encoder.encode(coord_tokenizer.frac_vocs()))
    pos_iter = itr.count(start_position)

    coords = []
    for i in range(n_atom*3):
        int_token = yield True, [next(pos_iter)], int_token_range
        frac_token = yield True, [next(pos_iter)], frac_token_range
        try:
            coord = float(''.join(voc_encoder.decode(int_token+frac_token)))
        except Exception:
            yield False, [next(pos_iter)], [voc_encoder.voc2i['[END]']]
            return None
        coords.append(coord)
    coords = np.array(coords).reshape(-1, 3)
    return coords

class ProteinStructureStreamer(GeneratorStreamer):
    def __init__(self, voc_encoder: VocEncoder, out_token_dir: str, protein: OBMol, out_pdb_path: str, coord_range: float, no_token_range: bool):
        super().__init__(voc_encoder, out_token_dir)
        self.protein = protein
        coord_tokenizer = FloatTokenizer

    def put_generator(self) -> Generator[tuple[bool, list[int], list[int]], list[int], None]:
        prompt_tokens = yield
        prompt_tokens = self.voc_encoder.decode(prompt_tokens)
        assert prompt_tokens[-1] == '[XYZ]'
        self.is_prompt = False
        start_position = len(prompt_tokens)

        n_atom = self.protein.NumAtoms()
        coords = yield from coord_streamer(self.protein.NumAtoms(), )




if __name__ == '__main__':
    parser = ArgumentParser()
    ## training
    parser.add_argument("--studyname", required=True)
    parser.add_argument("--opt", type=int)
    parser.add_argument("--batch-size", type=int, required=True)
    ## environment
    parser.add_argument("--batch-size", type=int, default=4)
    ## generation
    parser.add_argument("--genname")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=0)
    args = parser.parse_args()

    train_dir = f"./training/results/{args.studyname}"
    targs = Namespace(**yaml.safe_load(open(f"{train_dir}/args.yaml")))
    assert targs.PDBUniMolRandom > 0
    out_dir = "./generate/protein_structure/training"+(f"/{args.genname}" if args.genname is not None else "")+f"/{args.studyname}/{args.opt}"

    protein = PDBUniMolRandomDataset('valid')

    protein_token = ProteinTokenizeDataset(protein, heavy_atom=not targs.no_pocket_heavy_atom, heavy_coord=not targs.no_pocket_heavy_coord, h_atom=targs.pocket_h_atom, h_coord=targs.pocket_h_coord, coord_follow_atom=targs.coord_follow_atom, atom_order=getattr(targs, 'pocket_atom_order', False), coord_range=targs.coord_range)
    protein_atom_token = TokenSplitDataset(protein_token, '[XYZ]')
    prompt_data = SentenceDataset('[POCKET]', protein_atom_token, '[XYZ]')


    generate(out_dir, targs, f"{train_dir}/models/{args.opt}.pth", prompt_data, )




