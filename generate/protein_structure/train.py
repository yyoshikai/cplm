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
from openbabel.openbabel import OBMol, OBResidueIter, OBResidueAtomIter, OBMolAtomIter, OBConversion

from src.utils.logger import get_logger, add_file_handler, set_third_party_logger
from src.utils.rdkit import ignore_rdkit_warning
from src.utils import slice_str
from src.data import CacheDataset
from src.data.datasets.pdb import PDBUniMolRandomDataset
from src.data.protein import ProteinTokenizeDataset
from src.data.tokenizer import SentenceDataset, VocEncoder, TokenEncodeDataset, TokenSplitDataset, FloatTokenizer
from src.train import get_model
from src.evaluate import parse_coord_tokens
from src.generate import generate, GeneratorStreamer

def coord_streamer(n_atom: int, start_position: int, voc_encoder: VocEncoder, coord_range: float, no_token_range: bool) -> Generator[tuple[bool, list[int], list[int]]]:
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
            return None
        coords.append(coord)
    coords = np.array(coords).reshape(-1, 3)
    return coords

class ProteinStructureStreamer(GeneratorStreamer):
    def __init__(self, voc_encoder: VocEncoder, out_token_dir: str, protein: OBMol, out_pdb_path: str, coord_range: float, no_token_range: bool, h_atom: bool, h_coord: bool):
        super().__init__(voc_encoder, out_token_dir)
        self.protein = protein
        self.coord_range = coord_range
        self.no_token_range = no_token_range
        self.out_pdb_path = out_pdb_path
        self.h_atom = h_atom
        self.h_coord = h_coord

    def put_generator(self) -> Generator[tuple[bool, list[int], list[int]], list[int], None]:
        prompt_tokens = yield
        prompt_tokens = self.voc_encoder.decode(prompt_tokens)
        assert prompt_tokens[-1] == '[XYZ]'
        self.is_prompt = False
        if self.h_atom:
            self.protein.AddHydrogens()
        else:
            self.protein.DeleteHydrogens()
        atoms = np.array([atom.GetResidue().GetAtomID(atom).strip() for atom in OBMolAtomIter(protein)])
        residue_idxs = np.array([atom.GetResidue().GetIdx() for atom in OBMolAtomIter(protein)])
        orders = np.argsort(residue_idxs, kind='stable')
        if not self.h_coord:
            atoms_ordered = atoms[orders]
            is_h = slice_str(atoms_ordered, 1) == 'H'
            orders = orders[~is_h]

        coords = yield from coord_streamer(len(orders), len(prompt_tokens), self.voc_encoder, self.coord_range, self.no_token_range)
        if coords is not None:

            for i_coord, order in enumerate(orders):
                atom = self.protein.GetAtom(order)
                atom.SetVector(coords[i_coord, 0], coords[i_coord, 1], coords[i_coord, 2])
            if not self.h_coord:
                self.protein.DeleteHydrogens()
            os.makedirs(os.path.dirname(self.out_pdb_path), exist_ok=True)
            obc = OBConversion()
            obc.WriteFile(self.protein, self.out_pdb_path)
        yield False, [len(prompt_tokens)+len(orders)*6], [self.voc_encoder.voc2i['[END]']]

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
    parser.add_argument("--no-token-range", action='store_true')
    args = parser.parse_args()

    train_dir = f"./training/results/{args.studyname}"
    targs = Namespace(**yaml.safe_load(open(f"{train_dir}/args.yaml")))
    assert targs.PDBUniMolRandom > 0
    out_dir = "./generate/protein_structure/training"+(f"/{args.genname}" if args.genname is not None else "")+f"/{args.studyname}/{args.opt}"
    assert args.pocket_heavy_atom and args.pocket_heavy_coord

    protein = PDBUniMolRandomDataset('valid')
    protein = CacheDataset(protein)
    protein_token = ProteinTokenizeDataset(protein, heavy_atom=not targs.no_pocket_heavy_atom, heavy_coord=not targs.no_pocket_heavy_coord, h_atom=targs.pocket_h_atom, h_coord=targs.pocket_h_coord, coord_follow_atom=targs.coord_follow_atom, atom_order=getattr(targs, 'pocket_atom_order', False), coord_range=targs.coord_range)
    protein_atom_token = TokenSplitDataset(protein_token, '[XYZ]')
    sentence = SentenceDataset('[POCKET]', protein_atom_token, '[XYZ]')
    prompt = StackDataset(protein, sentence)
    streamer_fn = lambda item, i_trial, voc_encoder: ProteinStructureStreamer(voc_encoder, f"./{out_dir}/generate/{item}/{i_trial}", item[0], f"./{out_dir}/pdb/{item}/{i_trial}.pdb", targs.coord_range, args.no_token_range)
    get_token_position_fn = lambda item: (item[1], list(range(len(item[1]))))

    generate(out_dir, targs, f"{train_dir}/models/{args.opt}.pth", prompt, streamer_fn, get_token_position_fn, 5, None, 10000, None, 1, 0)




