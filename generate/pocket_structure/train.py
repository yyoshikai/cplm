import sys, os, math
from argparse import ArgumentParser, Namespace
import yaml
import numpy as np
from torch.utils.data import StackDataset, Subset

from src.utils import slice_str
from src.data import CacheDataset, index_dataset
from src.data.datasets.unimol import UniMolPocketDataset
from src.data.protein import PocketTokenizeDataset, Pocket
from src.data.protein import Pocket
from src.data.tokenizer import SentenceDataset, VocEncoder, TokenSplitDataset
from src.generate import generate, GeneratorStreamer, coord_streamer


class PocketStructureStreamer(GeneratorStreamer):
    def __init__(self, name: str, prompt_token_path: str, new_token_path: str, new_coord_path: str, pocket: Pocket, coord_range: float, voc_encoder: VocEncoder, no_token_range: bool, atom_order: bool, h_atom: bool, h_coord: bool):
        super().__init__(name, prompt_token_path, new_token_path, voc_encoder)
        self.pocket = pocket
        self.new_coord_path = new_coord_path

        self.coord_range = coord_range
        self.no_token_range = no_token_range
        self.atom_order = atom_order
        self.h_atom = h_atom
        self.h_coord = h_coord

    def put_generator(self):
        prompt_tokens = yield
        prompt_tokens = self.voc_encoder.decode(prompt_tokens)
        assert prompt_tokens[0] == '[POCKET]' and prompt_tokens[-1] == '[XYZ]'

        if self.h_coord:
            n_atom = len(self.pocket.atoms)
        else:
            n_atom = np.sum(slice_str(self.pocket.atoms, 1) != 'H')
        start_position = 2 if self.atom_order else len(prompt_tokens)
        coords, pos = yield from coord_streamer(n_atom, start_position, self.new_coord_path, self.voc_encoder, self.coord_range, self.no_token_range, self.atom_order)
        yield False, [pos], [self.voc_encoder.voc2i['[END]']]

if __name__ == '__main__':

    parser = ArgumentParser()
    ## training
    parser.add_argument("--studyname", required=True)
    parser.add_argument("--opt", type=int)
    ## environment
    parser.add_argument("--batch-size", type=int, default=4)
    ## generation
    parser.add_argument("--genname")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-token-range", action='store_true')
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--trial", type=int, default=5)
    ## tests
    parser.add_argument("--log-position", action='store_true')
    parser.add_argument("--log-token-range", action='store_true')
    args = parser.parse_args()

    train_dir = f"./training/results/{args.studyname}"
    targs = Namespace(**yaml.safe_load(open(f"{train_dir}/args.yaml")))
    assert targs.UniMolPocket > 0
    if getattr(targs, 'pocket_atom_order', False):
        raise NotImplementedError

    out_dir = "./generate/pocket_structure/training"+(f"/{args.genname}" if args.genname is not None else "")+f"/{args.studyname}/{args.opt}"
    assert not targs.no_pocket_heavy_atom and not targs.no_pocket_heavy_coord
    pocket = UniMolPocketDataset('valid')
    pocket = CacheDataset(pocket)
    pocket_token = PocketTokenizeDataset(pocket, heavy_atom=not targs.no_pocket_heavy_atom, heavy_coord=not targs.no_pocket_heavy_coord, h_atom=targs.pocket_h_atom, h_coord=targs.pocket_h_coord, coord_follow_atom=targs.coord_follow_atom, atom_order=getattr(targs, 'pocket_atom_order', False), coord_range=targs.coord_range)
    protein_atom_token, _coord_token = TokenSplitDataset(pocket_token, '[XYZ]').untuple()
    sentence = SentenceDataset('[POCKET]', protein_atom_token, '[XYZ]')
    index, sentence = index_dataset(sentence)
    prompt = StackDataset(index, pocket, sentence)
    if args.n < len(prompt):
        sample_idxs = np.random.default_rng(args.sample_seed).choice(len(prompt), args.n, replace=False)
        prompt = Subset(prompt, sample_idxs)

    def streamer_fn(item, i_trial: int, voc_encoder: VocEncoder):
        return PocketStructureStreamer(
            name=f"{item[0]}][{i_trial}",
            prompt_token_path=f"{out_dir}/prompt_token/{item[0]}/{i_trial}.txt",
            new_token_path=f"{out_dir}/new_token/{item[0]}/{i_trial}.txt", 
            new_coord_path=f"{out_dir}/new_coord_csv/{item[0]}/{i_trial}.csv",
            pocket=item[1],
            voc_encoder=voc_encoder, coord_range=targs.coord_range, no_token_range=args.no_token_range, atom_order=getattr(targs, 'pocket_atom_order', False), h_atom=targs.pocket_h_atom, h_coord=targs.pocket_h_coord

        )
    get_token_position_fn = lambda item: item[2]
    generate(out_dir, targs, f"{train_dir}/models/{args.opt}.pth", prompt, streamer_fn, get_token_position_fn, args.trial, 10000, None, 1, 0, args.log_position, args.log_token_range)
