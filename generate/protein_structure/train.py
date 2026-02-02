import sys, os, math
import itertools as itr
from argparse import ArgumentParser, Namespace
from collections.abc import Generator
import yaml
import numpy as np
import pandas as pd
import openbabel.openbabel as ob
from torch.utils.data import StackDataset, Subset

from src.utils.path import make_pardir
from src.data import CacheDataset, index_dataset
from src.data.datasets.pdb import PDBUniMolRandomDataset
from src.data.protein import ProteinTokenizeDataset
from src.data.tokenizer import SentenceDataset, VocEncoder, TokenSplitDataset
from src.generate import generate
from src.generate.streamer import coord_streamer, GeneratorStreamer, TokenWriteStreamer, TimeLogStreamer

class ProteinStructureStreamer(GeneratorStreamer):
    def __init__(self, prompt_pdb_path: str, prompt_atom_path: str, new_pdb_path: str, new_coord_path: str, protein: ob.OBMol, coord_range: float, voc_encoder: VocEncoder, no_token_range: bool, atom_order: bool, h_atom: bool, h_coord: bool):
        super().__init__()
        self.voc_encoder = voc_encoder
        self.prompt_pdb_path = prompt_pdb_path
        self.prompt_atom_path = prompt_atom_path
        self.coord_range = coord_range
        self.no_token_range = no_token_range
        self.new_pdb_path = new_pdb_path
        self.new_coord_path = new_coord_path
        self.protein = protein
        self.atom_order = atom_order
        self.h_atom = h_atom
        self.h_coord = h_coord
        self.n_generated_atom = None
    def estimated_n_token(self):
        if self.n_generated_atom is not None:
            return self.n_generated_atom*6
        else:
            return None
        
    def put_generator(self) -> Generator[tuple[bool, list[int], list[int]], list[int], None]:
        obc = ob.OBConversion()
        obc.SetOutFormat('pdb')
        prompt_tokens = yield
        prompt_tokens = self.voc_encoder.decode(prompt_tokens)
        assert prompt_tokens[0] == '[POCKET]' and prompt_tokens[-1] == '[XYZ]'
        if self.h_atom:
            self.protein.AddHydrogens()
        else:
            self.protein.DeleteHydrogens()
        
        make_pardir(self.prompt_pdb_path)
        obc.WriteFile(self.protein, self.prompt_pdb_path)
        
        atom_data = []
        for atom in ob.OBMolAtomIter(self.protein):
            residue = atom.GetResidue()
            atom_data.append((residue.GetIdx(), residue.GetName(), residue.GetAtomID(atom).strip(), ob.GetSymbol(atom.GetAtomicNum())))
        df = pd.DataFrame(atom_data, columns=['residue_idx', 'residue', 'name', 'element'])
        orders = np.argsort(df['residue_idx'].values, kind='stable')
        if not self.h_coord:
            orders = orders[df['element'][orders] != 'H']
        df['order'] = -1
        df.loc[orders, 'order'] = np.arange(len(orders))
        make_pardir(self.prompt_atom_path)
        df.to_csv(self.prompt_atom_path, index_label='atom_idx')
        self.n_generated_atom = len(orders)

        start_position = 2 if self.atom_order else len(prompt_tokens)
        coords, pos, error = yield from coord_streamer(len(orders), start_position, self.new_coord_path, self.voc_encoder, self.coord_range, self.no_token_range, self.atom_order, center=None)

        if coords is not None:
            for i_coord, order in enumerate(orders):
                atom = self.protein.GetAtom(int(order)+1)
                try:
                    atom.SetVector(float(coords[i_coord, 0]), float(coords[i_coord, 1]), float(coords[i_coord, 2]))
                except Exception as e:
                    print(f"{order=}, {i_coord=}", flush=True)
                    raise e
            if not self.h_coord:
                self.protein.DeleteHydrogens()
            make_pardir(self.new_pdb_path)
            obc.WriteFile(self.protein, self.new_pdb_path)
        yield False, pos, [self.voc_encoder.voc2i['[END]']]

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
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--trial", type=int, default=5)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--no-token-range", action='store_true')
    ## tests
    parser.add_argument("--log-position", action='store_true')
    parser.add_argument("--log-token-range", action='store_true')
    args = parser.parse_args()

    train_dir = f"./training/results/{args.studyname}"
    targs = Namespace(**yaml.safe_load(open(f"{train_dir}/args.yaml")))
    assert targs.PDBUniMolRandom > 0
    out_dir = "./generate/protein_structure/training"+(f"/{args.genname}" if args.genname is not None else "")+f"/{args.studyname}/{args.opt}"
    assert not targs.no_pocket_heavy_atom and not targs.no_pocket_heavy_coord

    protein = PDBUniMolRandomDataset('valid')
    protein = CacheDataset(protein)
    protein_token = ProteinTokenizeDataset(protein, heavy_atom=not targs.no_pocket_heavy_atom, heavy_coord=not targs.no_pocket_heavy_coord, h_atom=targs.pocket_h_atom, h_coord=targs.pocket_h_coord, coord_follow_atom=targs.coord_follow_atom, atom_order=getattr(targs, 'pocket_atom_order', False), coord_range=targs.coord_range)
    protein_atom_token, _coord_token = TokenSplitDataset(protein_token, '[XYZ]').untuple()
    sentence = SentenceDataset('[POCKET]', protein_atom_token, '[XYZ]')
    index, sentence = index_dataset(sentence)
    prompt = StackDataset(index, protein, sentence)
    sample_idxs = np.random.default_rng(args.sample_seed).choice(len(prompt), args.n, replace=False)
    prompt = Subset(prompt, sample_idxs)

    def streamer_fn(item, i_trial, voc_encoder):
        idx, protein, (token, position) = item
        streamer = ProteinStructureStreamer(
            prompt_pdb_path=f"{out_dir}/prompt_pdb/{item[0]}/{i_trial}.pdb", 
            prompt_atom_path=f"{out_dir}/prompt_atoms/{item[0]}/{i_trial}.csv",
            new_pdb_path=f"{out_dir}/new_pdb/{item[0]}/{i_trial}.pdb", 
            new_coord_path=f"{out_dir}/new_coord_csv/{item[0]}/{i_trial}.csv",
            protein=protein,
            voc_encoder=voc_encoder, coord_range=targs.coord_range, no_token_range=args.no_token_range, atom_order=getattr(targs, 'pocket_atom_order', False), h_atom=targs.pocket_h_atom, h_coord=targs.pocket_h_coord
        )
        streamer = TokenWriteStreamer(streamer, voc_encoder,
            prompt_position=position,
            prompt_csv_path=f"{out_dir}/prompt_token/{item[0]}/{i_trial}.csv",
            new_csv_path=f"{out_dir}/new_token/{item[0]}/{i_trial}.csv", 
        )
        streamer = TimeLogStreamer(streamer, name=f"{item[0]}][{i_trial}")
        return streamer
    get_token_position_fn = lambda item: item[2]
    generate(out_dir, targs, f"{train_dir}/models/{args.opt}.pth", prompt, streamer_fn, get_token_position_fn, args.trial, 10000, None, args.batch_size, 0, args.log_position, args.log_token_range)




