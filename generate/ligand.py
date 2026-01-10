import os
import itertools as itr
from argparse import ArgumentParser, Namespace
from collections.abc import Generator
import yaml
import numpy as np
from rdkit import Chem
from rdkit.Chem import Conformer
from rdkit.Chem.rdDetermineBonds import DetermineBondOrders
from rdkit.Geometry import Point3D
from src.data.tokenizer import VocEncoder, SmilesTokenizer, FloatTokenizer
from src.data.molecule import element_symbols
from src.generate import generate, GeneratorStreamer
from .protein_structure.train import coord_streamer, make_pardir

def array_to_conf(coord: np.ndarray) -> Conformer:
    conf = Conformer()
    for i in range(len(coord)):
        conf.SetAtomPosition(i, Point3D(*coord[i].tolist()))
    return conf

class LigandStreamer(GeneratorStreamer):
    def __init__(self, name: str, prompt_token_path: str, new_token_path: str, new_sdf_path: str, coord_range: float, voc_encoder: VocEncoder, no_token_range: bool, h_atom: bool, h_coord: bool):
        super().__init__(name, prompt_token_path, new_token_path, voc_encoder)

        smi_tokenizer = SmilesTokenizer()
        coord_tokenizer = FloatTokenizer('', -coord_range, coord_range)
        smi_vocs = list(smi_tokenizer.vocs())+['[XYZ]']
        self.smi_token_range = sorted(self.voc_encoder.encode(smi_vocs))
        self.int_token_range = sorted(self.voc_encoder.encode(coord_tokenizer.int_vocs()))
        self.frac_token_range = sorted(self.voc_encoder.encode(coord_tokenizer.frac_vocs()))
        if no_token_range:
            self.smi_token_range = self.int_token_range = self.frac_token_range = list(range(self.voc_encoder.voc_size))
        self.new_sdf_path = new_sdf_path

    def put_generator(self) -> Generator[tuple[bool, list[int], list[int]], list[int], None]:
        prompt_tokens = yield
        prompt_tokens = self.voc_encoder.decode(prompt_tokens)
        assert prompt_tokens[-1] == '[LIGAND]'
        pos_iter = itr.count(len(prompt_tokens))
        # smiles
        smi_tokens = []
        while True:
            tokens = yield True, [next(pos_iter)], self.smi_token_range
            assert len(tokens) == 1
            token = tokens[0]
            if token == self.voc_encoder.voc2i['[XYZ]']: break
            smi_tokens.append(token)
        smi = ''.join(self.voc_encoder.decode(smi_tokens))
        param = Chem.SmilesParserParams()
        param.removeHs = False
        mol = Chem.MolFromSmiles(smi, param)
        # conformer
        if mol is not None:
            n_atom = mol.GetNumAtoms()
            coord = yield from coord_streamer(n_atom, next(pos_iter), self.new_coord_path, self.voc_encoder, self.coord_range, self.no_token_range)
            mol.AddConformer(array_to_conf(coord))
            make_pardir(self.new_sdf_path)
            with Chem.SDWriter(self.new_sdf_path) as w:
                w.write(mol)
        yield False, [next(pos_iter)], [self.voc_encoder.voc2i['[END]']]

class AtomLigandStreamer(GeneratorStreamer):
    def __init__(self, name: str, prompt_token_path: str, new_token_path: str, new_sdf_path: str, coord_range: float, voc_encoder: VocEncoder, no_token_range: bool, h_atom: bool, h_coord: bool):
        super().__init__(name, prompt_token_path, new_token_path, voc_encoder)
        self.new_sdf_path = new_sdf_path
        self.coord_range = coord_range
        self.no_token_range = no_token_range
        self.atom_token_range = sorted(self.voc_encoder.encode(element_symbols()+{'[XYZ]'}))
        if self.no_token_range:
            self.all_token_range = list(range(voc_encoder.voc_size))
        self.n_generated_atom = None
    def put_generator(self) -> Generator[tuple[bool, list[int], list[int]], list[int], None]:
        prompt_tokens = yield
        prompt_tokens = self.voc_encoder.decode(prompt_tokens)
        pos_iter = itr.count(len(prompt_tokens))
        assert prompt_tokens[-1] == '[LIGAND]'

        atoms = []
        while True:
            tokens = yield True, [next(pos_iter)], (self.all_token_range if self.no_token_range else self.atom_token_range)
            token = tokens[0]
            if token == self.voc_encoder.voc2i['[XYZ]']: break
            if token not in self.atom_token_range:
                yield False, [next(pos_iter)], [self.voc_encoder.voc2i['[END]']]
                return
            atoms.append(token)
        self.n_generated_atom = len(atoms)

        coords = yield from coord_streamer(self.n_generated_atom, next(pos_iter), None, self.voc_encoder, self.coord_range, self.no_token_range)
        mol = Chem.RWMol()
        for symbol in atoms:
            mol.AddAtom(Chem.Atom(symbol))
        mol.AddConformer(array_to_conf(coords))
        DetermineBondOrders(mol)
        make_pardir(self.new_sdf_path)
        with Chem.SDWriter(self.new_sdf_path) as w:
            w.write(mol)
        yield False, [next(pos_iter)], [self.voc_encoder.voc2i['[END]']]

if __name__ == '__main__':
    # arguments
    parser = ArgumentParser()
    ## training
    parser.add_argument("--studyname", required=True)
    parser.add_argument("--opt", required=True, type=int)
    parser.add_argument('--finetune', action='store_true')
    ## environment
    parser.add_argument("--batch-size", type=int, default=4)
    ## generation
    parser.add_argument("--genname")
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--max-prompt-len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-token", type=int, default=1000)
    parser.add_argument("--no-token-range", action='store_true')
    args = parser.parse_args()
    sname = args.studyname

    # directories
    if args.finetune:
        train_dir = f"./finetune/results/{args.studyname}"
    else:
        train_dir = f"./training/results/{args.studyname}"
    # config
    targs = Namespace(**yaml.safe_load(open(f"{train_dir}/args.yaml")))
    if targs.lig_atoms:
        raise NotImplementedError
    
    out_dir = f"./generate/ligand/{'finetune' if args.finetune else 'training'}" \
            f"{f'/{args.genname}' if args.genname is not None else ''}/{args.studyname}/{args.opt}"

    os.makedirs(out_dir, exist_ok=True)

    if targs.lig_atoms:
        def streamer_fn(item, i_trial, voc_encoder):
            return AtomLigandStreamer(
                name=f"{item[0]}", 
                prompt_token_path=f"{out_dir}/prompt_token/{item}.txt",
                new_token_path=f"{out_dir}/new_token/{item}.txt",
                new_sdf_path=f"{out_dir}/new_sdf/{item}.txt",
                coord_range=targs.coord_range, voc_encoder=voc_encoder, no_token_range=args.no_token_range, h_atom=targs.pocket_h_atom, h_coord=targs.pocket_h_coord
            )
    else:
        def streamer_fn(item, i_trial, voc_encoder):
            return LigandStreamer(
                name=f"{item}",
                prompt_token_path=f"{out_dir}/prompt_token/{item}.txt",
                new_token_path=f"{out_dir}/new_token/{item}.txt",
                new_sdf_path=f"{out_dir}/new_sdf/{item}.txt",
                coord_range=targs.coord_range, voc_encoder=voc_encoder, no_token_range=args.no_token_range, h_atom=targs.pocket_h_atom, h_coord=targs.pocket_h_coord
            )
    get_token_position_fn = lambda item: (['[LIGAND]'], [0])

    prompt_data = list(range(args.n))
    generate(out_dir, targs, f"{train_dir}/models/{args.opt}.pth", prompt_data, streamer_fn, get_token_position_fn, 1, None, args.max_prompt_len, args.max_new_token, args.batch_size, args.seed, )

