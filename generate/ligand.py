import os
import itertools as itr
from argparse import ArgumentParser, Namespace
from collections.abc import Generator
import yaml
from rdkit import Chem
from rdkit.Chem import Conformer
from rdkit.Geometry import Point3D
from src.model.transformer import Streamer
from src.data.tokenizer import VocEncoder, SmilesTokenizer, FloatTokenizer
from src.generate import generate, GeneratorStreamer

class LigandStreamer(GeneratorStreamer):
    def __init__(self, voc_encoder: VocEncoder, out_token_dir: str, out_sdf_path: str, coord_range: float, no_token_range: bool):
        super().__init__(voc_encoder, out_token_dir)

        smi_tokenizer = SmilesTokenizer()
        coord_tokenizer = FloatTokenizer('', -coord_range, coord_range)
        smi_vocs = list(smi_tokenizer.vocs())+['[XYZ]']
        self.smi_token_range = sorted(self.voc_encoder.encode(smi_vocs))
        self.int_token_range = sorted(self.voc_encoder.encode(coord_tokenizer.int_vocs()))
        self.frac_token_range = sorted(self.voc_encoder.encode(coord_tokenizer.frac_vocs()))
        if no_token_range:
            self.smi_token_range = self.int_token_range = self.frac_token_range = list(range(self.voc_encoder.voc_size))
        self.out_sdf_path = out_sdf_path

    def put_generator(self) -> Generator[tuple[bool, list[int], list[int]], list[int], None]:
        prompt_tokens = yield
        prompt_tokens = self.voc_encoder.decode(prompt_tokens)
        assert prompt_tokens[-1] == '[LIGAND]'
        pos_iter = itr.count()
        next_positions = [next(pos_iter) for _ in prompt_tokens]
        self.is_prompt = False

        # smiles
        smi_tokens = []
        while True:
            tokens = yield True, next_positions, self.smi_token_range
            assert len(tokens) == 1
            token = tokens[0]
            if token == self.voc_encoder.voc2i['[XYZ]']: break
            smi_tokens.append(token)
            next_positions = [next(pos_iter)]
        ## parse
        smi = ''.join(self.voc_encoder.decode(smi_tokens))
        param = Chem.SmilesParserParams()
        param.removeHs = False
        mol = Chem.MolFromSmiles(smi, param)
        if mol is not None:
            n_atom = mol.GetNumAtoms()
            coords = []
            for i in range(n_atom*3):
                int_token = yield True, [next(pos_iter)], self.int_token_range
                frac_token = yield True, [next(pos_iter)], self.frac_token_range
                try:
                    coord = float(''.join(self.voc_encoder.decode(int_token+frac_token)))
                except Exception:
                    yield False, [next(pos_iter)], [self.voc_encoder.voc2i['[END]']]
                    return
                coords.append(coord)
            conf = Conformer()
            for i_atom in range(n_atom):
                conf.SetAtomPosition(i_atom, Point3D(*coords[i_atom*3:i_atom*3+3]))
            mol.AddConformer(conf)
            os.makedirs(os.path.dirname(self.out_sdf_path), exist_ok=True)
            with Chem.SDWriter(self.out_sdf_path) as w:
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

    streamer_fn = lambda item, i_trial, voc_encoder: LigandStreamer(voc_encoder, f"{out_dir}/generate/{item}/{i_trial}", f"{out_dir}/sdfs/{item}.sdf", targs.coord_range, args.no_token_range)
    get_token_fn = lambda item: ['[LIGAND]']

    prompt_data = list(range(args.n))
    generate(out_dir, targs, f"{train_dir}/models/{args.opt}.pth", prompt_data, streamer_fn, get_token_fn, 1, None, args.max_prompt_len, args.max_new_token, args.batch_size, args.seed, )

