"""
PoseBustersデータセットで評価


"""
import yaml, math
import numpy as np
from argparse import Namespace, ArgumentParser
from rdkit import Chem

from src.utils import setdefault
from src.utils.path import make_pardir
from src.data import StackDataset, index_dataset, untuple_dataset
from src.data.tokenizer import SentenceDataset, VocEncoder
from src.data.molecule import RandomScoreDataset
from src.data.datasets.posebusters import PosebustersV2ProteinDataset, PosebustersV2LigandDataset
from src.finetune import get_finetune_data
from src.data.tokenizer import TokenRSplitDataset
from src.generate.streamer import GeneratorStreamer, coord_streamer, array_to_conf, TokenWriteStreamer
from src.generate import generate

class LigandCoordStreamer(GeneratorStreamer):
    def __init__(self, mol: Chem.Mol, new_sdf_path: str|None, coord_range: float, voc_encoder: VocEncoder, no_token_range: bool, center: np.ndarray|None=None):
        self.voc_encoder = voc_encoder
        self.coord_range = coord_range
        self.no_token_range = no_token_range
        self.new_sdf_path = new_sdf_path
        self.center = center
        self.mol = Chem.Mol(mol, quickCopy=True) # remove conformers
        self.error = 'PARSE_NOT_ENDED'
        super().__init__()

    def put_generator(self):
        prompt_tokens = yield
        prompt_tokens = self.voc_encoder.decode(prompt_tokens)
        assert prompt_tokens[-1] == '[XYZ]'
        n_atom = self.mol.GetNumAtoms()
        coord, pos, self.error = yield from coord_streamer(n_atom, len(prompt_tokens), None, self.voc_encoder, self.coord_range, self.no_token_range, False, self.center)

        if coord is not None:
            self.mol.AddConformer(array_to_conf(coord))
            if self.new_sdf_path is not None:
                make_pardir(self.new_sdf_path)
                with Chem.SDWriter(self.new_sdf_path) as w:
                    w.write(self.mol)
        yield False, pos, [self.voc_encoder.voc2i['[END]']]


if __name__ == '__main__':

    parser = ArgumentParser()
    ## training
    parser.add_argument("--sname", required=True)
    parser.add_argument("--opt", required=True, type=int)
    parser.add_argument("--reinforce", action='store_true')
    ## generation
    parser.add_argument("--genname")
    parser.add_argument("--n", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--trial", type=int, default=5)
    parser.add_argument("--no-token-range", action='store_true')
    ## environment
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--log-position", action='store_true')
    parser.add_argument("--log-token-range", action='store_true')
    parser.add_argument("--max-workers", type=int)
    args = parser.parse_args()

    # finetuning / training args
    ## finetuning
    if args.reinforce:
        reinforce_dir = f"reinforce/results/{args.sname}"
        rargs = Namespace(**yaml.safe_load(open(f"{reinforce_dir}/args.yaml")))
        finetune_dir = f"finetune/results/{rargs.finetune_name}"
        model_path = f"{reinforce_dir}/models/{args.opt}.pth"
    else:
        finetune_dir = f"finetune/results/{args.sname}"
        model_path = f"{finetune_dir}/models/{args.opt}.pth"
    fargs = Namespace(**yaml.safe_load(open(f"{finetune_dir}/args.yaml")))
    setdefault(fargs, 'lig_atoms', False)

    out_dir = ("./generate/protein_ligand/"
            +("reinforce" if args.reinforce else "finetune")
            +(f"/{args.genname}" if args.genname is not None else '')
            +f"/{args.sname}/{args.opt}")

    if not fargs.protein:
        raise NotImplementedError
    protein = PosebustersV2ProteinDataset()
    lig =PosebustersV2LigandDataset()
    score = RandomScoreDataset(-12, -10, len(protein), args.seed)
    raw_data = StackDataset(protein, lig, score)
    _voc_encoder, _raw, protein_data, lig_data, token_data, position_data, _weight, center_data, data_logs = get_finetune_data(fargs, 'test', add_ligand=True, random_rotate=False, added_vocs=set(), prompt_score='none' if fargs.no_score else 'low', raw_data=raw_data, encode=False)
    token_position_data = StackDataset(token_data, position_data)
    token_position_data, _ = TokenRSplitDataset(token_position_data, '[XYZ]').untuple()
    token_position_data = SentenceDataset(token_position_data, '[XYZ]')
    token_data, position_data = token_position_data.untuple()
    idx_data, token_data = index_dataset(token_data)
    prompt_data = StackDataset(idx_data, lig_data, token_data, position_data)

    def streamer_fn(item, i_trial: int, voc_encoder):
        idx, lig, token, position = prompt_data
        streamer = LigandCoordStreamer(
            lig, 
            new_sdf_path=f"{out_dir}/new_sdf/{idx}/{i_trial}.sdf", 
            coord_range=fargs.coord_range, 
            voc_encoder=voc_encoder, 
            no_token_range=args.no_token_range
        )
        streamer = TokenWriteStreamer(
            prompt_token_path=f"{out_dir}/prompt_token/{idx}/{i_trial}.txt",
            new_token_path=f"{out_dir}/new_token/{idx}/{i_trial}/.txt",
            voc_encoder=voc_encoder
        )
        return streamer
    get_token_position_fn = lambda item: (item[2], item[3])
    generate(out_dir, fargs, model_path, prompt_data, streamer_fn, get_token_position_fn, max_n_sample=args.trial, max_prompt_len=math.inf, max_new_token=None, batch_size=args.batch_size, seed=args.seed, log_position=args.log_position, log_token_range=args.log_token_range)

