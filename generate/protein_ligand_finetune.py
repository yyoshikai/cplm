import sys, os, yaml, math
from argparse import ArgumentParser, Namespace
import numpy as np
from src.utils import setdefault
from src.data import StackDataset, Subset, index_dataset
from src.data.tokenizer import SmilesTokenizer
from src.finetune import get_finetune_data
from src.generate import generate
from src.generate.streamer import LigandStreamer, AtomLigandStreamer

if __name__ == '__main__':
    # Argument
    parser = ArgumentParser()
    ## training
    parser.add_argument("--sname", required=True)
    parser.add_argument("--opt", required=True, type=int)
    ## generation
    parser.add_argument("--genname")
    parser.add_argument("--n", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--trial", type=int, default=10)
    parser.add_argument("--no-token-range", action='store_true')
    ## environment
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--log-position", action='store_true')
    parser.add_argument("--log-token-range", action='store_true')
    args = parser.parse_args()

    logs = []
    
    # finetuning / training args
    ## finetuning
    finetune_dir = f"finetune/results/{args.sname}"
    fargs = yaml.safe_load(open(f"{finetune_dir}/args.yaml"))
    fargs = Namespace(**fargs)
    setdefault(fargs, 'lig_atoms', False)

    out_dir = ".generate/protein_ligand/finetune"+(f"/{args.genname}" if args.genname is not None else '')+f"/{args.sname}/{args.opt}"

    added_vocs = SmilesTokenizer().vocs()
    _voc_encoder, _raw, prompt_token_data, position_data, _weight, center_data, _rotation, _protein_finetune_data, _ligand_filename_data, data_logs = get_finetune_data(fargs, 'test', add_ligand=False, random_rotate=False, added_vocs=added_vocs, prompt_score='none' if fargs.no_score else 'low', encode=False)

    idx_data, prompt_token_data = index_dataset(prompt_token_data)
    prompt_data = StackDataset(idx_data, prompt_token_data, position_data, center_data)
    if args.n is not None and args.n < len(prompt_data):
        sample_idxs = np.random.default_rng(args.sample_seed).choice(len(prompt_data), args.n, replace=False)
        prompt_data = Subset(prompt_data, sample_idxs)

    get_token_position_fn = lambda item: (item[1], item[2])

    if fargs.lig_atoms:
        raise NotImplementedError
    else:
        def streamer_fn(item, i_trial, voc_encoder):
            idx, prompt_token, position, center = item
            return LigandStreamer(
                name=f"{idx}][{i_trial}", 
                prompt_token_path=f"{out_dir}/prompt_token/{idx}/{i_trial}.txt", 
                new_token_path=f"{out_dir}/new_token/{idx}/{i_trial}/.txt", 
                new_sdf_path=f"{out_dir}/new_sdf/{idx}/{i_trial}.sdf", 
                coord_range=fargs.coord_range, voc_encoder=voc_encoder, no_token_range=args.no_token_range, h_atom=fargs.pocket_h_atom, h_coord=fargs.pocket_h_coord, center=center
            )
    
    generate(out_dir, fargs, f"{finetune_dir}/models/{args.opt}.pth", prompt_data, streamer_fn, get_token_position_fn, max_n_sample=args.trial, max_prompt_len=math.inf, max_new_token=None, batch_size=args.batch_size, seed=args.seed, log_position=args.log_position, log_token_range=args.log_token_range)