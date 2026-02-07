import sys, os, yaml, math
import concurrent.futures as cf
from argparse import ArgumentParser, Namespace
import numpy as np
from src.utils import setdefault
from src.utils.path import mwrite
from src.data import StackDataset, Subset, index_dataset
from src.data.tokenizer import SmilesTokenizer
from src.train.data import get_finetune_data
from src.generate import generate
from src.generate.streamer import LigandStreamer, TokenWriteStreamer, RangeWriteStreamer
from src.chem import obmol2pdb


if __name__ == '__main__':
    # Argument
    parser = ArgumentParser()
    ## training
    parser.add_argument("--sname", required=True)
    parser.add_argument("--opt", required=True, type=int)
    parser.add_argument("--reinforce", action='store_true')
    ## generation
    parser.add_argument("--gname")
    parser.add_argument("--n", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--trial", type=int, default=5)
    parser.add_argument("--no-token-range", action='store_true')
    ## environment
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-workers", type=int)
    ## check
    parser.add_argument("--check-token-range", action='store_true')
    args = parser.parse_args()

    logs = []
    
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

    out_dir = ("./generate/rec_ligand/"
            +("reinforce" if args.reinforce else "finetune")
            +(f"/{args.gname}" if args.gname is not None else '')
            +f"/{args.sname}/{args.opt}")
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/args.yaml", 'w') as f:
        yaml.dump(vars(args), f)

    added_vocs = SmilesTokenizer().vocs()
    _voc_encoder, _raw, rec_data, _lig, prompt_token_data, position_data, _weight, center_data, data_logs = get_finetune_data(fargs, 'test', 1.0, add_ligand=False, random_rotate=False, added_vocs=added_vocs, prompt_score='none' if fargs.no_score else 'low', encode=False)

    idx_data, prompt_token_data = index_dataset(prompt_token_data)
    prompt_data = StackDataset(idx_data, rec_data, prompt_token_data, position_data)
    if args.n is not None and args.n < len(prompt_data):
        sample_idxs = np.random.default_rng(args.sample_seed).choice(len(prompt_data), args.n, replace=False)
        prompt_data = Subset(prompt_data, sample_idxs)
    N = len(prompt_data)

    get_token_position_fn = lambda item: (item[2], item[3])

    with cf.ProcessPoolExecutor() as e:
        if fargs.lig_atoms:
            raise NotImplementedError
        else:
            def streamer_fn(item, i_trial, voc_encoder):
                idx, rec, prompt_token, position = item
                mwrite(f"{out_dir}/prompt_rec_pdb/{idx}/{i_trial}.pdb", obmol2pdb(rec))
                streamer = LigandStreamer(
                    new_sdf_path=f"{out_dir}/new_sdf/{idx}/{i_trial}.sdf", 
                    coord_range=fargs.coord_range, voc_encoder=voc_encoder, no_token_range=args.no_token_range, h_atom=fargs.pocket_h_atom, h_coord=fargs.pocket_h_coord
                )
                streamer = TokenWriteStreamer(streamer, voc_encoder,
                    prompt_position=position,
                    prompt_csv_path=f"{out_dir}/prompt_token/{idx}/{i_trial}.csv",
                    new_csv_path=f"{out_dir}/new_token/{idx}/{i_trial}.csv",
                )
                if args.check_token_range and (idx*5//N) == i_trial > ((idx-1)*5//N):
                    streamer = RangeWriteStreamer(streamer, voc_encoder,
                    range_path=f"{out_dir}/token_range/{idx}_{i_trial}.txt"
                )
                return streamer
        generate(out_dir, fargs, model_path, prompt_data, streamer_fn, get_token_position_fn, max_n_sample=args.trial, max_prompt_len=math.inf, max_new_token=None, batch_size=args.batch_size, seed=args.seed)
