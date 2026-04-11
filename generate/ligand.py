from argparse import ArgumentParser, Namespace
import yaml
from src.generate import generate
from src.generate.streamer import SaveLigandStreamer, get_ligand_streamer, TokenWriteStreamer, RangeWriteStreamer

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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-token", type=int, default=1000)
    parser.add_argument("--no-token-range", action='store_true')
    ## check
    parser.add_argument("--check-token-range", action='store_true')
    args = parser.parse_args()
    sname = args.studyname

    # directories
    if args.finetune:
        train_dir = f"./finetune/results/{args.studyname}"
    else:
        train_dir = f"./training/results/{args.studyname}"
    # config
    targs = Namespace(**yaml.safe_load(open(f"{train_dir}/args.yaml")))
    
    out_dir = f"./generate/ligand/{'finetune' if args.finetune else 'training'}" \
            f"{f'/{args.genname}' if args.genname is not None else ''}/{args.studyname}/{args.opt}"

    def streamer_fn(item, i_trial, voc_encoder):
        streamer = get_ligand_streamer(targs.lig_format, targs.coord_range, voc_encoder, args.no_token_range, targs.lig_h, targs.smiles_voc_file)
        streamer = SaveLigandStreamer(streamer, f"{out_dir}/new_sdf/{item}.sdf")
        streamer = TokenWriteStreamer(streamer,
            voc_encoder=voc_encoder,
            prompt_position=[0],
            prompt_csv_path=f"{out_dir}/prompt_token/{item}.csv",
            new_csv_path=f"{out_dir}/new_token/{item}.csv",
        )
        if args.check_token_range and (item*5//args.n) > ((item-1)*5//args.n):
            streamer = RangeWriteStreamer(streamer, voc_encoder,
            range_path=f"{out_dir}/token_range/{item}.txt"
        )
        return streamer
    get_token_position_fn = lambda item: (['[LIGAND]'], [0])

    prompt_data = list(range(args.n))
    streamerss = generate(out_dir, targs, f"{train_dir}/models/{args.opt}.pth", prompt_data, streamer_fn, get_token_position_fn, 1, 1, args.max_new_token, args.batch_size, args.seed)

    for i, streamer in enumerate(streamerss):
        streamer = streamer[0].streamer.streamer
        print(streamer.error)