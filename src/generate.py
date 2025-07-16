import sys, os, itertools, pickle, yaml
from argparse import ArgumentParser, Namespace
from addict import Dict
import numpy as np, pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, StackDataset
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem

sys.path += ["/workspace/cplm"]
from src.data.tokenizer import FloatTokenizer, \
    ProteinAtomTokenizer, VocEncoder, TokenEncodeDataset
from src.data import untuple_dataset, index_dataset
from src.data.finetune import  CDDataset, RandomScoreDataset
from src.data.tokenizer import TokenizeDataset, ArrayTokenizeDataset, SentenceDataset
from src.model import Model
from src.utils import logend, set_random_seed
from src.evaluate import parse_mol_tokens, parse_mol
from src.utils.logger import add_stream_handler, add_file_handler, get_logger
from src.utils.path import cleardir, subs_vars
PROJ_DIR = "/workspace/cplm"

def add_pocket_conditioned_generate_args(parser: ArgumentParser):
    parser.add_argument("--data-dir")
    parser.add_argument("--score-min", type=float)
    parser.add_argument("--score-max", type=float)
    parser.add_argument("--index", required=True)
    parser.add_argument("--gtype", type=int, default=2, choices=[1,2,3])
    parser.add_argument("--max-len", type=int, default=1000)
    parser.add_argument("--token-per-batch", type=int)
    parser.add_argument("--seed", type=int, default=0)


def pocket_conditioned_generate(args: Namespace, rdir: str, fdir: str, model_path: str):
    
    fargs = Dict(yaml.safe_load(open(f"{fdir}/config.yaml")))
    fargs = subs_vars(fargs, {'/work/02/ga97/a97003': '/workspace'})
    if args.token_per_batch is None:
        args.token_per_batch = fargs.token_per_batch

    if os.path.exists(f"{rdir}/info.csv"):
        print(f"{rdir} already finished.")
        return
    
    # Environment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_random_seed(args.seed)
    ## Result dir
    cleardir(rdir)

    ## Logger
    logger = get_logger()
    add_stream_handler(logger)
    add_file_handler(logger, f"{rdir}/debug.log")

    # 引数の保存
    with open(f"{rdir}/config.yaml", 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)


    # Data
    with logend(logger, 'Prepare data'):

        ## CD
        data = CDDataset(args.data_dir, fargs.seed, random_rotate=False, mol_atom_h=True, mol_coord_h=True, 
            pocket_coord_heavy=fargs.pocket_coord_heavy)
        pocket_atom, pocket_coord, _, _, score, center  = untuple_dataset(data, 6)

        ## Vocs from state
        with logend(logger, 'loading state'):
            state = torch.load(model_path, weights_only=True)
            state = {key[7:]: value for key, value in state.items()}
            state_vocs = state['vocs']

        ## Sentence
        float_tokenizer = FloatTokenizer(-fargs.coord_range, fargs.coord_range)
        pocket_atom = TokenizeDataset(pocket_atom, ProteinAtomTokenizer())
        pocket_coord = ArrayTokenizeDataset(pocket_coord, float_tokenizer)
        sentence = ['[POCKET]']

        sentence += [pocket_atom, '[XYZ]', pocket_coord]

        if not fargs.no_score:
            if args.score_min is not None and args.score_max is not None:
                score = RandomScoreDataset(args.score_min, args.score_max, 
                    len(pocket_atom), args.seed)
            else:
                assert args.score_min is None and args.score_max is None
            score = TokenizeDataset(score, float_tokenizer)
            sentence += ['[SCORE]', score]
        sentence += ['[LIGAND]']
        data = SentenceDataset(*sentence)
        
        data_vocs = data.vocs()
        assert data_vocs <= set(state_vocs)
        assert state_vocs[0] == '[PAD]'
        voc_encoder = VocEncoder(state_vocs[1:])
        data = TokenEncodeDataset(data, voc_encoder)

        ## Stack 
        idx_data, data = index_dataset(data)
        data = StackDataset(idx_data, data, center)

        ## Generation data index
        indices = np.load(f"../index/results/{args.index}.npy")
        data = Subset(data, indices)

    model = Model(8, 768, 12, 4, 0.1, 'gelu', True, voc_encoder.i2voc, voc_encoder.pad_token)
    logger.info(model.load_state_dict(state))
    model.to(device)

    # 生成
    batch_size = len(data) if args.gtype == 3 \
        else args.token_per_batch // args.max_len
    def collate_fn(batch):
        idxs, batch, centers = list(zip(*batch))
        batch = pad_sequence(batch, padding_value=voc_encoder.pad_token)
        return idxs, batch, centers
    train_loader = DataLoader(data, shuffle=False, num_workers=28, batch_size=batch_size, collate_fn=collate_fn, )
    idxs = []
    outputs = []
    centers = []

    with torch.inference_mode(), logend(logger, 'generate'):
        for idxs_batch, batch, centers_batch in train_loader:
            if args.gtype != 3:
                batch = batch.to(device)

            match args.gtype:
                case 1:
                    output = model.generate(batch, '[END]', args.max_len, voc_encoder.pad_token)
                case 2:
                    output = model.generate2(batch, '[END]', args.max_len, voc_encoder.pad_token, 10)
                case 3:
                    output = model.generate3(batch, '[END]', args.max_len, voc_encoder.pad_token, args.token_per_batch, np.arange(100, args.max_len+1, 100))

            outputs += output
            centers += centers_batch
            idxs += idxs_batch
        with open(f"{rdir}/tokens.pkl", 'wb') as f:
            pickle.dump(outputs, f)

    # detokenize
    with logend(logger, 'detokenize'):
        end_token = voc_encoder.voc2i['[END]']
        with open(f"{rdir}/tokens.txt", 'w') as f:
            for i in range(len(outputs)):
                tokens = outputs[i]
                tokens = itertools.takewhile(lambda x: x != end_token, tokens)
                words = voc_encoder.decode(tokens)
                f.write(','.join(words)+'\n')

    # parse SMILES and coordinates
    with logend(logger, 'parse tokens'):
        with open(f"{rdir}/tokens.txt") as f:
            wordss = [line.split(',') for line in f.read().splitlines()]

        smiless = []
        errors = []
        os.makedirs(f"{rdir}/sdf", exist_ok=True)
        for i in range(len(wordss)):

            words = wordss[i]
            center = centers[i]

            error, smiles, coords = parse_mol_tokens(words)
            smiless.append(smiles)
            if error != "":
                errors.append(error)
                continue

            coords += center
            error, mol = parse_mol(smiles, coords)
            errors.append(error)
            if error != "":
                continue
            with open(f"{rdir}/sdf/{i}.sdf", 'w') as f:
                f.write(Chem.MolToMolBlock(mol))

        df = pd.DataFrame({'idx': idxs, 'smiles': smiless, 'error': errors})
        df.to_csv(f"{rdir}/info.csv")
