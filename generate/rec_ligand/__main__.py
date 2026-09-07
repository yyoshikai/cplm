"""
For reinforce use only.
Accelerate generation by multiple generation from the same prompt.
"""
import sys, os, math, yaml
from argparse import Namespace, ArgumentParser
from glob import glob
import torch
from reinforce_atom import CDNamedDataset
from src.utils.logger import get_logger, add_file_handler
from src.data.datasets.crossdocked import CDProteinTestDataset
from src.data.mol_tokenizer import get_mol_tokenizer
from src.train.data import get_finetune_data
from src.train import get_model
from src.streamer import LigandStreamer, SaveLigandStreamer

# Arguments
parser = ArgumentParser()
## training
parser.add_argument("--sname", required=True)
parser.add_argument("--opt", type=int, required=True)
## generation
parser.add_argument("--gname")
parser.add_argument("--n-gen", type=int, default=10)
parser.add_argument("--max-trial", type=int, default=100)
parser.add_argument("--protein", nargs='*', default=None)
## environment
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--max-prompt-len", type=int, required=True)
args = parser.parse_args()

# Environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
odir = f"generate/rec_ligand/{args.gname or ''}/{args.sname}/{args.opt}"
os.makedirs(odir, exist_ok=True)
logger = get_logger(stream=True)
add_file_handler(logger, f"{odir}/debug.log")
## save args
with open(f"{odir}/args.yaml", 'w') as f:
    yaml.dump(vars(args), f, sort_keys=False)

## load training args
with open(f"reinforce/results/{args.sname}/args.yaml") as f:
    rargs = Namespace(**yaml.safe_load(f))
with open(f"finetune/results/{rargs.finetune_name}/args.yaml") as f:
    fargs = Namespace(**yaml.safe_load(f))

## Load model
model, voc_encoder = get_model(fargs, voc_encoder=None, 
    init_state_path=f"reinforce/results/{args.sname}/models/{args.opt}.pth",
    device=device)
model.eval()

# Make data
## raw data
if getattr(rargs, "protein", None) is not None:
    raw_data = CDNamedDataset(rargs.protein, fargs.pocket_cls)
    idxs = list(range(len(raw_data)))
else:
    raw_data = CDProteinTestDataset(fargs.pocket_cls)
    if args.protein is not None:
        idxs = [i for i, (pname, lname) in enumerate(raw_data.split_by_name)
                if pname.split('_rec_')[0] in args.protein]
        logger.debug(f"{idxs=}")
    else:
        idxs = list(range(len(raw_data)))

## prompt protein dataset
added_vocs = set(voc_encoder.i2voc) - {'[PAD]'}
data_voc_encoder, raw_data, _rec_data, _lig, prompt_token_data, position_data, _weight, \
    _center_data, _data_logs = get_finetune_data(args=fargs, split='test', 
    sample=1.0, add_ligand=False, random_ligand=False, random_rotate=False, 
    added_vocs=added_vocs,
    prompt_score='none', raw_data=raw_data, tensor_position=False)

if data_voc_encoder.i2voc != voc_encoder.i2voc:
    logger.error(f"{voc_encoder.i2voc=}")
    logger.error(f"{data_voc_encoder.i2voc=}")
    raise ValueError

## mol_tokenizer
mol_tokenizer = get_mol_tokenizer(fargs.lig_format, fargs.lig_order, fargs.smiles_voc_dir, fargs.lig_h)

# Generation
for idx in idxs:
    ## get data
    prompt = prompt_token_data[idx].to(device)
    position = position_data[idx]

    ## skip large prompts
    if len(prompt) >= args.max_prompt_len:
        logger.warning(f"Skipped {idx=}: {len(prompt)=}")
        continue
    prompt_cache = None
    for nb in range(math.floor(args.max_trial/args.batch_size)):
        ## streamer
        streamers = []
        for b in range(args.batch_size):
            streamer = LigandStreamer(mol_tokenizer, voc_encoder, end_token='[END]', cls='rdkit')
            streamer = SaveLigandStreamer(streamer, f"{odir}/new_sdf/{idx}/{nb*args.batch_size+b}.sdf")
            streamers.append(streamer)
        generated_prompt_caches = model.generate3(
            contexts=[prompt]*args.batch_size, 
            positions=[position]*args.batch_size, 
            streamers=streamers, 
            max_new_token=rargs.max_new_token, 
            prompt_caches = [prompt_cache] * args.batch_size, 
            return_prompt_caches=True            
        )
        prompt_cache = generated_prompt_caches[0]

        ## count generated sdfs
        if len(glob(f"{odir}/new_sdf/{idx}/*.sdf")) >= args.n_gen:
            break
    logger.info(f"{idx}/{len(prompt_token_data)} finished.")

    