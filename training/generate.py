import sys, os, argparse
import math, itertools, pickle
import yaml
import torch


WORKDIR = os.environ.get('WORKDIR', "/workspace")
sys.path += [f"{WORKDIR}/cplm"]
from src.utils.logger import get_logger, add_stream_handler, add_file_handler
from src.model import Model
from src.data.tokenizer import VocEncoder

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--studyname", required=True)
parser.add_argument("--genname", default='mol')
parser.add_argument("--step", required=True, type=int)
parser.add_argument("--start-voc", default='[LIGAND]')
parser.add_argument("--end-voc", default='[END]')
parser.add_argument("--token-per-batch", type=int, help='defaults to same as training')
parser.add_argument("--n", type=int, default=25)
parser.add_argument("--max-len", type=int, default=1000)
args = parser.parse_args()
sname = args.studyname
step = args.step

# directories
train_dir = f"./training/results/{args.studyname}"
rdir = f"./training/generate/results/{args.studyname}/{args.step}/{args.genname}"
os.makedirs(rdir, exist_ok=True)

# save args
with open(f"{rdir}/args.yaml", 'w') as f:
    yaml.dump(vars(args), f, sort_keys=False)

# config
config = yaml.safe_load(open(f"{train_dir}/config.yaml"))
coord_range = config.get('coord_range', 20)
token_per_batch = args.token_per_batch if args.token_per_batch is not None \
    else config['token_per_batch']

# environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = get_logger()
add_stream_handler(logger)
add_file_handler(logger, f"{rdir}/log.log")


# state
state = torch.load(f"{train_dir}/models/{step}.pth", weights_only=True)
state2 = {key[7:]: value for key, value in state.items()}

# data
vocs = state2['vocs']
assert args.start_voc in vocs, f"{args.start_voc=} not in vocs"
assert args.end_voc in vocs, f"{args.end_voc=} not in vocs"
voc_encoder = VocEncoder(vocs[1:])
assert vocs == voc_encoder.i2voc
end_token = voc_encoder.voc2i[args.end_voc]

# model
model = Model(config.get('n_layer', 8), 768, 12, 4, 0.1, 'gelu', True, vocs, voc_encoder.pad_token)
logger.info(model.load_state_dict(state2))
model.to(device)
model.eval()

# generate tokens
max_len = args.max_len
batch_size = token_per_batch // max_len
nbatch = math.ceil(float(args.n)/float(batch_size))
outputs = []
for ibatch in range(nbatch):
    bsz0 = min(batch_size, args.n-batch_size*ibatch)
    context = torch.full((1, bsz0), dtype=torch.long, device=device, 
        fill_value=voc_encoder.voc2i[args.start_voc])
    with torch.inference_mode():
        output = model.generate(context, args.end_voc, args.max_len, voc_encoder.pad_token)
        outputs += output.cpu().numpy().tolist()

with open(f"{rdir}/tokens.pkl", 'wb') as f:
    pickle.dump(outputs, f)

# detokenize
with open(f"{rdir}/words.txt", 'w') as f:
    for i in range(len(outputs)):
        tokens = outputs[i]
        tokens = itertools.takewhile(lambda x: x != end_token, tokens)
        words = voc_encoder.decode(tokens)
        f.write(','.join(words)+'\n')
