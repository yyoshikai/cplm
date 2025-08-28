import sys, os, math
import numpy as np, pandas as pd
import yaml
from addict import Dict
from tqdm import tqdm
sys.path += ["/workspace", "/workspace/cplm" ]
from src.utils.logger import get_logger
logger = get_logger(stream=True)
from src.data.pretrain import UniMolLigandDataset, MoleculeDataset, CoordTransform
from src.data.tokenizer import StringTokenizer, FloatTokenizer

result_dir = "./tmp0"
os.makedirs(result_dir, exist_ok=True)

org_dir = "/workspace/cplm/training/results/250619_mamba"
with open(f"{org_dir}/config.yaml") as f:
    args = Dict(yaml.safe_load(f))
args.mol_data = '/workspace/cheminfodata/unimol/ligands/train.lmdb'

coord_transform = CoordTransform(args.seed, True, True, args.coord_noise_std)
smiles_tokenizer = StringTokenizer(open("/workspace/cplm/src/data/smiles_tokens.txt").read().splitlines())
coord_tokenizer = FloatTokenizer(-args.coord_range, args.coord_range, log_interval=args.tokenizer_log_interval)

mol_data = UniMolLigandDataset(args.mol_data, 10, seed=args.seed, 
    atom_h=not args.no_lig_atom_h, coord_h=not args.no_lig_coord_h, randomize=args.lig_randomize, 
    sample_save_dir=f"{result_dir}/ligand_sample")
mol_data = MoleculeDataset(mol_data, coord_transform, smiles_tokenizer, coord_tokenizer)
vocs0 = mol_data.vocs()

items0 = []
for i in range(3):
    items0.append(mol_data[i])

# mod
import sys, os, math
import numpy as np, pandas as pd
import yaml
from addict import Dict
from tqdm import tqdm
sys.path += ["/workspace", "/workspace/cplm" ]
from src.utils.logger import get_logger
logger = get_logger(stream=True)
from src.data.pretrain2 import UniMolLigandDataset, MolProcessDataset
from src.data.coord_transform2 import CoordTransformDataset
from src.data.tokenizer import StringTokenizer, FloatTokenizer, SentenceDataset, TokenizeDataset, ArrayTokenizeDataset
from src.data import untuple

result_dir = "./tmp"
os.makedirs(result_dir, exist_ok=True)

org_dir = "/workspace/cplm/training/results/250619_mamba"
with open(f"{org_dir}/config.yaml") as f:
    args = Dict(yaml.safe_load(f))
args.mol_data = '/workspace/cheminfodata/unimol/ligands/train.lmdb'

smiles_tokenizer = StringTokenizer(open("/workspace/cplm/src/data/smiles_tokens.txt").read().splitlines())
coord_tokenizer = FloatTokenizer(-args.coord_range, args.coord_range, log_interval=args.tokenizer_log_interval)

mol_data = UniMolLigandDataset(args.mol_data, 10)
mol_data = MolProcessDataset(mol_data, seed=args.seed, h_atom=not args.no_lig_atom_h, h_coord=not args.no_lig_coord_h, randomize=args.lig_randomize, sample_save_dir=f"{result_dir}/ligand_sample")
smi_data, coord_data = untuple(mol_data, 2)
coord_data = CoordTransformDataset(coord_data, rstate=args.seed, normalize_coord=True, random_rotate=True, coord_noise_std=args.coord_noise_std)
coord_data = untuple(coord_data, 3)[0]

smi_data = TokenizeDataset(smi_data, smiles_tokenizer)
coord_data = ArrayTokenizeDataset(coord_data, coord_tokenizer)

mol_data = SentenceDataset('[LIGAND]', smi_data, '[XYZ]', coord_data, '[END]')

vocs = mol_data.vocs()

items = []
for i in range(3):
    items.append(mol_data[i])

assert vocs == vocs0
for i in range(3):
    assert items[i] == items0[i]

print(f'{items[1]=}')
print(f'{items0[1]=}')