# ref
import sys, os, pickle
from addict import Dict
import yaml
sys.path.append('/workspace/cplm')
from src.data.finetune import CDDataset

sdir = "/workspace/cplm/finetune/results/250628_mamba"
with open(f"{sdir}/config.yaml") as f:
    args = Dict(yaml.safe_load(f))
args.finetune_save_dir = "/workspace/cplm/ssd/preprocess/results/finetune/r4_all"
cddata = CDDataset(args.finetune_save_dir, args.seed, mol_atom_h=True,
        mol_coord_h=True, pocket_coord_heavy=args.pocket_coord_heavy)

os.makedirs("items/ref", exist_ok=True)
for i in range(3):
    with open(f"items/ref/{i}.pkl", 'wb') as f:
        pickle.dump(cddata[0], f)

# ref
import sys, os, pickle
from addict import Dict
import yaml
import filecmp
sys.path.append('/workspace/cplm')
from src.data.finetune2 import CDDataset2, CDDataset

sdir = "/workspace/cplm/finetune/results/250628_mamba"
with open(f"{sdir}/config.yaml") as f:
    args = Dict(yaml.safe_load(f))
args.finetune_save_dir = "/workspace/cplm/ssd/preprocess/results/finetune/r4_all"
cddata = CDDataset2(args.finetune_save_dir)
cddata = CDDataset(cddata, args.seed, mol_atom_h=True,
        mol_coord_h=True, pocket_coord_heavy=args.pocket_coord_heavy)

os.makedirs("items/mod", exist_ok=True)
for i in range(3):
    with open(f"items/mod/{i}.pkl", 'wb') as f:
        pickle.dump(cddata[0], f)
    assert (filecmp.cmp(f"items/ref/{i}.pkl", f"items/mod/{i}.pkl"))