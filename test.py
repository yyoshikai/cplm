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
        pickle.dump(cddata[i], f)

# ref
import sys, os, pickle
from addict import Dict
import yaml
import filecmp
import numpy as np
sys.path.append('/workspace/cplm')
from src.data.finetune2 import CDDataset2, CDDataset, MolProcessDataset, ProteinProcessDataset
from src.data import untuple

sdir = "/workspace/cplm/finetune/results/250628_mamba"
with open(f"{sdir}/config.yaml") as f:
    args = Dict(yaml.safe_load(f))
args.finetune_save_dir = "/workspace/cplm/ssd/preprocess/results/finetune/r4_all"
rstate = np.random.RandomState(args.seed)
cddata = CDDataset2(args.finetune_save_dir)
protein, lig, score = untuple(cddata, 3)
lig = MolProcessDataset(lig, rstate, h_atom=True, h_coord=True)
lig_smi, lig_coord = untuple(lig, 2)
protein = ProteinProcessDataset(protein, heavy_coord=args.pocket_coord_heavy)
protein_atoms, protein_coord = untuple(protein, 2)
cddata = CDDataset(protein_atoms, protein_coord, lig_smi, lig_coord, score, rstate)


os.makedirs("items/mod", exist_ok=True)
for i in range(3):
    with open(f"items/mod/{i}.pkl", 'wb') as f:
        pickle.dump(cddata[i], f)
    assert (filecmp.cmp(f"items/ref/{i}.pkl", f"items/mod/{i}.pkl"))
    print('OK')