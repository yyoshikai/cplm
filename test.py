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
from torch.utils.data import StackDataset
sys.path.append('/workspace/cplm')
from src.data.finetune2 import CDDataset2, CDDataset, MolProcessDataset, ProteinProcessDataset, CentralizeCoordsDataset, RandomRotateDataset
from src.data import untuple

sdir = "/workspace/cplm/finetune/results/250628_mamba"
with open(f"{sdir}/config.yaml") as f:
    args = Dict(yaml.safe_load(f))
args.finetune_save_dir = "/workspace/cplm/ssd/preprocess/results/finetune/r4_all"
rstate = np.random.RandomState(args.seed)
protein, lig, score = untuple(CDDataset2(args.finetune_save_dir), 3)
lig_smi, lig_coord = untuple(MolProcessDataset(lig, rstate, h_atom=True, h_coord=True), 2)
protein_atoms, protein_coord = untuple(ProteinProcessDataset(protein, heavy_coord=args.pocket_coord_heavy), 2)
center, lig_coord, protein_coord = untuple(CentralizeCoordsDataset(lig_coord, protein_coord), 3)
rotation_matrix, lig_coord, protein_coord = untuple(RandomRotateDataset(rstate, lig_coord, protein_coord), 3)
cddata = StackDataset(lig_smi, lig_coord, protein_atoms, protein_coord, score, center, rotation_matrix)


os.makedirs("items/mod", exist_ok=True)
for i in range(3):
    with open(f"items/mod/{i}.pkl", 'wb') as f:
        lig_smi, lig_coord, protein_atoms, protein_coord, score, center, rotation_matrix = cddata[i]
        item = (protein_atoms, protein_coord, lig_smi, lig_coord, score, center, rotation_matrix)
        pickle.dump(item, f)
    assert (filecmp.cmp(f"items/ref/{i}.pkl", f"items/mod/{i}.pkl"))
    print('OK')