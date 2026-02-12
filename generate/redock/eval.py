import sys, os, shutil
from argparse import ArgumentParser
from collections import defaultdict
import yaml
import pandas as pd
import openbabel.openbabel as ob
from tqdm import tqdm
from rdkit import Chem
from src.data.datasets.posebusters import PosebustersV2ProteinDataset
from src.chem import get_coord_from_mol, set_conf, pdb_path2obmol
from src.utils.logger import set_third_party_logger
from src.utils.path import WORKDIR
sys.path.append(f"{WORKDIR}/github/posebusters")
from posebusters.posebusters import PoseBusters

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gname', required=True)
    args = parser.parse_args()
    gdir = f"generate/redock/{args.gname}"
    set_third_party_logger()

    data = PosebustersV2ProteinDataset()
    
    for data_idx in tqdm(range(len(data)), desc='centering ligand', dynamic_ncols=True):
        if all((
                not os.path.exists(f"{gdir}/new_sdf/{data_idx}/{t}.sdf") 
                or os.path.exists(f"{gdir}/centered_new_sdf/{data_idx}/{t}.sdf")
        ) for t in range(5)):
            continue
        obc = ob.OBConversion()
        obc.SetInFormat('pdb')
        pdb_id = data.ids[data_idx]
        raw_rec = pdb_path2obmol(f"{data.pb_dir}/posebusters_benchmark_set/{pdb_id}/{pdb_id}_protein.pdb")
        raw_center = get_coord_from_mol(raw_rec).mean(axis=0)
        for trial_idx in range(5):
            in_lig_path = f"{gdir}/new_sdf/{data_idx}/{trial_idx}.sdf"
            out_lig_path = f"{gdir}/centered_new_sdf/{data_idx}/{trial_idx}.sdf"
            if not os.path.exists(in_lig_path) or os.path.exists(out_lig_path):
                continue
            with open(in_lig_path) as f:
                lig = Chem.MolFromMolBlock(f.read(), removeHs=False)
            conf = lig.GetConformer()
            coord = conf.GetPositions()
            coord += raw_center
            set_conf(conf, coord)
            os.makedirs(f"{gdir}/centered_new_sdf/{data_idx}", exist_ok=True)
            with open(out_lig_path, 'w') as f:
                f.write(Chem.MolToMolBlock(lig))
    
    file_paths = []
    for data_idx in range(len(data)):
        pdb_id = data.ids[data_idx]
        os.makedirs(f"{gdir}/dirs/{data_idx}", exist_ok=True)
        raw_rec_path = f"{data.pb_dir}/posebusters_benchmark_set/{pdb_id}/{pdb_id}_protein.pdb"
        link_rec_path = f"{gdir}/dirs/{data_idx}/raw_rec.pdb"
        if not os.path.exists(link_rec_path):
            os.symlink(raw_rec_path, link_rec_path)
        raw_lig_path = f"{data.pb_dir}/posebusters_benchmark_set/{pdb_id}/{pdb_id}_ligand.sdf"
        link_lig_path = f"{gdir}/dirs/{data_idx}/raw_ligand.sdf"
        if not os.path.exists(link_lig_path):
            os.symlink(raw_lig_path, link_lig_path)
        for trial_idx in range(5):
            raw_gen_path = f"{gdir}/centered_new_sdf/{data_idx}/{trial_idx}.sdf"
            link_gen_path = f"{gdir}/dirs/{data_idx}/centered_new_{trial_idx}.sdf"
            if os.path.exists(raw_gen_path):
                if not os.path.exists(link_lig_path):
                    os.symlink(f"../../centered_new_sdf/{data_idx}/{trial_idx}.sdf", link_gen_path)
                if not os.path.exists(f"{gdir}/eval/{data_idx}_{trial_idx}.yaml"):
                    file_paths.append((raw_gen_path, raw_lig_path, raw_rec_path))

    pb = PoseBusters('redock', None, max_workers=None, chunk_size=None)
    pb.file_paths = pd.DataFrame(file_paths, columns=['mol_pred', 'mol_true', 'mol_cond'])
    results = pb._run()
    os.makedirs(f"{gdir}/eval", exist_ok=True)
    for (path, _molecule_name, _position), result in tqdm(results, total=len(pb.file_paths), dynamic_ncols=True, desc="posebusters"):
        *_, data_idx, trial_idx = path.split('/')
        data_idx, trial_idx = int(data_idx), int(trial_idx.split('.')[0])
        data = defaultdict(dict)
        for k0, k1, v in result:
            data[k0][k1] = str(v)
        with open(f"{gdir}/eval/{data_idx}_{trial_idx}.yaml", 'w') as f:
            yaml.dump(dict(data), f)
    print(f"generate.redock.eval {args.gname} finished!", flush=True)