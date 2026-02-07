import sys, os, shutil
from argparse import ArgumentParser
import pandas as pd
import openbabel.openbabel as ob
from tqdm import tqdm
from rdkit import Chem
from src.data.datasets.posebusters import PosebustersV2ProteinDataset
from src.chem import get_coord_from_mol, set_conf, pdb_path2obmol
from src.utils.logger import set_third_party_logger
sys.path.append("/workspace/github/posebusters")
from posebusters.posebusters import PoseBusters

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gname', required=True)
    args = parser.parse_args()
    gdir = f"generate/redock/{args.gname}"
    set_third_party_logger()

    data = PosebustersV2ProteinDataset()
    """for data_idx in tqdm(range(len(data)), desc='centering ligand', dynamic_ncols=True):
        obc = ob.OBConversion()
        obc.SetInFormat('pdb')
        pdb_id = data.ids[data_idx]
        raw_rec = pdb_path2obmol(f"{data.pb_dir}/posebusters_benchmark_set/{pdb_id}/{pdb_id}_protein.pdb")
        raw_center = get_coord_from_mol(raw_rec).mean(axis=0)
        for trial_idx in range(5):
            lig_path = f"{gdir}/new_sdf/{data_idx}/{trial_idx}.sdf"
            if not os.path.exists(lig_path): continue
            with open(lig_path) as f:
                lig = Chem.MolFromMolBlock(f.read(), removeHs=False)
            conf = lig.GetConformer()
            coord = conf.GetPositions()
            coord += raw_center
            set_conf(conf, coord)
            os.makedirs(f"{gdir}/centered_new_sdf/{data_idx}", exist_ok=True)
            with open(f"{gdir}/centered_new_sdf/{data_idx}/{trial_idx}.sdf", 'w') as f:
                f.write(Chem.MolToMolBlock(lig))
    """
    
    file_paths = []
    if os.path.exists(f"{gdir}/dirs"):
        shutil.rmtree(f"{gdir}/dirs")
    for data_idx in range(len(data)):
        pdb_id = data.ids[data_idx]
        os.makedirs(f"{gdir}/dirs/{data_idx}", exist_ok=True)
        raw_rec_path = f"{data.pb_dir}/posebusters_benchmark_set/{pdb_id}/{pdb_id}_protein.pdb"
        os.symlink(
            raw_rec_path,
            f"{gdir}/dirs/{data_idx}/raw_rec.pdb"
        )
        raw_lig_path = f"{data.pb_dir}/posebusters_benchmark_set/{pdb_id}/{pdb_id}_ligand.sdf"
        os.symlink(
            raw_lig_path,
            f"{gdir}/dirs/{data_idx}/raw_ligand.sdf"
        )
        for trial_idx in range(5):
            lig_path = f"{gdir}/centered_new_sdf/{data_idx}/{trial_idx}.sdf"
            if os.path.exists(lig_path):
                os.symlink(f"../../centered_new_sdf/{data_idx}/{trial_idx}.sdf", f"{gdir}/dirs/{data_idx}/centered_new_{trial_idx}.sdf")
                file_paths.append((lig_path, raw_lig_path, raw_rec_path))

    pb = PoseBusters('redock', None, max_workers=None, chunk_size=100)
    pb.file_paths = pd.DataFrame(file_paths, columns=['mol_pred', 'mol_true', 'mol_cond'])
    results = pb._run()
    data = {}
    for (path, _molecule_name, _position), result in tqdm(results, total=len(pb.file_paths), dynamic_ncols=True, desc="posebusters"):
        *_, data_idx, trial_idx = path.split('/')
        data_idx, trial_idx = int(data_idx), int(trial_idx.split('.')[0])
        data[data_idx, trial_idx] = result
    df = pb._make_table(data, pb.config, full_report=True)
    df.index.names = ["data_idx", "trial"]
    df.sort_index(inplace=True)
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    df.to_csv(f"{gdir}/posebusters.csv")
