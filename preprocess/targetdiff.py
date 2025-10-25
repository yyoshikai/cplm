import sys, os, logging
import argparse
sys.path.append(".")
import pickle
from src.data.datasets.crossdocked import CDDataset
from src.utils.logger import get_logger, add_file_handler
from src.utils.rdkit import ignore_warning
from src.utils.lmdb import load_lmdb, new_lmdb
from src.data.datasets.targetdiff import DEFAULT_TARGETDIFF_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--output")
parser.add_argument("--radius", type=int, default=4)
parser.add_argument("--rank", type=int, required=True)
parser.add_argument("--size", type=int, required=True)
args = parser.parse_args()

# Environment
result_dir = f"preprocess/targetdiff/{args.output}/workers/{args.rank}"
os.makedirs(result_dir, exist_ok=True)
logger = get_logger(stream=True)
add_file_handler(logger, f"{result_dir}/debug.log")
logger.info(f"{args=}")
ignore_warning()

data_env, data_txn = load_lmdb(f"{DEFAULT_TARGETDIFF_DIR}/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb")
env, txn = new_lmdb(f"{result_dir}/main.lmdb")

n_invalid = 0
n_far = 0
idx = 0
logger.info("Processing...")
lig_idx = None
data_dnames = []
data_lig_names = []
data_protein_names = []
data_sdf_idxs = []

for i, idx in enumerate(range(args.rank, data_env.stat()['entries'], args.rank)):
    key = str(idx).encode('ascii')
    value = txn.get(key)
    data = pickle.loads(data)

sys.exit()

with open("/workspace/cheminfodata/crossdocked/projects/survey/files/dirs.txt") as fd:
    dnames = sorted(fd.read().splitlines())
    if size is not None:
        dnames = dnames[rank::size]
    for idir, dname in enumerate(dnames):
        with open(f"/workspace/cheminfodata/crossdocked/projects/survey/files/{dname}.txt") as ff:
            for basename in sorted(ff.read().splitlines()):
                if not basename.endswith(ends): continue

                protein_name = basename.split('_rec_')[0]
                protein_agroup = parsePDB(f"{cddir}/{dname}/{protein_name}_rec.pdb")
                protein_contact = Contacts(protein_agroup)

                # for debug
                if test:
                    if protein2n[protein_name] >= 10:
                        continue
                
                lig_sup = Chem.SDMolSupplier(f"{cddir}/{dname}/{basename}", removeHs=False)
                t_addh = 0
                t_prody = 0
                t_data = 0
                for lig_idx, lig_mol in enumerate(lig_sup):
                    if lig_mol is None:
                        n_invalid+=1
                        continue
                    try:
                        # 水素付加
                        start = time()
                        lig_mol = Chem.AddHs(lig_mol, addCoords=True)
                        t_addh += time()-start

                        # ポケットをProDyで抽出
                        start = time()
                        lig_pdb = Chem.MolToPDBBlock(lig_mol)
                        lig_agroup = parsePDBStream(io.StringIO(lig_pdb))
                        pocket = protein_contact.select(radius, lig_agroup)
                        if pocket is None:
                            n_far += 1
                            continue
                        t_prody += time() - start

                        # データ作成
                        start = time()
                        pocket_atoms = pocket.getData('name')
                        pocket_coord = pocket.getCoords()
                        data = {
                            'lig_mol': lig_mol,  
                            'score': lig_mol.GetProp('minimizedAffinity'),
                            'pocket_atoms': pocket_atoms,
                            'pocket_coordinate': pocket_coord
                        }
                        data_dnames.append(dname)
                        data_lig_names.append(basename)
                        data_protein_names.append(f"{protein_name}_rec.pdb")
                        data_sdf_idxs.append(lig_idx)
                        txn.put(str(idx).encode('ascii'), pickle.dumps(data))
                        idx += 1
                        t_data += time() - start
                        
                        if test:
                            protein2n[protein_name] += 1
                            if protein2n[protein_name] >= 10:
                                break

                    except Exception as e:
                        logger.error(f"Error at {dname, basename, lig_idx}: {e}")
                        logger.error(f"{lig_mol=}")
                        logger.error(f"{lig_pdb=}")
                        logger.error(f"{lig_agroup=}")
                        logger.error(f"{pocket=}")
                        logger.error(f"{protein_agroup=}")
                        logger.error(f"{protein_contact=}")
                        raise e
                    
                lig_idx = None
        logger.info(f'finished: ({idir}){dname}')
txn.commit()
env.close()
df = pd.DataFrame({'dname': data_dnames, 'lig_name': data_lig_names, 
    'protein_name': data_protein_names, 'sdf_idx': data_sdf_idxs})
df.to_csv(f"{save_dir}/filenames.csv.gz", index_label='idx')
logger.info(f"# of data: {idx}")
logger.info(f"# of invalid mols: {n_invalid}")
logger.info(f"# of far away ligand: {n_far}")

