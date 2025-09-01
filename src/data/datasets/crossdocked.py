import os, pickle, io, logging, yaml
from logging import getLogger
from collections import defaultdict
from time import time

import pandas as pd
from prody import parsePDB, parsePDBStream, confProDy, Contacts, addMissingAtoms
from ...utils.lmdb import new_lmdb
from ..lmdb import PickleLMDBDataset
from ..data import WrapDataset
from rdkit import Chem
confProDy(verbosity='none')
from ...utils.logger import add_file_handler, get_logger
from ...utils.rdkit import ignore_warning
from ...utils.utils import CompressedArray
from ..protein import Protein

WORKDIR = os.environ.get('WORKDIR', "/workspace")
DEFAULT_SAVE_DIR = f"{WORKDIR}/cplm/ssd/preprocess/results/finetune/r4_all"

class CDDataset(WrapDataset[tuple[Protein, Chem.Mol, float]]):
    def __init__(self, save_dir: str=DEFAULT_SAVE_DIR, out_filename: bool=False):
        self.lmdb_dataset = PickleLMDBDataset(f"{save_dir}/main.lmdb", idx_to_key='str')
        super().__init__(self.lmdb_dataset)
        self.out_filename = out_filename
        if self.out_filename:
            self.logger.info("Loading filenames.csv.gz ... ")
            df = pd.read_csv(f"{save_dir}/filenames.csv.gz")
            self.logger.info("loaded.")
            self.df = {'idx': df['idx'].values}
            for key in ['dname', 'lig_name', 'protein_name']:
                self.df[key] = CompressedArray(df[key].values)
            self.df['sdf_idx'] = df['sdf_idx'].values
            del df

    def __getitem__(self, idx: int):
        data = self.lmdb_dataset[idx]

        # ligand
        lig_mol: Chem.Mol = data['lig_mol']
        score = float(data['score'])

        # pocket
        pocket_atoms, pocket_coord = data['pocket_atoms'], data['pocket_coordinate']
        protein = Protein(pocket_atoms, pocket_coord)
        
        output = (protein, lig_mol, score)
        if self.out_filename:
            output += ({key: self.df[key][idx] for key in self.df}, )
        return output
    
    
    @classmethod
    def preprocess(cls, args, cddir, save_dir, radius, 
            ends=['lig_tt_docked.sdf', 'lig_tt_min.sdf'], 
            map_size: int=int(100e9), test=False, rank=None, size=None):
        if size is not None:
            assert rank is not None
            save_dir = f"{save_dir}/{rank}"
        os.makedirs(save_dir, exist_ok=True)
        add_file_handler(getLogger(), f"{save_dir}/root.log", level=logging.INFO)
        ignore_warning()
        logger = get_logger(f"{cls.__name__}.preprocess")
        ends = tuple(ends)
        with open(f"{save_dir}/args.yaml", 'w') as f:
            yaml.dump(args, f)
        env, txn = new_lmdb(f"{save_dir}/main.lmdb", map_size=map_size)
        n_invalid = 0
        n_far = 0
        idx = 0
        logger.info("Processing...")
        lig_idx = None
        if test:
            protein2n = defaultdict(int)
        data_dnames = []
        data_lig_names = []
        data_protein_names = []
        data_sdf_idxs = []
        with open("/workspace/cheminfodata/crossdocked/projects/survey/files/dirs.txt") as fd:
            dnames = sorted(fd.read().splitlines())
            if test: dnames = dnames[:20]
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

class CDProteinDataset(WrapDataset[tuple[Protein, Chem.Mol, float]]):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, save_dir: str, cddir: str=f"{WORKDIR}/cheminfodata/crossdocked/CrossDocked2020",  out_filename: bool=False):
        self.lmdb_dataset = PickleLMDBDataset(f"{save_dir}/main.lmdb", idx_to_key='str')
        super().__init__(self.lmdb_dataset)
        self.out_filename = out_filename
        self.cddir = cddir
        os.makedirs("tmp", exist_ok=True)

        self.logger.info("Loading filenames.csv.gz ... ")
        df = pd.read_csv(f"{save_dir}/filenames.csv.gz")
        self.logger.info("loaded.")
        self.df = {'idx': df['idx'].values}
        for key in ['dname', 'lig_name', 'protein_name']:
            self.df[key] = CompressedArray(df[key].values)
        self.df['sdf_idx'] = df['sdf_idx'].values
        del df

    def __getitem__(self, idx: int):
        data = self.lmdb_dataset[idx]

        # ligand
        lig_mol: Chem.Mol = data['lig_mol']
        score = float(data['score'])

        # protein
        dname = self.df['dname'][idx]
        protein_name = self.df['protein_name'][idx]
        addMissingAtoms(f"{self.cddir}/{dname}/{protein_name}", outfile=f"./tmp/{dname}_{protein_name}.pdb")
        protein = parsePDB(f"./tmp/{dname}_{protein_name}.pdb")
        protein = Protein(protein.getData('name'), protein.getCoords())
        
        output = (protein, lig_mol, score)
        if self.out_filename:
            output += ({key: self.df[key][idx] for key in self.df}, )
        return output
