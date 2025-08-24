import os, pickle, io, logging, yaml, random
from logging import getLogger
from collections import defaultdict

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from time import time
from dataclasses import dataclass
from prody import parsePDB, parsePDBStream, confProDy, Contacts
from ..utils.lmdb import new_lmdb
from .coord_transform import get_random_rotation_matrix
from .lmdb import PickleLMDBDataset
from .data import WrapDataset
from ..utils import slice_str
from rdkit import Chem
confProDy(verbosity='none')
from ..utils.logger import add_file_handler, get_logger
from ..utils.rdkit import ignore_warning
from ..utils.utils import CompressedArray

@dataclass
class Protein:
    atoms: np.ndarray
    coord: np.ndarray

class CDDataset2(WrapDataset[tuple[Protein, Chem.Mol, float]]):
    def __init__(self, save_dir: str, out_filename: bool=False):
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
        
class MolProcessDataset(WrapDataset[tuple[str, np.ndarray]]):
    def __init__(self, mol_data: Dataset[Chem.Mol], rstate: np.random.RandomState,
            h_atom: bool=False, h_coord: bool=True):
        super().__init__(mol_data)
        self.mol_data = mol_data
        self.h_atom = h_atom
        self.h_coord = h_coord
        self.rstate = rstate
        assert not ((not self.h_atom) and self.h_coord), 'Not supported.'

    def __getitem__(self, idx: int):
        lig_mol = self.mol_data[idx]

        ## randomize
        nums = np.arange(lig_mol.GetNumAtoms())
        self.rstate.shuffle(nums)
        lig_mol = Chem.RenumberAtoms(lig_mol, nums.tolist())
        
        ## remove hydrogen
        if not self.h_atom:
            lig_mol = Chem.RemoveHs(lig_mol)

        lig_smi = Chem.MolToSmiles(lig_mol, canonical=False)
        conf_pos = lig_mol.GetConformer().GetPositions()
        atom_idxs = np.array(lig_mol.GetProp('_smilesAtomOutputOrder', autoConvert=True))
        if self.h_atom and not self.h_coord:
            atom_idxs = [idx for idx in atom_idxs if lig_mol.GetAtomWithIdx(idx).GetSymbol() != 'H']
            lig_coord = conf_pos[atom_idxs]
        else:
            lig_coord = conf_pos[atom_idxs]
        return lig_smi, lig_coord
    
class ProteinProcessDataset(WrapDataset[tuple[list[str], np.ndarray]]):
    def __init__(self, protein_data: Dataset[Protein],
            heavy_atom: bool=True, h_atom: bool=False,
            heavy_coord: bool=False, h_coord: bool=False):
        super().__init__(protein_data)
        self.protein_data = protein_data
        self.heavy_atom = heavy_atom
        self.h_atom = h_atom
        self.heavy_coord = heavy_coord
        self.h_coord = h_coord

    def __getitem__(self, idx: int):
        protein = self.protein_data[idx]
        atoms = protein.atoms
        coord = protein.coord

        # calc mask
        is_ca = atoms == 'CA'
        is_h = slice_str(atoms, 1) == 'H'
        is_heavy = (~is_ca)&(~is_h)

        # atoms 
        atom_mask = is_ca.copy()
        if self.heavy_atom: atom_mask |= is_heavy
        if self.h_atom: atom_mask |= is_h
        atoms = atoms[atom_mask]

        # coord
        coord_mask = is_ca.copy()
        if self.heavy_coord: coord_mask |= is_heavy
        if self.h_coord: coord_mask |= is_h
        coord = coord[coord_mask]

        return atoms, coord
    
class CentralizeCoordsDataset(WrapDataset[tuple[np.ndarray, ...]]):
    def __init__(self, base_coord_data: Dataset[np.ndarray], *coord_datas: list[Dataset[np.ndarray]]):
        super().__init__(base_coord_data)
        self.base_coord_data = base_coord_data
        self.coord_datas = coord_datas
    
    def __getitem__(self, idx: int):
        base_coord = self.base_coord_data[idx]
        center = np.mean(base_coord, axis=0)
        base_coord -= center
        return (center, base_coord, ) + tuple(coord_data[idx] - center 
                for coord_data in self.coord_datas)

class RandomRotateDataset(WrapDataset[tuple[np.ndarray, ...]]):
    def __init__(self, rstate: np.random.RandomState, *coord_datas: list[Dataset[np.ndarray]]):
        super().__init__(coord_datas[0])
        self.rstate = rstate
        self.coord_datas = coord_datas

    def __getitem__(self, idx: int):
        rotation_matrix = get_random_rotation_matrix(self.rstate)
        return (rotation_matrix, )+tuple(np.matmul(coord_data[idx], rotation_matrix) 
                for coord_data in self.coord_datas)


class CDDataset(Dataset):
    logger = get_logger(f"{__module__}.{__qualname__}")
    def __init__(self, protein_atoms, protein_coord, lig_smi, lig_coord, center, score, rotation_matrix, rstate,
            coord_center: str='ligand', random_rotate: bool=True):
        """
        train.py: 
            mol: atom_h=True, coord_h=True, 
            pocket: heavy_atom: bool = True, atom_h: bool = False,
                coord_heavy: bool=False, coord_h: bool = False
        BindGPTも↑と同じ。
        """
        self.protein_atoms = protein_atoms
        self.protein_coord = protein_coord
        self.lig_smi = lig_smi
        self.lig_coord = lig_coord
        self.score = score
        self.center = center
        self.rotation_matrix = rotation_matrix
        self.rstate = rstate
        self.coord_center = coord_center
        assert self.coord_center in ['ligand', 'pocket', 'none']
        self.random_rotate = random_rotate
        
    def __getitem__(self, idx):
        score = self.score[idx]
        lig_smi = self.lig_smi[idx]
        lig_coord = self.lig_coord[idx]
        pocket_atoms = self.protein_atoms[idx]
        pocket_coord = self.protein_coord[idx]
        center = self.center[idx]
        rotation_matrix = self.rotation_matrix[idx]

        output = (pocket_atoms, pocket_coord, lig_smi, lig_coord, score, center, rotation_matrix)
        return output

    def __len__(self):
        return len(self.lmdb_dataset)


class RandomScoreDataset(Dataset[float]):
    def __init__(self, min: float, max: float, size: int, seed: int):
        self.min = min
        self.max = max
        self.size = size
        self.rng = random.Random(seed)

    def __getitem__(self, idx: int):
        return self.rng.uniform(self.min, self.max)

    def __len__(self):
        return self.size