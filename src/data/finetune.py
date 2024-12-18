import sys, os, pickle, io, psutil, logging
from tqdm import tqdm
from logging import getLogger
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset
from prody import parsePDB, parsePDBStream, confProDy, Contacts
from ..utils.lmdb import new_lmdb
from .data import LMDBDataset, get_random_rotation_matrix
from ..utils import logtime
from rdkit import Chem
from .tokenizer import ProteinAtomTokenizer, StringTokenizer, FloatTokenizer
confProDy(verbosity='none')
from ..utils.logger import add_file_handler, get_logger
from ..utils.rdkit import set_rdkit_logger
from .protein import slice_str

class FinetuneDataset(Dataset):
    logger = get_logger(f"{__module__}.{__qualname__}")
    def __init__(self, save_dir: str, protein_atom_tokenizer: ProteinAtomTokenizer,
            smiles_tokenizer: StringTokenizer, coord_tokenizer: FloatTokenizer,
            seed:int=0, out_ligand: bool=True, coord_center: str='ligand', 
            pocket_atom_heavy: bool=True, pocket_atom_h: bool=False,
            pocket_coord_heavy: bool=False, pocket_coord_h: bool=False,
            mol_atom_h: bool=False, mol_coord_h: bool=True):
        """
        train.py: 
            mol: atom_h=True, coord_h=True, 
            pocket: atom_heavy: bool = True, atom_h: bool = False,
                coord_heavy: bool=False, coord_h: bool = False
        BindGPTも↑と同じ。
        """
        self.lmdb_dataset = LMDBDataset(f"{save_dir}/main.lmdb", key_is_indexed=True)
        self.protein_atom_tokenizer = protein_atom_tokenizer
        self.smiles_tokenizer = smiles_tokenizer
        self.coord_tokenizer = coord_tokenizer
        self.rstate = np.random.RandomState(seed)
        self.out_ligand = out_ligand
        self.coord_center = coord_center
        assert self.coord_center in ['ligand', 'pocket', 'none']
        self.pocket_atom_heavy = pocket_atom_heavy
        self.pocket_atom_h = pocket_atom_h
        self.pocket_coord_heavy = pocket_coord_heavy
        self.pocket_coord_h = pocket_coord_h
        self.mol_atom_h = mol_atom_h
        self.mol_coord_h = mol_coord_h
        assert not ((not self.mol_atom_h) and self.mol_coord_h), 'Not supported.'

    def __getitem__(self, idx):
        data = self.lmdb_dataset[idx]
        with logtime(self.logger, f"[{idx}]"):
            
            # ligand
            lig_mol: Chem.Mol
            lig_mol, score, = (data[key] for key in ['lig_mol', 'score'])
            score = float(score)
            
            ## randomize
            nums = np.arange(lig_mol.GetNumAtoms())
            self.rstate.shuffle(nums)
            lig_mol = Chem.RenumberAtoms(lig_mol, nums.tolist())
            
            ## remove hydrogen
            if not self.mol_atom_h:
                lig_mol = Chem.RemoveHs(lig_mol)

            lig_smi = Chem.MolToSmiles(lig_mol, canonical=False)
            conf_pos = lig_mol.GetConformer().GetPositions()
            atom_idxs = np.array(lig_mol.GetProp('_smilesAtomOutputOrder', autoConvert=True))
            if self.mol_atom_h and not self.mol_coord_h:
                atom_idxs = [idx for idx in atom_idxs if lig_mol.GetAtomWithIdx(idx).GetSymbol() != 'H']
                lig_coord = conf_pos[atom_idxs]
            else:
                lig_coord = conf_pos[atom_idxs]

            # pocket
            pocket_atoms, pocket_coord = data['pocket_atoms'], data['pocket_coordinate']

            ## calc mask
            is_ca = pocket_atoms == 'CA'
            is_h = slice_str(pocket_atoms, 1) == 'H'
            is_heavy = (~is_ca)&(~is_h)
            atom_mask = is_ca.copy()
            if self.pocket_atom_heavy: atom_mask |= is_heavy
            if self.pocket_atom_h: atom_mask |= is_h
            pocket_atoms = pocket_atoms[atom_mask]
            coord_mask = is_ca.copy()
            if self.pocket_coord_heavy: coord_mask |= is_heavy
            if self.pocket_coord_h: coord_mask |= is_h
            pocket_coord = pocket_coord[coord_mask]

            # normalize coords
            ## random rotation
            rotation_matrix = get_random_rotation_matrix(self.rstate)
            lig_coord = np.matmul(lig_coord, rotation_matrix)
            pocket_coord = np.matmul(pocket_coord, rotation_matrix)
            
            ## set center
            if self.coord_center == 'ligand':
                center = np.mean(lig_coord, axis=0)
            elif self.coord_center == 'pocket':
                center = np.mean(pocket_coord, axis=0)
            else:
                center = np.zeros(3, dtype=float)
            lig_coord -= center
            pocket_coord -= center

            # output
            output = []
            ## pocket
            output += ['[POCKET]']+self.protein_atom_tokenizer.tokenize(pocket_atoms)+\
                ['[XYZ]']+self.coord_tokenizer.tokenize_array(pocket_coord.ravel())

            ## score
            output += ['[SCORE]']+self.coord_tokenizer.tokenize(score)

            ## ligand
            if self.out_ligand:
                output += ['[LIGAND]']+self.smiles_tokenizer.tokenize(lig_smi)+\
                ['[XYZ]']+self.coord_tokenizer.tokenize_array(lig_coord.ravel())
            output += ['[END]']

            return output

    def __len__(self):
        return len(self.lmdb_dataset)
    
    def vocs(self):
        return self.protein_atom_tokenizer.vocs()|\
            self.coord_tokenizer.vocs()|\
            self.smiles_tokenizer.vocs()|{'[POCKET]', '[XYZ]', '[END]', '[SCORE]', '[LIGAND]'}

    @classmethod
    def preprocess(cls, cddir, save_dir, radius, test=False):
        os.makedirs(save_dir, exist_ok=True)
        add_file_handler(getLogger(), f"{save_dir}/root.log", level=logging.INFO)
        add_file_handler(cls.logger, f"{save_dir}/process.log")
        set_rdkit_logger()
        cls.logger.info(f"{radius=}")

        env, txn = new_lmdb(f"{save_dir}/main.lmdb")
        n_invalid = 0
        n_far = 0
        idx = 0
        cls.logger.info("Processing...")
        lig_idx = None
        if test:
            protein2n = defaultdict(int)
        with open("/workspace/cheminfodata/crossdocked/projects/survey/files/dirs.txt") as fd:
            dnames = sorted(fd.read().splitlines())
            if test: dnames = dnames[:20]
            for dname in (pbar:=tqdm(dnames)):
                with open(f"/workspace/cheminfodata/crossdocked/projects/survey/files/{dname}.txt") as ff:
                    for basename in sorted(ff.read().splitlines()):
                        if not basename.endswith(('lig_tt_docked.sdf', 'lig_tt_min.sdf')): continue

                        protein_name = basename.split('_rec_')[0]
                        protein_agroup = parsePDB(f"{cddir}/{dname}/{protein_name}_rec.pdb")
                        protein_contact = Contacts(protein_agroup)

                        # for debug
                        if test:
                            if protein2n[protein_name] >= 10:
                                continue

                        lig_sup = Chem.SDMolSupplier(f"{cddir}/{dname}/{basename}", removeHs=False)
                        for lig_idx, lig_mol in enumerate(lig_sup):
                            if lig_mol is None:
                                n_invalid+=1
                                continue
                            try:
                                # 水素付加
                                lig_mol = Chem.AddHs(lig_mol, addCoords=True)

                                # ポケットをProDyで抽出
                                lig_pdb = Chem.MolToPDBBlock(lig_mol)
                                lig_agroup = parsePDBStream(io.StringIO(lig_pdb))
                                pocket = protein_contact.select(radius, lig_agroup)
                                if pocket is None:
                                    cls.logger.warning(f"Far away ligand: {dname=}, {basename=}, {lig_idx=}")
                                    n_far += 1
                                    continue

                                # データ作成
                                pocket_atoms = pocket.getData('name')
                                pocket_coord = pocket.getCoords()
                                data = {
                                    'lig_mol': lig_mol,  
                                    'score': lig_mol.GetProp('minimizedAffinity'),
                                    'pocket_atoms': pocket_atoms,
                                    'pocket_coordinate': pocket_coord
                                }
                                txn.put(str(idx).encode('ascii'), pickle.dumps(data))
                                idx += 1

                                if idx % 100 == 0:
                                    mem = psutil.virtual_memory()
                                    pbar.set_postfix_str(f"memory={mem.used/(2**30):.03f}/{mem.total/(2**30):.03f}")
                                
                                if test:
                                    protein2n[protein_name] += 1
                                    if protein2n[protein_name] >= 10:
                                        break

                            except Exception as e:
                                cls.logger.error(f"Error at {dname, basename, lig_idx}: {e}")
                                cls.logger.error(f"{lig_mol=}")
                                cls.logger.error(f"{lig_pdb=}")
                                cls.logger.error(f"{lig_agroup=}")
                                cls.logger.error(f"{pocket=}")
                                cls.logger.error(f"{protein_agroup=}")
                                cls.logger.error(f"{protein_contact=}")
                                raise e

                        lig_idx = None
        txn.commit()
        env.close()
        cls.logger.info(f"# of data: {idx}")
        cls.logger.info(f"# of invalid mols: {n_invalid}")
        cls.logger.info(f"# of far away ligand: {n_far}")
