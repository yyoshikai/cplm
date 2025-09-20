import os, pickle
import numpy as np
from rdkit import Chem
from prody import parsePDB, parsePDBStream, confProDy, Contacts, addMissingAtoms

from ...utils.lmdb import load_lmdb
from ..protein import Protein
from ..data import TupleDataset

WORKDIR = os.environ.get('WORKDIR', __file__.split('/cplm/')[0])
DEFAULT_CD1_1_TYPES_DIR = f"{WORKDIR}/cheminfodata/crossdocked/CrossDocked2020_v1.1_types"
DEFAULT_TARGETDIFF_DIR = f"{WORKDIR}/cheminfodata/crossdocked/targetdiff"

# 処理を確認 @ /workspace/cplm/experiments/250824_modify_data_code/source.ipynb
class TargetDiffScafCDDataset(TupleDataset[tuple[Protein, Chem.Mol, float]]):
    def __init__(self, split: str, targetdiff_dir: str=DEFAULT_TARGETDIFF_DIR, 
            crossdocked_dir: str=DEFAULT_CD1_1_TYPES_DIR, out_fname: bool=False):
        """
        NOTE: 
            For ligand, hydrogen is NOT included
            For pocket, hydrogen is NOT included 
            in this dataset.

        - f"{self.targetdiff_dir}/mask/{split}_idxs.npy" は /workspace/cheminfodata/crossdocked/targetdiff/make_mask.pyで作成
        - 
        
        """
        super().__init__(5 if out_fname else 3)
        self.targetdiff_dir = targetdiff_dir
        self.crossdocked_dir = crossdocked_dir
        self.out_fname = out_fname
        self.lmdb_path = f"{self.targetdiff_dir}/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"
        self.score_lmdb_path = f"{self.targetdiff_dir}/crossdocked_v1.1_rmsd1.0_pocket10_processed_final_score.lmdb"
        self.key_idxs = np.load(f"{self.targetdiff_dir}/mask/split_mol/250920_0/{split}_idxs.npy")

    def __getitem__(self, idx: int):
        env, txn = load_lmdb(self.lmdb_path)

        key_idx = self.key_idxs[idx]
        key = str(key_idx).encode('ascii')
        value = txn.get(key)
        data = pickle.loads(value)

        # Pocket
        pocket = Protein(np.array(data['protein_atom_name']), data['protein_pos'].numpy())

        # Molecule
        ligand_filename = data['ligand_filename']
        with open(f"{self.targetdiff_dir}/crossdocked_v1.1_rmsd1.0/{ligand_filename}") as f:
            mol = Chem.MolFromMolBlock(f.read())
        
        # Score
        score_env, score_txn = load_lmdb(self.score_lmdb_path)
        score = pickle.loads(score_txn.get(key))

        output = (pocket, mol, score)
        if self.out_fname:
            output += (data['protein_filename'], data['ligand_filename'])

        return output

    def __len__(self):
        return len(self.key_idxs)

class TargetDiffScafCDProteinDataset(TupleDataset[tuple[Protein, Chem.Mol, float]]):
    def __init__(self, split: str, targetdiff_dir: str=DEFAULT_TARGETDIFF_DIR, 
            crossdocked_dir: str=DEFAULT_CD1_1_TYPES_DIR, out_fname: bool=False):
        """
        NOTE: 
            For ligand, hydrogen is NOT included
            For pocket, hydrogen is NOT included 
            in this dataset.

        - f"{self.targetdiff_dir}/mask/{split}_idxs.npy" は /workspace/cheminfodata/crossdocked/targetdiff/make_mask.pyで作成
        - 
        
        """
        super().__init__(5 if out_fname else 3)
        self.targetdiff_dir = targetdiff_dir
        self.crossdocked_dir = crossdocked_dir
        self.out_fname = out_fname
        self.lmdb_path = f"{self.targetdiff_dir}/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"
        self.score_lmdb_path = f"{self.targetdiff_dir}/crossdocked_v1.1_rmsd1.0_pocket10_processed_final_score.lmdb"
        self.key_idxs = np.load(f"{self.targetdiff_dir}/mask/split_mol/250920_0/{split}_idxs.npy")

    def __getitem__(self, idx: int):
        env, txn = load_lmdb(self.lmdb_path)

        key_idx = self.key_idxs[idx]
        key = str(key_idx).encode('ascii')
        value = txn.get(key)
        data = pickle.loads(value)

        # Pocket
        protein_filename = data['protein_filename']
        protein_filename = protein_filename.split('_rec_')[0]+'_rec.pdb'
        protein = parsePDB(f"{self.targetdiff_dir}/crossdocked_v1.1_rmsd1.0/{protein_filename}")
        protein = Protein(protein.getData('name'), protein.getCoords())

        # Molecule
        ligand_filename = data['ligand_filename']
        with open(f"{self.targetdiff_dir}/crossdocked_v1.1_rmsd1.0/{ligand_filename}") as f:
            mol = Chem.MolFromMolBlock(f.read())
        
        # Score
        score_env, score_txn = load_lmdb(self.score_lmdb_path)
        score = pickle.loads(score_txn.get(key))

        output = (protein, mol, score)
        if self.out_fname:
            output += (data['protein_filename'], data['ligand_filename'])

        return output

    def __len__(self):
        return len(self.key_idxs)





