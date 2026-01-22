import os, pickle, re
import numpy as np
from rdkit import Chem
from openbabel.openbabel import OBMol, OBConversion

from ...utils.lmdb import load_lmdb
from ..protein import Pocket
from ..data import TupleDataset

WORKDIR = os.environ.get('WORKDIR', __file__.split('/cplm/')[0])
DEFAULT_CD1_1_TYPES_DIR = f"{WORKDIR}/cheminfodata/crossdocked/CrossDocked2020_v1.1_types"
DEFAULT_TARGETDIFF_DIR = f"{WORKDIR}/cheminfodata/crossdocked/targetdiff"

# 処理を確認 @ /workspace/cplm/experiments/250824_modify_data_code/source.ipynb
class TargetDiffScafCDDataset(TupleDataset[tuple[Pocket, Chem.Mol, float, str, str]]):
    def __init__(self, split: str, targetdiff_dir: str=DEFAULT_TARGETDIFF_DIR, 
            crossdocked_dir: str=DEFAULT_CD1_1_TYPES_DIR):
        """
        NOTE: 
            For ligand, hydrogen is NOT included
            For pocket, hydrogen is NOT included 
            in this dataset.

        - f"{self.targetdiff_dir}/mask/{split}_idxs.npy" は /workspace/cheminfodata/crossdocked/targetdiff/make_mask.pyで作成
        - 
        
        """
        super().__init__(5)
        self.targetdiff_dir = targetdiff_dir
        self.crossdocked_dir = crossdocked_dir
        self.lmdb_path = f"{self.targetdiff_dir}/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"
        self.score_lmdb_path = f"{self.targetdiff_dir}/crossdocked_v1.1_rmsd1.0_pocket10_processed_final_score.lmdb"
        self.key_idxs = np.load(f"{self.targetdiff_dir}/mask/split_mol/250920_0/{split}_idxs.npy")

        self.pfname_pattern = re.compile(r"(.+?/.+?_rec)_.+")

    def __getitem__(self, idx: int):
        env, txn = load_lmdb(self.lmdb_path)

        key_idx = self.key_idxs[idx]
        key = str(key_idx).encode('ascii')
        value = txn.get(key)
        data = pickle.loads(value)

        # Pocket
        pocket = Pocket(np.array(data['protein_atom_name']), data['protein_pos'].numpy())

        # Get actual file path
        ligand_filename = data['ligand_filename']
        ligand_path = f"{self.targetdiff_dir}/crossdocked_v1.1_rmsd1.0/{ligand_filename}"
        protein_path = f"{self.targetdiff_dir}/crossdocked_v1.1_rmsd1.0/" \
            +re.match(self.pfname_pattern, data['protein_filename']).group(1)+'.pdb'

        # Molecule
        with open(ligand_path) as f:
            mol = Chem.MolFromMolBlock(f.read()) # 260122 removeHs=False を入れたほうがよい？
        
        # Score
        score_env, score_txn = load_lmdb(self.score_lmdb_path)
        score = pickle.loads(score_txn.get(key))

        return pocket, mol, score, protein_path, ligand_path

    def __len__(self):
        return len(self.key_idxs)

class TargetDiffScafCDProteinDataset(TupleDataset[tuple[OBMol, Chem.Mol, float]]):
    def __init__(self, split: str, targetdiff_dir: str=DEFAULT_TARGETDIFF_DIR, 
            crossdocked_dir: str=DEFAULT_CD1_1_TYPES_DIR):
        """
        NOTE: 
            For ligand, hydrogen is NOT included
            For pocket, hydrogen is NOT included 
            in this dataset.

        - f"{self.targetdiff_dir}/mask/{split}_idxs.npy" は /workspace/cheminfodata/crossdocked/targetdiff/make_mask.pyで作成
        - 
        
        """
        super().__init__(5)
        self.targetdiff_dir = targetdiff_dir
        self.crossdocked_dir = crossdocked_dir
        self.lmdb_path = f"{self.targetdiff_dir}/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"
        self.score_lmdb_path = f"{self.targetdiff_dir}/crossdocked_v1.1_rmsd1.0_pocket10_processed_final_score.lmdb"
        self.key_idxs = np.load(f"{self.targetdiff_dir}/mask/split_mol/250920_0/{split}_idxs.npy")

        self.pfname_pattern = re.compile(r"(.+?/.+?_rec)_.+")
        self.obc = OBConversion()
        self.obc.SetInFormat('pdb')
        
    def __getitem__(self, idx: int):
        env, txn = load_lmdb(self.lmdb_path)

        key_idx = self.key_idxs[idx]
        key = str(key_idx).encode('ascii')
        value = txn.get(key)
        data = pickle.loads(value)

        # Get actual file path
        ligand_filename = data['ligand_filename']
        ligand_path = f"{self.targetdiff_dir}/crossdocked_v1.1_rmsd1.0/{ligand_filename}"
        protein_path = f"{self.targetdiff_dir}/crossdocked_v1.1_rmsd1.0/" \
            +re.match(self.pfname_pattern, data['protein_filename']).group(1)+'.pdb'


        # Pocket
        protein = OBMol()
        with open(protein_path) as f:
            self.obc.ReadString(f.read())

        # Molecule
        ligand_filename = data['ligand_filename']
        with open(ligand_path) as f:
            mol = Chem.MolFromMolBlock(f.read()) # 260122 removeHs=False を入れたほうがよい？
        
        # Score
        score_env, score_txn = load_lmdb(self.score_lmdb_path)
        score = pickle.loads(score_txn.get(key))

        return protein, mol, score, protein_path, ligand_path

    def __len__(self):
        return len(self.key_idxs)





