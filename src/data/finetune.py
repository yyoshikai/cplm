import sys, os, logging, pickle, struct, bisect
from glob import glob
import torch
from tqdm import tqdm
from logging import getLogger
import numpy as np
import pandas as pd
import concurrent.futures as cf
from torch.utils.data import Dataset
try:
    from openbabel import pybel
except ModuleNotFoundError:
    pybel = None
from ..lmdb import new_lmdb
from .data import LMDBDataset, get_random_rotation_matrix
from ..utils import load_gninatypes, logtime
from rdkit import Chem
from .tokenizer import ProteinAtomTokenizer, StringTokenizer, FloatTokenizer

def randomize_smiles(smi, rstate: np.random.RandomState):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smi}')
    nums = np.arange(mol.GetNumAtoms())
    rstate.shuffle(nums)
    mol = Chem.RenumberAtoms(mol, nums.tolist())
    ran = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)
    return ran

# dataset: {'lig_smi': str, 'lig_coord': np.ndarray, 'rec_atoms': list[str],
#       'rec_coord': np.ndarray, 'score': float}
class FinetuneDataset(Dataset):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, dataset:Dataset, 
            protein_atom_tokenizer: ProteinAtomTokenizer,
            smiles_tokenizer: StringTokenizer,
            coord_tokenizer: FloatTokenizer,
            seed:int=0, 
            out_ligand: bool=True, 
            coord_center: str='ligand'):
        self.dataset = dataset
        self.protein_atom_tokenizer = protein_atom_tokenizer
        self.smiles_tokenizer = smiles_tokenizer
        self.coord_tokenizer = coord_tokenizer
        self.rstate = np.random.RandomState(seed)
        self.out_ligand = out_ligand
        self.coord_center = coord_center
        assert self.coord_center in ['ligand', 'pocket', 'none']
        
    def __getitem__(self, idx):
        data = self.dataset[idx]
        with logtime(self.logger, f'[{idx}]'):
            output = []

            # normalize coords
            lig_coord = data['lig_coord']
            rec_coord = data['rec_coord']

            ## random rotation
            rotation_matrix = get_random_rotation_matrix(self.rstate)
            lig_coord = np.matmul(lig_coord, rotation_matrix)
            rec_coord = np.matmul(rec_coord, rotation_matrix)

            ## set center
            if self.coord_center == 'ligand':
                center = np.mean(lig_coord, axis=0)
            elif self.coord_center == 'pocket':
                center = np.mean(rec_coord, axis=0)
            else:
                center = np.zeros(3, dtype=float)
            lig_coord -= center
            rec_coord -= center

            # pocket
            output += ['[POCKET]']+self.protein_atom_tokenizer.tokenize(data['rec_atoms'])+\
                ['[XYZ]']+self.coord_tokenizer.tokenize_array(rec_coord.ravel())

            # score
            output += ['[SCORE]']+self.coord_tokenizer.tokenize(data['score'])

            # ligand
            if self.out_ligand:
                # randomize SMILES
                try:
                    lig_smi_ran = randomize_smiles(data['lig_smi'], self.rstate)
                except Exception as e:
                    self.logger.warning(f"Error in randomizing {data['lig_smi']}; Original SMILS is used.")
                    lig_smi_ran = data['lig_smi']

                
                ['[LIGAND]']+self.smiles_tokenizer.tokenize(lig_smi_ran)+\
                ['[XYZ]']+self.coord_tokenizer.tokenize_array(lig_coord.ravel())+\
                ['[END]']

            return output

    def __len__(self):
        return len(self.dataset)

    def vocs(self):
        return self.protein_atom_tokenizer.vocs() |\
            self.smiles_tokenizer.vocs() |\
            self.coord_tokenizer.vocs() |\
            {'[POCKET]', '[XYZ]', '[SCORE]', '[LIGAND]', '[END]'}

def process_line(line):
    label, pk, rmsd, rec_file, lig_file, score = line[:-1].split()
    score = float(score[1:])
    rec_pocket, rec = rec_file.split('/')

    lig_pocket, lig = lig_file.split('/')
    lig_rec, lig = lig.split('_rec_')
    lig, lig_cond = lig.split('_lig_')

    assert rec == lig_rec+'_rec_0.gninatypes', f"rec={rec}, lig_rec={lig_rec}"
    rec = lig_rec
    assert lig_pocket == rec_pocket, f"lig_pocket={lig_pocket}, rec_pocket={rec_pocket}"
    pocket = lig_pocket
    assert lig_cond[-11:] == '.gninatypes', lig_cond
    lig_cond = lig_cond[:-11]
    lig_mol = next(pybel.readfile('pdb', f"{cddir}/{pocket}/{lig}_lig.pdb"))
    lig_smi = lig_mol.write().split('\t')[0]
    data = {
        'label': label,
        'pk': pk,
        'rmsd': rmsd,
        'pocket': pocket,
        'lig': lig,
        'lig_smi': lig_smi,
        'rec': rec,
        'lig_cond': lig_cond,
        'score': score
    }
    data = pickle.dumps(data)
    return data

cddir = "/workspace/cheminfodata/crossdocked/CrossDocked2020"
class CDDataset(Dataset):
    logger = logging.getLogger(__qualname__)
    def __init__(self, lmdb_path):
        """
        CrossDockedの何らかのデータから取り出す。
        タンパク質は全てのタンパク質にする。
        """
        self.net_dataset = LMDBDataset(lmdb_path, True)

    def __getitem__(self, idx):
        data = self.net_dataset[idx]
        pocket, rec, lig, lig_cond, lig_smi = data['pocket'], data['rec'], \
            data['lig'], data['lig_cond'], data['lig_smi']

        # ligand coordinate
        lig_coord = load_gninatypes(f"{cddir}/{pocket}/{rec}_rec_{lig}_lig_{lig_cond}.gninatypes")
        lig_coord = np.array(lig_coord)[:, :3]
        

        # ポケットの原子と座標
        rec_atoms = []
        rec_coord = []
        with open(f"{cddir}/{pocket}/{rec}_rec.pdb") as f:
            for line in f:
                if line[:6] == 'ATOM  ':
                    atom = line[13:16]
                    if atom[:2] == 'CA':
                        rec_atoms.append('CA')
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        rec_coord.append((x, y, z))
                    elif atom[0] in ['C', 'N', 'O', 'S']:
                        rec_atoms.append(atom[0])
        rec_coord = np.array(rec_coord)
        
        return {'lig_smi': lig_smi, 'lig_coord': lig_coord,
            'rec_atoms': rec_atoms, 'rec_coord': rec_coord, 
            'score': data['score']}

    def __len__(self):
        return len(self.net_dataset)
    

    @classmethod
    def process_types(cls, input, output):
        """
        cross dockedのtypesファイルをlmdbに変換する。
        ※テキストファイルで読み込むと並列が重いため。
        """
        os.makedirs(os.path.dirname(output), exist_ok=True)

        lmdb_path = f"{output}.lmdb"
        env, txn = new_lmdb(lmdb_path)

        idx = 0
        with logtime(cls.logger, 'process time='):
            with open(input) as f:
                for idx, line in enumerate(tqdm(f, file=sys.stdout)):
                    data = process_line(line)
                    txn.put(str(idx).encode('ascii'), data)
        txn.commit()
        env.close()

class CDBindGPTDataset(Dataset):
    logger = getLogger(f'{__module__}.{__qualname__}')
    def __init__(self, save_dir, 
            pocket_atom_heavy: bool=True, pocket_atom_h: bool=False,
            pocket_coord_heavy: bool=False, pocket_coord_h: bool=False,
            mol_atom_h: bool=False, mol_coord_h: bool=True):
        """
        BindGPTに従い, 'docked'と'minimized'データのみ利用する。
        """
        self.df = pd.read_csv(f"{save_dir}/files.tsv", sep='\t', index_col=0)
        self.offsets = [0]+np.cumsum(self.df['n']).tolist()
        self.pocket_atom_heavy = pocket_atom_heavy
        self.pocket_atom_h = pocket_atom_h
        self.pocket_coord_heavy = pocket_coord_heavy
        self.pocket_coord_h = pocket_coord_h
        self.mol_atom_h = mol_atom_h
        self.mol_coord_h = mol_coord_h

    def __getitem__(self, idx: int):
        with logtime(self.logger, f"[{idx}]"):
            file_idx = bisect.bisect_right(self.offsets, idx) - 1
            mol_idx = idx - self.offsets[file_idx]
            assert 0 <= file_idx < len(self.df), f"index {idx} is out of bounds"
            file = self.df.index[file_idx]
            mol = list(Chem.SDMolSupplier(file))[mol_idx]

    def __len__(self):
        return self.offsets[-1]


    @classmethod
    def preprocess(self, save_dir):
        files = glob(f"{cddir}/*/*_lig_tt_min.sdf")\
            +glob(f"{cddir}/*/*_lig_tt_docked.sdf")
        files = sorted(files)

        ns = []
        for file in files:
            mols = len(Chem.SDMolSupplier(file))
            ns.append(len(mols))
        
        df = pd.DataFrame({'file':files, 'n_mol': ns})
        df.to_csv(f"{save_dir}/files.tsv", index=False, sep='\t')


        


