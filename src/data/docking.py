import sys, os, logging, pickle, struct
from functools import partial, lru_cache
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from openbabel import pybel
from ..lmdb import new_lmdb
from .data import LMDBDataset, get_random_rotation_matrix
from ..tokenizer import MoleculeProteinTokenizer
from ..utils import load_gninatypes
from rdkit import Chem


cddir = "/workspace/cheminfodata/crossdocked/CrossDocked2020"
class CDDataset(Dataset):
    logger = logging.getLogger(__qualname__)
    def __init__(self, lmdb_path, 
            atom_h,
            coord_ca, coord_heavy, coord_h):
        """
        CrossDockedの何らかのデータから取り出す。
        タンパク質は全てのタンパク質にする。

        原子:
            C: 必ず入る
            heavy: 必ず入る
        座標:
            

        """
        self.net_dataset = LMDBDataset(lmdb_path, True)

    @lru_cache(maxsize=1)
    def __getitem__(self, idx):
        data = self.net_dataset[idx]
        pocket, rec, lig, lig_cond = data['pocket'], data['rec'], data['lig'], data['lig_cond']

        # ligand SMILES
        lig_mol = next(pybel.readfile('pdb', f"{cddir}/{pocket}/{lig}_lig.pdb"))
        lig_smi = lig_mol.write().split('\t')[0]

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
        with open(input) as f:
            for line in tqdm(f):
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
                data = {
                    'label': label,
                    'pk': pk,
                    'rmsd': rmsd,
                    'pocket': pocket,
                    'lig': lig,
                    'rec': rec,
                    'lig_cond': lig_cond,
                    'score': score
                }
                txn.put(str(idx).encode('ascii'), pickle.dumps(data))
                idx+=1
        txn.commit()
        env.close()

def randomize_smiles(smi, rstate: np.random.RandomState):
    mol = Chem.MolFromSmiles(smi)
    nums = np.arange(mol.GetNumAtoms())
    rstate.shuffle(nums)
    mol = Chem.RenumberAtoms(mol, nums.tolist())
    ran = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)
    return ran

class FinetuneDataset(Dataset):
    def __init__(self, net_dataset:Dataset, 
            tokenizer: MoleculeProteinTokenizer, 
            seed:int=0):
        self.net_dataset = net_dataset
        tokenizer.add_voc('end')
        tokenizer.add_voc('score')
        self.tokenizer = tokenizer
        self.rstate = np.random.RandomState(seed)
        

    def __getitem__(self, idx):
        data = self.net_dataset[idx]

        # randomize SMILES
        lig_smi_ran = randomize_smiles(data['lig_smi'])

        # random rotations
        lig_coord = data['lig_coord']
        rec_coord = data['rec_coord']
        rotation_matrix = get_random_rotation_matrix(self.rstate)
        lig_coord = np.matmul(lig_coord, rotation_matrix)
        rec_coord = np.matmul(rec_coord, rotation_matrix)

        # adjust ligand to O
        mean = np.mean(lig_coord, axis=0)
        lig_coord -= mean
        rec_coord -= mean

        rec_atoms = self.tokenizer.tokenize_atoms(data['rec_atoms'])
        rec_coord = self.tokenizer.tokenize_coord(rec_coord)
        score = [self.tokenizer.added_voc2tok['score']]+\
            self.tokenizer.tokenize_float(data['score'])
        lig_smi = self.tokenizer.tokenize_smi(lig_smi_ran)
        lig_coord = self.tokenizer.tokenize_coord(lig_coord)

        return rec_atoms+rec_coord+score+lig_smi+lig_coord

    def __len__(self):
        return len(self.net_dataset)

