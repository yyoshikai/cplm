import sys, os, logging, pickle
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from ..lmdb import new_lmdb


# とりあえずやってみる。
class DockingDataset(Dataset):
    logger = logging.getLogger(__qualname__)
    def __init__(self, net_dataset):
        pass

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
                    'lig_cond': lig_cond
                }
                txn.put(str(idx).encode('ascii'), pickle.dumps(data))
                idx+=1
        txn.commit()
        env.close()
