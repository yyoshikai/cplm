import sys, os
import logging, bisect, gzip, pickle, gc
import concurrent.futures as cf
from logging import getLogger
from collections import defaultdict
from glob import glob
import psutil
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm as _tqdm
from time import time
 
from .tokenizer import ProteinAtomTokenizer, FloatTokenizer
from ..utils.lmdb import new_lmdb, load_lmdb
from .data import CoordTransform, LMDBDataset, LMDB
from ..utils import logtime
try:
    from Bio.PDB import FastMMCIFParser, PPBuilder
except ModuleNotFoundError:
    FastMMCIFParser = PPBuilder = None
from ..utils.logger import get_logger, add_file_handler

# https://stackoverflow.com/questions/39042214/how-can-i-slice-each-element-of-a-numpy-array-of-strings
def slice_str(x: np.ndarray, end: int):
    b = x.view((str,1)).reshape(len(x),-1)[:, :end]
    return np.fromstring(b.tostring(),dtype=(str,end))

# net_datasetは {'atoms': list, 'coordinate': np.ndarray} を出力すればよい。
# 水素は含んでいても含んでいなくてもよいが, atomとcoordでそろえること。
class ProteinDataset(Dataset):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, net_dataset: Dataset, atom_tokenizer: ProteinAtomTokenizer, 
            coord_tokenizer: FloatTokenizer,
            coord_transform: CoordTransform, 
            atom_heavy: bool = True, atom_h: bool = False,
            coord_heavy: bool=False, coord_h: bool = False):
        """
        Pretrain用のポケットデータを生成する。
        coord_heavy: ca, heavy, h のうちheavyのcoordを抽出するかどうか。
        """
        self.net_dataset = net_dataset
        self.atom_tokenizer = atom_tokenizer
        self.coord_tokenizer = coord_tokenizer
        self.coord_transform = coord_transform
        self.atom_heavy = atom_heavy
        self.atom_h = atom_h
        self.coord_heavy = coord_heavy
        self.coord_h = coord_h

    def __getitem__(self, idx):
        data = self.net_dataset[idx]
        with logtime(self.logger, f"[{idx}]"):
            atoms = np.array(data['atoms'])
            coords = data['coordinate']
            assert len(atoms) == len(coords)

            # calc mask
            is_ca = atoms == 'CA'
            is_h = slice_str(atoms, 1) == 'H'
            is_heavy = (~is_ca)&(~is_h)

            atom_mask = is_ca.copy()
            if self.atom_heavy: atom_mask |= is_heavy
            if self.atom_h: atom_mask |= is_h
            atoms = atoms[atom_mask]
            coord_mask = is_ca.copy()
            if self.coord_heavy: coord_mask |= is_heavy
            if self.coord_h: coord_mask |= is_h
            coords = coords[coord_mask]

            coords = self.coord_transform(coords)
            return ['[POCKET]']+self.atom_tokenizer.tokenize(atoms) \
                +['[XYZ]']+self.coord_tokenizer.tokenize_array(coords.ravel())+['[END]']

    def vocs(self) -> set[str]:
        return self.atom_tokenizer.vocs()|self.coord_tokenizer.vocs()|{'[POCKET]', '[XYZ]', '[END]'}

    def __len__(self):
        return len(self.net_dataset)

class UniMolPocketDataset(Dataset):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, lmdb_path, **kwargs):
        self.dataset = LMDBDataset(lmdb_path, **kwargs)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        with logtime(self.logger, f"[{idx}]"):
            data['coordinate'] = data.pop('coordinates')[0]
            return data

    def __len__(self):
        return len(self.dataset)

def _process0_init(radius_mean_i, radius_std_i, max_n_atom_i, logger_i: logging.Logger):
    global parser, ppbuilder, amino_resnames, radius_mean, radius_std, max_n_atom, logger
    parser = FastMMCIFParser(QUIET=True)
    ppbuilder = PPBuilder()
    amino_resnames = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        'ASX', 'GLX', 'SEC'
    }
    radius_mean = radius_mean_i
    radius_std = radius_std_i
    max_n_atom = max_n_atom_i
    logger = logger_i

def _process0_in(path):
    name = process0_get_name(path)
    try:
        start = time()
        with gzip.open(path, mode='rt') as f:
            structure = parser.get_structure("none", f)
        str_time = time() - start

        rng = np.random.default_rng(int(name, base=36))
        
        # remove non-polypeptides
        start = time()
        pps = ppbuilder.build_peptides(structure)
        peptide_time = time() - start
        if len(pps) == 0:
            return 'non_polypeptide', None

        # check atom num
        if max_n_atom is not None:
            n_atom = 0
            for atom in structure.get_atoms():
                n_atom += 1
            if n_atom > max_n_atom:
                logger.warning(f"Too large({n_atom} atoms): {path}")
                return 'large', None

        # get atom information
        start = time()
        names = []
        elements = []
        coords = []
        is_aminos = []
        for atom in structure.get_atoms():
            residue = atom.get_parent()
            resname = residue.get_resname()
            hetflag, resseq, icode = residue.get_id()
            
            is_aminos.append(resname in amino_resnames and hetflag[:2] != 'H_')
            names.append(atom.name)
            elements.append(atom.element)
            coords.append(atom.get_coord())
        atom_time = time() - start

        start = time()
        coords = np.array(coords)
        is_aminos = np.array(is_aminos)

        # calc distance matrix
        norm = np.sum(coords**2, axis=1) # [N]

        # sample substructure idxs
        sample_finished = ~is_aminos.copy()
        sub_idxss = []
        prepare_coord_time = time()-start
        
        start = time()
        while not np.all(sample_finished):
            center_idx = rng.choice(np.where(~sample_finished)[0])
            radius2 = max(1e-5, rng.normal(loc=radius_mean, scale=radius_std))**2

            dist2 = norm+norm[center_idx]-np.matmul(coords, coords[center_idx])*2

            sub_idxs = np.where(dist2 < radius2)[0]
            sample_finished[sub_idxs] = True
            sub_idxss.append(sub_idxs)
        sample_time = time()-start
        data = {
            'name': name,
            'atoms': np.array(names),
            'elements': np.array(elements),
            'coords': coords,
            'is_aminos': is_aminos,
            'sub_idxss': sub_idxss
        }
        return data, (str_time, peptide_time, atom_time, prepare_coord_time, sample_time)
    except Exception as e:
        logger.warning(f"Error in processing {path}: {e}")
        return e, None

def process0_get_name(path: str) -> str:
    return os.path.basename(path).split('.')[0]

class PDBFragmentDataset(Dataset):
    logger = logging.getLogger(__qualname__)
    def __init__(self, path):
        self.path = path
        self.dataset = LMDBDataset(path, key_is_indexed=True)
        self._lazy_offsets = None
    
    @property
    def offsets(self) -> np.ndarray:
        if self._lazy_offsets is None:
            self._lazy_offsets = pickle.load(open(self.path+".offset.pkl", 'rb'))
        return self._lazy_offsets

    def __len__(self):
        return self.offsets[-1]
    
    @property
    def n_protein(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        prot_idx = bisect.bisect_right(self.offsets, idx) - 1
        assert 0 <= prot_idx < len(self.dataset), f"index {idx} is out of bounds"
        prot = self.dataset[prot_idx]
        with logtime(self.logger, f"[{idx}]"):
            sub_idx = idx - self.offsets[prot_idx]
            sub_idxs = prot['sub_idxss'][sub_idx]
            # self.logger.debug(f"{sub_idxs=}") sub_idxsが番号順であることを確認

            atoms = prot['atoms'][sub_idxs]
            elements = prot['elements'][sub_idxs]
            elements[atoms == 'CA'] = 'CA'
            atoms = elements
            coords = prot['coords'][sub_idxs]
            return {'atoms': atoms, 'coordinate': coords}

    @classmethod
    def process0(cls, root_dir: str, out_path: str,
            radius_mean: float, radius_std: float,
            reset=False, num_workers: int=1, tqdm: bool=False, max_tasks_per_child: int=None,
            range_min: int=None, range_sup: int=None, max_n_atom: int=None):
        time_path = out_path+".0.process_time.lmdb"
        log_path = out_path+'.0.log'
        out_path = out_path+'.0.lmdb'

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        logger = get_logger(f"{cls.__module__}.{cls.__qualname__}.process0", logging.DEBUG)
        add_file_handler(logger, log_path, mode='w' if reset else 'a')
        logger.info("New process started.")

        if os.path.exists(out_path) and not reset:
            env, txn = load_lmdb(out_path)
            finished_keys = set(map(lambda x: x.decode('ascii'), txn.cursor().iternext(values=False)))
            logger.info(f"{len(finished_keys)} keys are found in out_path.")
            del env, txn
        else:
            finished_keys = None
        env, txn = new_lmdb(out_path, keep_exists=not reset)
        env_time, txn_time = new_lmdb(time_path, keep_exists=not reset)

        paths = sorted(glob(f"{root_dir}/**/*.cif.gz"))
        paths = paths[range_min:range_sup]
        if finished_keys is not None:
            paths = [path for path in paths if process0_get_name(path) not in finished_keys]
        del finished_keys
        logger.info(f"Processing {len(paths)} data...")
        
        with cf.ProcessPoolExecutor(num_workers, initializer=_process0_init, 
                initargs=(radius_mean, radius_std, max_n_atom, logger), max_tasks_per_child=max_tasks_per_child) as e:
            data_iter = e.map(_process0_in, paths, chunksize=10)
            if tqdm:
                data_iter = _tqdm(data_iter, total=len(paths))
            for i, (data, times) in enumerate(data_iter):
                key = str(process0_get_name(paths[i])).encode('ascii')
                txn.put(key, pickle.dumps(data))
                txn_time.put(key, pickle.dumps(times))
                if (i+1) % 100 == 0:
                    logger.debug(f"Committing after {i+1} process...")
                    txn.commit()
                    env.close()
                    txn_time.commit()
                    env_time.close()
                    del txn, env
                    gc.collect()
                    env, txn = new_lmdb(out_path, keep_exists=True)
                    env_time, txn_time = new_lmdb(time_path, keep_exists=True)
                    logger.debug(f"committed.")
                if tqdm and (i+1) % 100 == 0:
                    mem = psutil.virtual_memory()
                    mem = f"memory={mem.used/2**30:.03f}/{mem.total/2**30:.03f}"
                    logger.info(mem)
                    data_iter.set_postfix_str(mem)
        txn.commit()
        env.close()
        txn_time.commit()
        env_time.close()

    @classmethod
    def process1(cls, out_path: str, tqdm: bool=False):
        logger = get_logger(f"{cls.__module__}.{cls.__qualname__}.process0", logging.DEBUG)
        add_file_handler(logger, f"{out_path}.1.log", mode='w')

        env, txn = load_lmdb(out_path+".0.lmdb")
        keys = list(txn.cursor().iternext(values=False))
        keys = sorted(keys)
        logger.info(f"Number of data: {env.stat()['entries']}")

        env_out, txn_out = new_lmdb(out_path)

        idx = 0
        lengths = []
        errors = defaultdict(list)
        if tqdm:
            keys = _tqdm(keys)
        for key in keys:
            value = txn.get(key)
            data = pickle.loads(value)
            if isinstance(data, dict):
                lengths.append(len(data['sub_idxss']))
                txn_out.put(str(idx).encode('ascii'), value)
                idx += 1
            else:
                errors[data].append(key)
        env.close()
        logger.debug("Committing transaction...")
        txn_out.commit()
        logger.debug("Done.")
        env_out.close()

        
        offset = [0]+np.cumsum(lengths).tolist()
        with open(f"{out_path}.offset.pkl", 'wb') as f:
            pickle.dump(offset, f)
        logger.info(f"Dataset length: {offset[-1]}")

        with open(f"{out_path}.1.errors.pkl", 'wb') as f:
            pickle.dump(dict(errors), f)
        logger.info("Found errors:")
        for key, names in errors.items():
            logger.info(f"  {key}: {len(names)}")

def partial_load(path, start, stop):
    with open(path, 'rb') as f:
        major, minor = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f) # version compatibility
        # print(f"{init_pos=}")
        assert not fortran, "Fortran order arrays not supported"

        row_size = int(np.prod(shape[1:]))
        values = []
        v = np.fromfile(f, dtype=dtype, count=row_size*(stop-start), offset=start*row_size*dtype.itemsize)
        values.append(v)
        # print(values)
        values = np.array(values, dtype=dtype).reshape((-1,)+shape[1:])
        return values

class PDBFragment2Dataset(Dataset):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, path):
        """
        遅かった。
        """
        self.path = path
        self.sub_idxss = LMDB(f"{path}/sub_idxss.lmdb")
        with open(f"{path}/offsets.pkl", 'rb') as f:
            self.offsets = pickle.load(f)
    def __getitem__(self, idx: int):
        with logtime(self.logger, f"[{idx}]"):
            prot_idx = bisect.bisect_right(self.offsets, idx) - 1
            assert 0 <= prot_idx < len(self.offsets), f"index {idx} is out of bounds"
            sub_idx = idx - self.offsets[prot_idx]
            sub_idxs = self.sub_idxss[f"{prot_idx}_{sub_idx}".encode('ascii')]
            
            min_atom_idx = np.min(sub_idxs)
            sup_atom_idx = np.max(sub_idxs)+1

            atoms = partial_load(f"{self.path}/atoms/{prot_idx}.npy", min_atom_idx, sup_atom_idx)[sub_idxs-min_atom_idx]
            atoms = partial_load(f"{self.path}/atoms/{prot_idx}.npy", min_atom_idx, sup_atom_idx)[sub_idxs-min_atom_idx]
            elements = partial_load(f"{self.path}/elements/{prot_idx}.npy", min_atom_idx, sup_atom_idx)[sub_idxs-min_atom_idx]
            elements[atoms == 'CA'] = 'CA'
            atoms = elements
            coords = partial_load(f"{self.path}/coords/{prot_idx}.npy", min_atom_idx, sup_atom_idx)[sub_idxs-min_atom_idx]
            return {'atoms': atoms, 'coordinate': coords}

    def __len__(self):
        return self.offsets[-1]