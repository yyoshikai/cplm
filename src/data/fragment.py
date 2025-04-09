
import sys, os
import logging, bisect, gzip, pickle, gc, yaml, math
from logging import getLogger
import concurrent.futures as cf
from collections import defaultdict
from glob import glob
from time import time
import psutil
from tqdm import tqdm as _tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from .data import LMDBDataset
from ..utils import logtime, slice_str
from ..utils.lmdb import new_lmdb, load_lmdb
try:
    from Bio.PDB import FastMMCIFParser, PPBuilder
except ModuleNotFoundError:
    FastMMCIFParser = PPBuilder = None
from ..utils.logger import get_logger, add_file_handler

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
        assert not fortran, "Fortran order arrays not supported"

        row_size = int(np.prod(shape[1:]))
        values = []
        v = np.fromfile(f, dtype=dtype, count=row_size*(stop-start), offset=start*row_size*dtype.itemsize)
        values.append(v)
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


def _process3_init(sampler_i, max_n_atom_i, device_i):
    global parser, ppbuilder, amino_resnames, max_n_atom,\
         sampler, device, logger
    parser = FastMMCIFParser(QUIET=True)
    ppbuilder = PPBuilder()
    amino_resnames = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
        'ASX', 'GLX', 'SEC'
    }
    max_n_atom = max_n_atom_i
    device = device_i
    logger = getLogger(f"{__name__}.process_fragment3")
    sampler = sampler_i


def _process3_in(path):
    name = process0_get_name(path)
    try:
        with gzip.open(path, mode='rt') as f:
            structure = parser.get_structure("none", f)
        rng = torch.Generator(device=device)
        rng.manual_seed(int(name, base=36))
        # remove non-polypeptides
        pps = ppbuilder.build_peptides(structure)
        if len(pps) == 0:
            return 'non_polypeptide'

        # check atom num
        if max_n_atom is not None:
            n_atom = 0
            for atom in structure.get_atoms():
                n_atom += 1
            if n_atom > max_n_atom:
                logger.warning(f"Too large({n_atom} atoms): {path}")
                return 'large'
        # get atom information
        names = []
        coords = []
        is_aminos = []
        for atom in structure.get_atoms():
            residue = atom.get_parent()
            resname = residue.get_resname()
            hetflag, resseq, icode = residue.get_id()
            
            is_aminos.append(resname in amino_resnames and hetflag[:2] != 'H_')
            names.append(atom.name)
            coords.append(atom.get_coord())

        names = np.array(names)
        elements = slice_str(names, 1)
        coords = np.array(coords)
        coords = torch.tensor(coords, device=device) # [N, 3]
        is_aminos = torch.tensor(is_aminos, device=device) # [N]
        is_heavy = torch.tensor(elements != 'H', device=device)
        n_heavy = torch.sum(is_heavy).item()

        # calc distance matrix
        norm = torch.sum(coords**2, axis=1) # [N]

        # sample substructure idxs
        sample_remain = is_aminos.clone().detach()
        
        datas = []
        sub_idx = 0
        while torch.any(sample_remain):
            sample_remain_idxs = torch.argwhere(sample_remain)[:,0]
            center_idx = sample_remain_idxs[
                torch.randint(len(sample_remain_idxs), size=tuple(), generator=rng, device=device).item()
            ]
            center_dist = norm+norm[center_idx]-torch.mv(coords, coords[center_idx])*2

            n_sub_atom = min(n_heavy, sampler.sample(rng))
            sub_max_dist = torch.kthvalue(center_dist[is_heavy], n_sub_atom).values.item()
            sub_atom_idxs = torch.argwhere(center_dist <= sub_max_dist)[:,0]

            sample_remain[sub_atom_idxs] = False
            data = {
                'name': name,
                'sub_idx': sub_idx,
                # 'atom_idxs': sub_atom_idxs, # for debug
                'atoms': names[sub_atom_idxs.cpu().numpy()],
                'coordinate': coords[sub_atom_idxs].cpu().numpy(),
                # 'amino_idxs': torch.argwhere(is_aminos)[:,0].tolist() # for debug
            }
            datas.append(pickle.dumps(data))
            sub_idx+=1
        return datas
    except Exception as e:
        logger.warning(f"Error in processing {path}: {e}")
        return e.__class__.__name__

class NAtomSampler:
    def sample(self, generator: torch.Generator):
        raise NotImplementedError
    def __repr__(self):
        raise NotImplementedError
class LognormalSampler(NAtomSampler):
    def __init__(self, logmean: float, logstd: float, min: int, max: int):
        self.logmean = logmean
        self.logstd = logstd
        self.min = int(min)
        self.max = int(max)
    def sample(self, generator: torch.Generator):
        n = round(math.exp(torch.randn([], generator=generator, device=generator.device).item()*self.logstd+self.logmean))
        if n < self.min: n = self.min
        if n > self.max: n = self.max
        return n
    def __repr__(self):
        return f"LognormalSampler(logmean={self.logmean}, logstd={self.logstd}, "\
            f"min={self.min}, max={self.max})"

def process_fragment3(kwargs: dict, root_dir: str, out_dir: str,
        sampler: NAtomSampler, reset=False, num_workers: int=1, tqdm: bool=False, 
        max_tasks_per_child: int=None,
        idx_min: int=None, idx_sup: int=None, max_n_atom: int=None, 
        device: torch.device=torch.device('cpu')):
    """
    PDBFragmentDatasetと比べ, 
        原子数についてポケットの大きさを決める。
        データをポケットごとに保存する。
    結果はLMDBとしてそのまま使えるのでクラスは作らない。
    """
    lmdb_path = f"{out_dir}/main.lmdb"

    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger(f"{__name__}.process_fragment3", logging.DEBUG)
    add_file_handler(logger, f"{out_dir}/debug.log", mode='w' if reset else 'a')
    logger.info("New process started.")

    kwargs['sampler'] = repr(sampler)
    with open(f"{out_dir}/args.yaml", 'w') as f:
        yaml.dump(kwargs, f)

    if os.path.exists(lmdb_path) and not reset:
        env, txn = load_lmdb(lmdb_path)
        finished_keys = set(map(lambda x: x.decode('ascii'), txn.cursor().iternext(values=False)))
        logger.info(f"{len(finished_keys)} keys are found in lmdb_path.")
        del env, txn
    else:
        finished_keys = None
    env, txn = new_lmdb(lmdb_path, keep_exists=not reset)

    paths = sorted(glob(f"{root_dir}/**/*.cif.gz", recursive=True))
    paths = paths[idx_min:idx_sup]
    if finished_keys is not None:
        paths = [path for path in paths if process0_get_name(path) not in finished_keys]
    del finished_keys
    logger.info(f"Processing {len(paths)} data...")
    errors = defaultdict(int)
    if num_workers > 1:
        with cf.ProcessPoolExecutor(num_workers, initializer=_process3_init, 
                initargs=(sampler, max_n_atom, device), max_tasks_per_child=max_tasks_per_child) as e:
            data_iter = e.map(_process3_in, paths, chunksize=10)
            if tqdm:
                data_iter = _tqdm(data_iter, total=len(paths))
            idx = 0
            for datas in data_iter:
                if isinstance(datas, str):
                    errors[datas]+=1
                else:
                    for data in datas:
                        txn.put(str(idx).encode('ascii'), data)
                        idx+=1
                        if idx % 10000 == 0:
                            logger.debug(f"Committing after {idx} process...")
                            txn.commit()
                            env.close()
                            del txn, env
                            gc.collect()
                            env, txn = new_lmdb(lmdb_path, keep_exists=True)
                            logger.debug(f"committed.")
                        if tqdm and idx % 100 == 0:
                            mem = psutil.virtual_memory()
                            mem = f"memory={mem.used/2**30:.03f}/{mem.total/2**30:.03f}"
                            logger.info(mem)
                            data_iter.set_postfix_str(mem)
    else:
        _process3_init(sampler, max_n_atom, device)
        idx = 0
        if tqdm:
            paths = _tqdm(paths)
        for path in paths:
            datas = _process3_in(path)
            if isinstance(datas, str):
                errors[datas]+=1
            else:
                for data in datas:
                    txn.put(str(idx).encode('ascii'), data)
                    idx+=1
                    if idx % 10000 == 0:
                        logger.debug(f"Committing after {idx} process...")
                        txn.commit()
                        env.close()
                        del txn, env
                        gc.collect()
                        env, txn = new_lmdb(lmdb_path, keep_exists=True)
                        logger.debug(f"committed.")
                    if tqdm and idx % 100 == 0:
                        mem = psutil.virtual_memory()
                        mem = f"memory={mem.used/2**30:.03f}/{mem.total/2**30:.03f}"
                        logger.info(mem)
                        paths.set_postfix_str(mem)

    txn.commit()
    env.close()
