import os
import itertools as itr
import concurrent.futures as cf
from time import time
from collections.abc import Generator
from logging import getLogger
import numpy as np
from rdkit import Chem
from rdkit.Chem import Conformer
from rdkit.Chem.rdDetermineBonds import DetermineBondOrders
from rdkit.Geometry import Point3D
from openbabel.openbabel import OBMol, OBConversion
from src.utils import should_show
from src.utils.path import make_pardir
from src.model import Streamer, WrapperStreamer
from src.data.molecule import element_symbols
from src.data.tokenizer import FloatTokenizer, SmilesTokenizer, VocEncoder
from src.evaluate import eval_vina, eval_qvina, obmol2pdb

def coord_streamer(n_atom: int, start_position: int, new_coord_path: str|None, voc_encoder: VocEncoder, coord_range: float, no_token_range: bool, atom_order: bool, center: np.ndarray|None) -> Generator[tuple[bool, list[int], list[int]], list[int], tuple[np.ndarray|None, int, str|None]]:
    """
    Returns
    -------
    coordss: np.ndarray|None of 
        Shape: (n_atom, 3)
    pos: int
        next position
    error: str|None
        None if no error
    """

    coord_tokenizer = FloatTokenizer('', -coord_range, coord_range)
    if no_token_range:
        int_token_range = frac_token_range = list(range(voc_encoder.voc_size))
    else:
        int_token_range = sorted(voc_encoder.encode(coord_tokenizer.int_vocs()))
        frac_token_range = sorted(voc_encoder.encode(coord_tokenizer.frac_vocs()))
    pos = start_position
    if new_coord_path is not None:
        make_pardir(new_coord_path)
        with open(new_coord_path, 'w') as f:
            f.write("idx,x,y,z\n")
    coordss = []
    for i_atom in range(n_atom):
        coords = []
        for dim in range(3):
            int_token = yield True, [pos], int_token_range
            frac_token = yield True, [pos+1], frac_token_range
            pos += 2
            coord_str = ''.join(voc_encoder.decode(int_token+frac_token))
            try:
                coord = float(coord_str)
            except Exception:
                return None, pos, 'COORD_NOT_FLOAT'
            if center is not None:
                coord += center[dim]
            coords.append(coord)
        if atom_order:
            pos += 1
        if new_coord_path is not None:
            with open(new_coord_path, 'a') as f:
                f.write(f"{i_atom},{coords[0]},{coords[1]},{coords[2]}\n")
        coordss.append(coords)
    return np.array(coordss), pos, None

def array_to_conf(coord: np.ndarray) -> Conformer:
    conf = Conformer()
    for i in range(len(coord)):
        conf.SetAtomPosition(i, Point3D(*coord[i].tolist()))
    return conf

class GeneratorStreamer(Streamer):
    def __init__(self):
        self.put_gen = self.put_generator()
        next(self.put_gen)
    def put_generator(self) -> Generator[tuple[bool, list[int], list[int]], list[int], None]:
        raise NotImplementedError
    def put(self, tokens: list[int]):
        return self.put_gen.send(tokens)

class TokenWriteStreamer(WrapperStreamer):
    def __init__(self, streamer: Streamer, prompt_token_path: str|None, new_token_path: str|None, voc_encoder: VocEncoder):
        super().__init__(streamer)
        self.voc_encoder = voc_encoder
        self.prompt_token_path = prompt_token_path
        self.new_token_path = new_token_path
        self.is_prompt = True
        self.new_token_dir_made = False
    def put(self, tokens: list[int]):
        if self.is_prompt:
            if self.prompt_token_path is not None:
                make_pardir(self.prompt_token_path)
                with open(self.prompt_token_path, 'a') as f:
                    f.write(' '.join(self.voc_encoder.decode(tokens))+'\n')
            self.is_prompt = False
        else:
            if self.new_token_path is not None:
                if not self.new_token_dir_made:
                    make_pardir(self.new_token_path)
                    self.new_token_dir_made = True
                with open(self.new_token_path, 'a') as f:
                    f.write(self.voc_encoder.i2voc[tokens[0]]+' ')
        return self.streamer.put(tokens)

class TokenSaveStreamer(WrapperStreamer):
    def __init__(self, streamer: Streamer):
        super().__init__(streamer)
        self.prompt_tokens = None
        self.new_tokens = []
    def put(self, tokens: list[int]):
        if self.prompt_tokens is None:
            self.prompt_tokens = tokens
        else:
            self.new_tokens += tokens
        return self.streamer.put(tokens)

class PositionSaveStreamer(WrapperStreamer):
    def __init__(self, streamer: Streamer):
        super().__init__(streamer)
        self.new_positions = []
    def put(self, tokens: list[int]):
        is_remain, positions, token_range = self.streamer.put(tokens)
        self.new_positions += positions
        return is_remain, positions, token_range

class TimeLogStreamer(WrapperStreamer):
    logger = getLogger(f"{__qualname__}")
    def __init__(self, streamer: Streamer, name: str, wait_time: float=1.0):
        super().__init__(streamer)
        self.name = name
        self.is_prompt = True
        self.mean_dt = None
        self.n = 0
        self.wait_time = wait_time
    def put(self, tokens: list[int]):
        if self.is_prompt:
            self.start = self.init = time()
            self.is_prompt = False
        else:
            dt = (end:=time()) - self.start
            self.mean_dt = dt if self.mean_dt is None else self.mean_dt*0.9+dt*0.1
            self.start = end
            self.n += 1
            if should_show(self.n) and (t:=self.start-self.init) >= self.wait_time:
                est_n = self.estimated_n_token()
                if est_n is None:
                    self.logger.info(f"[{self.name}]generated {self.n}/? token in {t:.02f}s")
                else:
                    est_t = self.mean_dt*(est_n-self.n)
                    self.logger.info(f"[{self.name}]generated {self.n}/{est_n} token in {t:02f}s (estimated to end in {est_t:.02f}s)")
        return self.streamer.put(tokens)

class LigandStreamer(GeneratorStreamer):
    def __init__(self, new_sdf_path: str|None, coord_range: float, voc_encoder: VocEncoder, no_token_range: bool, h_atom: bool, h_coord: bool, center: np.ndarray|None=None):
        self.voc_encoder = voc_encoder
        self.coord_range = coord_range
        smi_tokenizer = SmilesTokenizer()
        smi_vocs = list(smi_tokenizer.vocs())+['[XYZ]']
        self.smi_token_range = sorted(self.voc_encoder.encode(smi_vocs))
        if no_token_range:
            self.smi_token_range = list(range(self.voc_encoder.voc_size))
        self.no_token_range = no_token_range
        self.new_sdf_path = new_sdf_path
        self.center = center
        self.mol = None
        self.error = 'PARSE_NOT_ENDED'
        super().__init__()

    def put_generator(self):
        prompt_tokens = yield
        prompt_tokens = self.voc_encoder.decode(prompt_tokens)
        assert prompt_tokens[-1] == '[LIGAND]'
        pos_iter = itr.count(len(prompt_tokens))
        # smiles
        smi_tokens = []
        while True:
            tokens = yield True, [next(pos_iter)], self.smi_token_range
            assert len(tokens) == 1
            token = tokens[0]
            if token == self.voc_encoder.voc2i['[XYZ]']: break
            smi_tokens.append(token)
        smi = ''.join(self.voc_encoder.decode(smi_tokens))
        param = Chem.SmilesParserParams()
        param.removeHs = False
        self.mol = Chem.MolFromSmiles(smi, param)
        # conformer
        if self.mol is not None:
            smi_out = Chem.MolToSmiles(self.mol, canonical=False)
            if smi_out != smi:
                self.error = 'SMILES_MISMATCH'
            else:
                n_atom = self.mol.GetNumAtoms()
                coord, pos, self.error = yield from coord_streamer(n_atom, next(pos_iter), None, self.voc_encoder, self.coord_range, self.no_token_range, False, self.center)
                if coord is not None:
                    self.mol.AddConformer(array_to_conf(coord))
                    if self.new_sdf_path is not None:
                        make_pardir(self.new_sdf_path)
                        with Chem.SDWriter(self.new_sdf_path) as w:
                            w.write(self.mol)
        else:
            self.error = 'SMILES'
        yield False, [next(pos_iter)], [self.voc_encoder.voc2i['[END]']]

class AtomLigandStreamer(GeneratorStreamer):
    def __init__(self, new_sdf_path: str|None, coord_range: float, voc_encoder: VocEncoder, no_token_range: bool, atom_order: bool, h_atom: bool, h_coord: bool):
        super().__init__()
        self.voc_encoder = voc_encoder
        self.new_sdf_path = new_sdf_path
        self.coord_range = coord_range
        self.no_token_range = no_token_range
        self.atom_token_range = sorted(self.voc_encoder.encode(element_symbols()+['[XYZ]']))
        self.atom_order = atom_order
        if self.no_token_range:
            self.all_token_range = list(range(voc_encoder.voc_size))
        self.n_generated_atom = None
        self.mol = None
    def put_generator(self):
        prompt_tokens = yield
        prompt_tokens = self.voc_encoder.decode(prompt_tokens)
        n_prompt_token = len(prompt_tokens)
        assert prompt_tokens[-1] == '[LIGAND]'

        atoms = []
        while True:
            n_atom = len(atoms)
            pos = n_prompt_token+n_atom*7 if self.atom_order else n_prompt_token+n_atom
            tokens = yield True, [pos], (self.all_token_range if self.no_token_range else self.atom_token_range)
            token = tokens[0]
            if token == self.voc_encoder.voc2i['[XYZ]']: break
            if token not in self.atom_token_range:
                yield False, [n_prompt_token+n_atom+1], [self.voc_encoder.voc2i['[END]']]
                return
            atoms.append(token)
        self.n_generated_atom = len(atoms)

        start_point = n_prompt_token+1 if self.atom_order else n_prompt_token+n_atom+1
        coords, pos, error = yield from coord_streamer(self.n_generated_atom, start_point, None, self.voc_encoder, self.coord_range, self.no_token_range, self.atom_order)
        if coords is not None:
            self.mol = Chem.RWMol()
            try:
                for symbol in atoms:
                    self.mol.AddAtom(Chem.Atom(symbol))
                self.mol.AddConformer(array_to_conf(coords))
                DetermineBondOrders(self.mol)
                if self.new_sdf_path is not None:
                    make_pardir(self.new_sdf_path)
                    with Chem.SDWriter(self.new_sdf_path) as w:
                        w.write(self.mol)
            except Exception as e:
                self.logger.warning(f"Error while making atom: {e.args[0]}")
                self.mol = None
        yield False, [pos], [self.voc_encoder.voc2i['[END]']]

class EvaluateStreamer(WrapperStreamer):
    logger = getLogger(f"{__module__}.{__qualname__}")
    def __init__(self, streamer: LigandStreamer|AtomLigandStreamer, e: cf.ProcessPoolExecutor, rec: OBMol, rec_pdbqt_path: str, vina_error_path: str|None, qvina_out_dir: str, qvina_cpu: int):
        """
        
        
        """
        super().__init__(streamer)
        self.rec = rec
        self.rec_pdbqt_path = rec_pdbqt_path
        self.vina_error_path = vina_error_path
        self.qvina_out_dir = qvina_out_dir
        self.vina_future = self.qvina_future = None
        self.vina = self.min_vina = self.qvina = None
        self.e = e
        self.qvina_cpu = qvina_cpu
    def put(self, tokens: list[int]):
        is_remain, positions, token_range = self.streamer.put(tokens)
        if not is_remain:
            self.logger.debug("Evaluating vina...")
            sdf_path = self.streamer.new_sdf_path
            if os.path.exists(sdf_path):
                with open(sdf_path) as f:
                    lig_sdf = f.read()
                rec_pdb = obmol2pdb(self.rec)
                os.makedirs(self.qvina_out_dir, exist_ok=True)
                with open(f"{self.qvina_out_dir}/rec.pdb", 'w') as f:
                    f.write(rec_pdb)
                self.vina_future = self.e.submit(eval_vina, 
                    ligand=lig_sdf, 
                    rec=rec_pdb, 
                    rec_pdbqt_path=self.rec_pdbqt_path
                )
                self.qvina_future = self.e.submit(eval_qvina, 
                    ligand=lig_sdf, 
                    rec_pdb_path=f"{self.qvina_out_dir}/rec.pdb", 
                    out_dir=self.qvina_out_dir,
                    cpu=self.qvina_cpu
                )
        return is_remain, positions, token_range
    def result(self):
        if self.vina_future is not None:
            self.vina, self.min_vina, e = self.vina_future.result()
            if e is not None:
                
        if self.qvina_future is not None:
            self.qvina, e, stdout, stderr = self.qvina_future.result()
            if isinstance(e, Exception):
                with open(f"{self.qvina_out_dir}/stdout.txt", 'w') as f:
                    f.write(stdout)
                with open(f"{self.qvina_out_dir}/stderr.txt", 'w') as f:
                    f.write(stderr)

