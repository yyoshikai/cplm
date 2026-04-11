import sys, os, traceback
import itertools as itr
import concurrent.futures as cf
from time import time
from collections.abc import Generator
from logging import getLogger
import numpy as np, pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.rdDetermineBonds import DetermineBondOrders, DetermineBonds
from openbabel.openbabel import OBMol
from ..utils import should_show
from ..utils.path import make_pardir, mwrite
from ..model import Streamer, WrapperStreamer
from ..data.molecule import element_symbols
from ..data.protein import AtomRepr
from ..data.tokenizer import FloatTokenizer, SmilesTokenizer, VocEncoder
from ..chem import array_to_conf

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
            int_token = yield True, pos, int_token_range
            frac_token = yield True, pos+1, frac_token_range
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

class GeneratorStreamer(Streamer):
    def __init__(self):
        self.put_gen = self.put_generator()
        next(self.put_gen)
    def put_generator(self) -> Generator[tuple[bool, list[int], list[int]], list[int], None]:
        raise NotImplementedError
    def put(self, tokens: list[int]):
        return self.put_gen.send(tokens)

class TokenWriteStreamer(WrapperStreamer):
    def __init__(self, streamer: Streamer, voc_encoder: VocEncoder, prompt_position: list[int], prompt_csv_path: str, new_csv_path: str):
        super().__init__(streamer)
        self.voc_encoder = voc_encoder
        self.prompt_position = prompt_position
        self.prompt_csv_path = prompt_csv_path
        self.new_csv_path = new_csv_path
        self.is_prompt = True
    def put(self, tokens: list[int]):
        token_ids = tokens
        tokens = self.voc_encoder.decode(tokens)
        if self.is_prompt:
            assert len(tokens) == len(self.prompt_position)
            make_pardir(self.prompt_csv_path)
            pd.DataFrame({'position': self.prompt_position, 'token': tokens}).to_csv(self.prompt_csv_path, index=False)
            self.prompt_position = None
            make_pardir(self.new_csv_path)
            with open(self.new_csv_path, 'w') as f:
                f.write("position,token\n")
        else:
            if len(tokens) == 0: # generation is ended
                return False, 0, [self.voc_encoder.pad_token]
            assert len(tokens) == 1, str(tokens)
            with open(self.new_csv_path, 'a') as f:
                f.write(f"{tokens[0]}\n")
        
        is_remain, position, token_range = self.streamer.put(token_ids)
        with open(self.new_csv_path, 'a') as f:
            f.write(f"{position},")
        self.is_prompt = False
        return is_remain, position, token_range
    
class RangeWriteStreamer(WrapperStreamer):
    def __init__(self, streamer: Streamer, voc_encoder: VocEncoder, range_path: str):
        super().__init__(streamer)
        self.voc_encoder = voc_encoder
        self.range_path = range_path
        self.is_init = True
    def put(self, tokens: list[int]):
        is_remain, position, token_range = self.streamer.put(tokens)
        if self.is_init:
            mwrite(self.range_path, "token_range\n")
            self.is_init = False
        with open(self.range_path, 'a') as f:
            f.write(','.join(self.voc_encoder.i2voc[t] for t in token_range)+'\n')
        return is_remain, position, token_range

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
        is_remain, position, token_range = self.streamer.put(tokens)
        self.new_positions.append(position)
        return is_remain, position, token_range

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

class TqdmStreamer(WrapperStreamer):
    def __init__(self, streamer: Streamer, total: int|None=None, desc: str|None=None):
        super().__init__(streamer)
        self.pbar = tqdm(total=total, desc=desc)
    
    def put(self, tokens: list[int]):
        self.pbar.update()
        return self.streamer.put(tokens)

class LigandStreamer(Streamer):
    def ligand(self) -> Chem.Mol|None:
        raise NotImplementedError
    def error(self) -> str|None:
        raise NotImplementedError

class SmilesLigandStreamer(GeneratorStreamer, LigandStreamer):
    def __init__(self, coord_range: float, voc_encoder: VocEncoder, no_token_range: bool, lig_h: AtomRepr, smiles_voc_file: str, center: np.ndarray|None=None):
        self.voc_encoder = voc_encoder
        self.coord_range = coord_range
        smi_tokenizer = SmilesTokenizer(smiles_voc_file)
        smi_vocs = list(smi_tokenizer.vocs())+['[XYZ]']
        self.smi_token_range = sorted(self.voc_encoder.encode(smi_vocs))
        if no_token_range:
            self.smi_token_range = list(range(self.voc_encoder.voc_size))
        self.no_token_range = no_token_range
        self.center = center
        self.mol = None
        self._error = 'PARSE_NOT_ENDED'
        if lig_h not in ['all', 'none']:
            raise NotImplementedError
        super().__init__()

    def put_generator(self):
        prompt_tokens = yield
        prompt_tokens = self.voc_encoder.decode(prompt_tokens)
        assert prompt_tokens[-1] == '[LIGAND]'
        pos_iter = itr.count(len(prompt_tokens))
        # smiles
        smi_tokens = []
        while True:
            tokens = yield True, next(pos_iter), self.smi_token_range
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
                self._error = 'SMILES_MISMATCH'
            else:
                n_atom = self.mol.GetNumAtoms()
                coord, pos, self._error = yield from coord_streamer(n_atom, next(pos_iter), None, self.voc_encoder, self.coord_range, self.no_token_range, False, self.center)
                if coord is not None:
                    self.mol.AddConformer(array_to_conf(coord))
        else:
            self._error = 'SMILES'
        yield False, next(pos_iter), [self.voc_encoder.voc2i['[END]']]

    def ligand(self):
        return self.mol
    def error(self):
        return self._error

class AtomsCoordsLigandStreamer(GeneratorStreamer, LigandStreamer):
    def __init__(self, coord_range: float, voc_encoder: VocEncoder, no_token_range: bool, atom_order: bool, lig_h: AtomRepr, center: np.ndarray|None = None):
        self.voc_encoder = voc_encoder
        self.coord_range = coord_range
        self.no_token_range = no_token_range
        self.atom_token_range = sorted(self.voc_encoder.encode(element_symbols()+['[XYZ]']))
        if self.no_token_range:
            self.all_token_range = list(range(voc_encoder.voc_size))
        self.atom_order = atom_order
        self.n_generated_atom = None
        self.mol = None
        if lig_h not in ['all', 'none']:
            raise NotImplementedError
        if center is not None:
            raise NotImplementedError
        self._error = 'PARSE_NOT_ENDED'
        super().__init__()
    def put_generator(self):
        prompt_tokens = yield
        prompt_tokens = self.voc_encoder.decode(prompt_tokens)
        n_prompt_token = len(prompt_tokens)
        assert prompt_tokens[-1] == '[LIGAND]'

        atoms = []
        while True:
            n_atom = len(atoms)
            pos = n_prompt_token+n_atom*7 if self.atom_order else n_prompt_token+n_atom
            tokens = yield True, pos, (self.all_token_range if self.no_token_range else self.atom_token_range)
            token = tokens[0]
            if token == self.voc_encoder.voc2i['[XYZ]']: break
            if token not in self.atom_token_range:
                yield False, n_prompt_token+n_atom+1, [self.voc_encoder.voc2i['[END]']]
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
            except Exception as e:
                self.logger.warning(f"Error while making atom: {e.args[0]}")
                self.mol = None
        yield False, pos, [self.voc_encoder.voc2i['[END]']]

    def ligand(self):
        return self.mol
    
class AtomCoordsLigandStreamer(GeneratorStreamer, LigandStreamer):
    logger = getLogger(f"{__module__}.{__qualname__}")

    def __init__(self, coord_range: float, voc_encoder: VocEncoder, no_token_range: bool, lig_h: AtomRepr, center: np.ndarray|None=None):
        self.voc_encoder = voc_encoder
        self.coord_range = coord_range
        self.no_token_range = no_token_range
        self.n_generated_atom = None
        self.mol = None
        self.center = center
        self._error = 'PARSE_NOT_ENDED'
        if lig_h not in ['all', 'none']:
            raise NotImplementedError
        super().__init__()

    def put_generator(self):
        prompt_tokens = yield
        prompt_tokens = self.voc_encoder.decode(prompt_tokens)
        n_prompt_token = len(prompt_tokens)
        assert prompt_tokens[-1] == '[LIGAND]'


        coord_tokenizer = FloatTokenizer('', -self.coord_range, self.coord_range)
        atom_tokens = set(self.voc_encoder.encode(element_symbols()+['[END]']))
        int_tokens = set(self.voc_encoder.encode(coord_tokenizer.int_vocs()))
        frac_tokens = set(self.voc_encoder.encode(coord_tokenizer.frac_vocs()))
        if self.no_token_range:
            atom_token_range = int_token_range = frac_token_range \
                = list(range(self.voc_encoder.voc_size))
        else:
            atom_token_range = sorted(atom_tokens)
            int_token_range = sorted(int_tokens)
            frac_token_range = sorted(frac_tokens)


        atoms = []
        coordss = []
        pos = n_prompt_token
        end_token = self.voc_encoder.voc2i['[END]']
        while True:
            atom_token = yield True, pos, atom_token_range
            assert len(atom_token) == 1
            atom_token = atom_token[0]
            pos += 1

            if atom_token == end_token:
                self.mol = Chem.RWMol()
                for atom in atoms:
                    self.mol.AddAtom(Chem.Atom(atom))
                try:
                    self.mol.AddConformer(array_to_conf(np.array(coordss)))
                except Exception as e:
                    self.logger.info(f"Error at AddConformer: {type(e).__name__}{e.args}")
                    self._error = 'ADD_CONFORMER'
                    self.mol = None
                    break
                try:
                    DetermineBonds(self.mol)
                except Exception as e:
                    self.logger.info(f"Error at DetermineBondOrders: {type(e)}{e.args}")
                    self._error = 'DETERMINE_BOND_ORDERS'
                    self.mol = None
                    break
                self._error = None
                break
            elif atom_token not in atom_tokens:
                self._error = 'ATOM'
                break
            else:
                atoms.append(self.voc_encoder.i2voc[atom_token])
                coords = []
                for dim in range(3):
                    int_token = yield True, pos, int_token_range
                    frac_token = yield True, pos+1, frac_token_range
                    pos += 2
                    coord_str = ''.join(self.voc_encoder.decode(int_token+frac_token))
                    try:
                        coord = float(coord_str)
                    except Exception:
                        self._error = 'COORD_NOT_FLOAT'
                        yield False, pos, [end_token]
                        return
                    if self.center is not None:
                        coord += self.center[dim]
                    coords.append(coord)
                coordss.append(coords)
        yield False, pos, [end_token]
        return 
    def ligand(self):
        return self.mol
    def error(self):
        return self._error

def get_ligand_streamer(
        format: str, 
        coord_range: float, 
        voc_encoder: VocEncoder, 
        no_token_range: bool,
        lig_h: AtomRepr, 
        smiles_voc_file: str,
        center: np.ndarray|None=None
) -> LigandStreamer:
    kwargs = dict(coord_range=coord_range, voc_encoder=voc_encoder, no_token_range=no_token_range, lig_h=lig_h, center=center)
    if format == 'smiles_coords':
        return SmilesLigandStreamer(**kwargs, smiles_voc_file=smiles_voc_file)
    elif format in ['atoms_coords', 'ordered_atoms_coords']:
        atom_order = format == 'ordered_atoms_coords'
        return AtomsCoordsLigandStreamer(**kwargs, atom_order=atom_order)
    elif format == 'atom_coords':
        return AtomCoordsLigandStreamer(**kwargs)
    else:
        raise ValueError(f"Unknown {format=}")
    
class SaveLigandStreamer(WrapperStreamer):
    def __init__(self, streamer: LigandStreamer, new_sdf_path: str):
        super().__init__(streamer)
        self.streamer = streamer
        self.new_sdf_path = new_sdf_path
    
    def put(self, tokens: list[int]):
        output = self.streamer.put(tokens)
        mol = self.streamer.ligand()
        if mol is not None and mol.GetNumConformers() > 0:
            make_pardir(self.new_sdf_path)
            with Chem.SDWriter(self.new_sdf_path) as w:
                w.write(mol)
        return output
