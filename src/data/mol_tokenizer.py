import math
from copy import copy, deepcopy
from collections.abc import Generator
from typing import Literal, Any
from logging import getLogger
import numpy as np
from torch.utils.data import Dataset
from rdkit import Chem
from openbabel import openbabel as ob

from ..utils.path import WORKDIR
from ..chem import get_atoms, get_coords, ELEMENT_SYMBOLS, set_coords, atoms_coords_to_mol
from .data import WrapTupleDataset
from .tokenizer import StringTokenizer2, FloatTokenizer, VocEncoder

logger = getLogger(__name__)

def get_smi_orders(mol: Chem.Mol|ob.OBMol, random: bool) -> tuple[str, list[int]]:
    if isinstance(mol, ob.OBMol):
        obc = ob.OBConversion()
        obc.SetOutFormat('smi')
        obc.AddOption('h', obc.OUTOPTIONS) # Output explicit hydrogens as such
        obc.AddOption('n', obc.OUTOPTIONS) # no molecule name (これをつけないと ...sdf みたいなのが付加される)
        if random:
            obc.AddOption('C', obc.OUTOPTIONS) # randomize
        obc.AddOption('O', obc.OUTOPTIONS) # save output atom order
        smi = obc.WriteString(mol).strip()
        orders = ob.toPairData(mol.GetData('SMILES Atom Order')).GetValue()
        orders = [int(o)-1 for o in orders.split(' ')]
    elif isinstance(mol, Chem.Mol):
        if random:
            smi = Chem.MolToSmiles(mol, doRandom=True)
        else:
            smi = Chem.MolToSmiles(mol, canonical=True)
        orders = eval(mol.GetProp('_smilesAtomOutputOrder'))
    else:
        raise ValueError(f"Unknown {type(mol)=}")
    return smi, orders

def get_orders(mol: Chem.Mol|ob.OBMol, order: Literal['residue', 'ran', 'can']) -> list[int]:
    if order == 'residue':   
        if isinstance(mol, ob.OBMol):
            residue_idxs = np.array([atom.GetResidue().GetIdx() for atom in ob.OBMolAtomIter(mol)])
            orders = np.argsort(residue_idxs, kind='stable').tolist()
        else:
            chain_id_is =[]
            chain_id2i = {} # 出現した順に並べる
            serial_numbers = []
            for atom in mol.GetAtoms():
                rinfo = atom.GetPDBResidueInfo()
                if rinfo is None:
                    rinfo = atom.GetNeighbors()[0].GetPDBResidueInfo()
                chain_id = rinfo.GetChainId()
                serial_numbers.append(rinfo.GetSerialNumber())
                if chain_id not in chain_id2i:
                    chain_id2i[chain_id] = len(chain_id2i)
                chain_id_is.append(chain_id2i[chain_id])
            orders = np.lexsort([serial_numbers, chain_id_is], ).tolist() # これがopenbabelと同じ順になる
        return orders
    else:
        return get_smi_orders(mol, random=order == 'ran')[1]

MAX_VALENCE = 10
def get_remain_valences(mol: Chem.Mol|ob.OBMol, orders: list[int]) -> list[int]:
    """
    orders順ではなく, molのatom順に返すようにした。
    """
    if isinstance(mol, ob.OBMol):
        remain_valences = [None] * len(orders)
        added_idxs = set()
        for o in orders:
            atom = mol.GetAtom(o+1)
            remain_valence = 0
            for natom in ob.OBAtomAtomIter(atom):
                if natom.GetId() not in added_idxs:
                    remain_valence += 1
            remain_valences[o] = min(remain_valence, MAX_VALENCE)
            added_idxs.add(o)
    else:
        remain_valences = [None] * len(orders)
        added_idxs = set()
        for o in orders:
            atom = mol.GetAtomWithIdx(o)
            remain_valence = 0
            for natom in atom.GetNeighbors():
                if natom.GetIdx() not in added_idxs:
                    remain_valence += 1
            remain_valences[o] = min(remain_valence, MAX_VALENCE)
            added_idxs.add(o)
    return remain_valences

def smi_to_mol_orders_inv(smi: str, cls: Literal['rdkit', 'ob']) -> tuple[Chem.Mol|ob.OBMol, np.ndarray]:
    if cls == 'rdkit':
        param = Chem.SmilesParserParams()
        param.removeHs = False
        mol = Chem.MolFromSmiles(smi, param)
        if mol is None:
            raise ValueError('SMILES is invalid.')
        smi_out = Chem.MolToSmiles(mol, canonical=False)
        if smi_out != smi: 
            raise ValueError('SMILES mismatch.')
        orders = np.array(eval(mol.GetProp('_smilesAtomOutputOrder')))
    else:
        # Validate SMILES with RDKit to prevent OpenBabel segmentation fault on invalid strings
        param = Chem.SmilesParserParams()
        param.removeHs = False
        temp_mol = Chem.MolFromSmiles(smi, param)
        if temp_mol is None:
            raise ValueError(f'SMILES is invalid: {smi}')
        mol = ob.OBMol()
        obc = ob.OBConversion()
        obc.SetInFormat('smi')
        r = obc.ReadString(mol, smi)
        if not r:
            raise ValueError(f'SMILES is invalid: {smi}')
        orders = np.arange(mol.NumAtoms())
    orders_inv = np.argsort(orders)
    return mol, orders_inv


def coord_stream(n_atom: int) \
        -> Generator[set[str], str, np.ndarray]:
    coord_tokenizer = FloatTokenizer('coord_stream', -250, 250)
    int_token_range = coord_tokenizer.int_vocs()
    frac_token_range = coord_tokenizer.frac_vocs()

    coordss = []
    for i_atom in range(n_atom):
        coords = []
        for dim in range(3):
            int_token = yield int_token_range
            frac_token = yield frac_token_range
            coord = float(int_token+frac_token)
            coords.append(coord)
        coordss.append(coords)
    return np.array(coordss)

def fix_token_range_stream(stream: Generator[tuple[set[str], int], str, Any], token_range: set[str]|list[int]) \
        -> Generator[tuple[set[str]|list[int], int], str, Any]:
    _token_range, pos  = next(stream)
    while True:
        token = yield token_range, pos
        try:
            _token_range, pos = stream.send(token)
        except StopIteration as e:
            return e.value


def pos_offset_stream(stream: Generator[tuple[set[str], int], str, Any], pos_offset: int
) \
        -> Generator[tuple[set[str]|list[int], int], str, Any]:
    token_range, pos  = next(stream)
    while True:
        token = yield token_range, pos+pos_offset
        try:
            token_range, pos = stream.send(token)
        except StopIteration as e:
            return e.value


def encode_token_stream(stream: Generator[tuple[set[str], int], str, Any], voc_encoder: VocEncoder,    
) \
        -> Generator[tuple[list[int], int], int, Any]:
    token_range, pos = next(stream)
    while True:
        token_range = sorted(voc_encoder.encode(token_range))
        token = yield token_range, pos
        token = voc_encoder.i2voc[token]
        try:
            token_range, pos = stream.send(token)
        except StopIteration as e:
            return e.value


def wrap_decode_stream(add_end_token: bool, add_range_pos: bool):
    def wrapper(stream_func):
        def wrapped_decode_stream(self, end_token: str, cls: Literal['rdkit', 'ob']):
            if add_end_token:
                gen = stream_func(self, cls)
            else:
                gen = stream_func(self, end_token, cls)

            out = next(gen)
            pos = 0
            poss = set()
            while True:
                if add_range_pos:
                    out = (out, pos)
                    pos += 1
                poss.add(out[1])
                token = yield out
                try:
                    out = gen.send(token)
                except StopIteration as e:
                    if add_end_token:
                        pos = min(set(range(len(poss)+1)) - poss)
                        yield {end_token}, pos
                    return e.value
        return wrapped_decode_stream
    return wrapper
 

class MolTokenizer:
    def encode(self, mol: Chem.Mol|ob.OBMol) -> tuple[list[str], list[int], list[int]]:
        """
        Returns
        -------
        tokens: list[str]
            - [START], [END] 等は含まない。 [XYZ] は含む。
            - encode -> decode とすると, 原子の順番が変わるので, 順番も同時に出力
        positions: list[int]
            no offset. First token is 0.
        orders: list[int]
            orders[i] 番目の原子が, tokens中でi番目に出てくる。
        """
        raise NotImplementedError
    
    def decode_stream(self, end_token: str, cls: Literal['rdkit', 'ob']) -> Generator[tuple[list[str], int], str, Chem.Mol|ob.OBMol]:
        """
        coord_streamer に近いものを目指す。
        is_remain はなくし, 終了したら単にreturnする。
        エラーは普通にraiseする。wrapするstreamerで補足してください。
        
        Yields
        ------
        token_range: set[str]
            choice of possible tokens
        pos: position of the previously received token (supposed to be next input). First output is -1 (corresponds to output of encode())
        
        Receives
        --------
        token: str
            Generated token.

        Returns
        -------
        mol: Chem.Mol|ob.OBMol
            generated molecule.
        atom_poss: list[int]
            molのi番目の原子が, atom_poss[i] 番目のトークンで表されている。
        coord_posss: list[list[int]]
            同じく, 座標について

        """
        raise NotImplementedError
    
    def vocs(self) -> set[str]:
        raise NotImplementedError


class AtomsCoordsTokenizer(MolTokenizer):
    def __init__(self, order: Literal['residue', 'ran', 'can']):
        self.float_tokenizer = FloatTokenizer(type(self).__name__, -250, 250)
        self.order = order

    def encode(self, mol):
        atoms = get_atoms(mol)
        coords = get_coords(mol)
        orders = get_orders(mol, self.order)
        tokens = [atoms[o] for o in orders]+['[XYZ]']+self.float_tokenizer.tokenize_array(coords[orders].ravel())
        positions= list(range(len(tokens)))
        return tokens, positions, orders
    
    @wrap_decode_stream(add_end_token=True, add_range_pos=True)
    def decode_stream(self, cls) -> Generator[list[str], str, Chem.Mol|ob.OBMol]:

        atom_token_range = set(ELEMENT_SYMBOLS) | {'CA', '[XYZ]'}
        atoms = []
        while True:
            atom_token = yield atom_token_range
            if atom_token == '[XYZ]':
                break
            atoms.append(atom_token)
        n_atom = len(atoms)
        coords = yield from coord_stream(n_atom)
        mol = atoms_coords_to_mol(atoms, coords, cls)
        atom_poss = np.arange(n_atom, dtype=int).tolist()
        coord_posss = (np.arange(n_atom*6)+n_atom+1).reshape(-1, 6).tolist()
        return mol, atom_poss, coord_posss

    def vocs(self):
        return set(ELEMENT_SYMBOLS) | self.float_tokenizer.vocs() | {'[XYZ]', 'CA'}


class AtomCoordsTokenizer(MolTokenizer):
    def __init__(self, order: Literal['residue', 'ran', 'can']):
        self.float_tokenizer = FloatTokenizer(type(self).__name__, -250, 250)
        self.order = order

    def encode(self, mol):
        atoms = get_atoms(mol)
        coords = get_coords(mol)
        orders = get_orders(mol, self.order)
        tokens = []
        for o in orders:
            tokens += [atoms[o]] + self.float_tokenizer.tokenize_array(coords[o])
        positions = list(range(len(tokens)))
        return tokens, positions, orders
    
    @wrap_decode_stream(add_end_token=False, add_range_pos=True)
    def decode_stream(self, end_token: str, cls):
        atom_token_range = set(ELEMENT_SYMBOLS) | {'CA', end_token}
        atoms = []
        coords = []
        while True:
            atom_token = yield atom_token_range
            if atom_token == end_token:
                break
            atom = 'C' if atom_token == 'CA' else atom_token
            atoms.append(atom)
            coord = yield from coord_stream(1)
            coords.append(coord)
        n_atom = len(atoms)
        coords = np.concatenate(coords, axis=0)
        mol = atoms_coords_to_mol(atoms, coords, cls)
        poss = np.arange(n_atom*7).reshape(-1, 7)
        atom_poss = poss[:,0].tolist()
        coord_posss = poss[:,1:].tolist()
        return mol, atom_poss, coord_posss

    def vocs(self):
        return set(ELEMENT_SYMBOLS) | self.float_tokenizer.vocs() | {'CA'}


class OrderedAtomsCoordsTokenizer(MolTokenizer):
    def __init__(self, order: Literal['residue', 'ran', 'can']):
        self.order = order
        self.float_tokenizer = FloatTokenizer(type(self).__name__, -250, 250)

    def encode(self, mol):
        atoms = get_atoms(mol)
        orders = get_orders(mol, self.order)
        coords = get_coords(mol)
        tokens = [atoms[o] for o in orders] + ['[XYZ]'] + self.float_tokenizer.tokenize_array(coords[orders].ravel())
        n_atom = len(atoms)
        positions = np.arange(n_atom*7, dtype=int).reshape(n_atom, 7)
        positions = positions[:,0].tolist()+[n_atom*7]+positions[:,1:].ravel().tolist()
        return tokens, positions, orders
    
    @wrap_decode_stream(add_end_token=True, add_range_pos=False)
    def decode_stream(self, cls):
        atom_token_range = set(ELEMENT_SYMBOLS)|{'CA', '[XYZ]'}
        pos = 0
        atoms = []
        while True:
            atom_token = yield atom_token_range, pos
            pos += 7
            if atom_token == '[XYZ]':
                break
            atoms.append(atom_token)
        
        int_token_range = self.float_tokenizer.int_vocs()
        frac_token_range = self.float_tokenizer.frac_vocs()
        coords = []
        pos = 1
        for i_atom in range(len(atoms)):
            coord = []
            for dim in range(3):
                int_token = yield int_token_range, pos
                pos += 1
                frac_token = yield frac_token_range, pos
                pos += 1
                coord.append(float(int_token+frac_token))
            coords.append(coord)
            pos += 1
        coords = np.array(coords)
        mol = atoms_coords_to_mol(atoms, coords, cls)
        n_atom = len(atoms)
        atom_poss = np.arange(n_atom, dtype=int).tolist()
        coord_posss = (np.arange(n_atom*6)+n_atom+1).reshape(-1, 6).tolist()
        return mol, atom_poss, coord_posss

    def vocs(self):
        return set(ELEMENT_SYMBOLS) | self.float_tokenizer.vocs() | {'[XYZ]', 'CA'}


class AtomValenceCoordsTokenizer(MolTokenizer):
    def __init__(self, order: Literal['residue', 'ran', 'can']):
        self.float_tokenizer = FloatTokenizer(type(self).__name__, -250, 250)
        self.order = order
    
    def encode(self, mol):
        atoms = get_atoms(mol)
        coords = get_coords(mol)
        orders = get_orders(mol, self.order)
        valences = get_remain_valences(mol, orders)
        tokens = []
        for o in orders:
            tokens += [atoms[o], str(valences[o])]+self.float_tokenizer.tokenize_array(coords[o])
        positions = list(range(len(tokens)))
        return tokens, positions, orders
    
    @wrap_decode_stream(add_end_token=False, add_range_pos=True)
    def decode_stream(self, end_token: str, cls):
        atom_token_range = set(ELEMENT_SYMBOLS)|{'CA', end_token}
        valence_token_range = {str(i) for i in range(MAX_VALENCE+1)}
        atoms = []
        coords = []
        while True:
            atom_token = yield atom_token_range
            if atom_token == end_token:
                break
            atom = 'C' if atom_token == 'CA' else atom_token
            atoms.append(atom)
            _valence_token = yield valence_token_range
            coord = yield from coord_stream(1)
            coords.append(coord)
        coords = np.concatenate(coords, axis=0)
        mol =  atoms_coords_to_mol(atoms, coords, cls)
        n_atom = len(atoms)
        poss = np.arange(n_atom*8).reshape(-1, 8)
        atom_poss = poss[:,0].tolist()
        coord_poss = poss[:,2:].tolist()
        return mol, atom_poss, coord_poss

    def vocs(self):
        return set(ELEMENT_SYMBOLS) | self.float_tokenizer.vocs() | {str(i) for i in range(MAX_VALENCE+1)} | {'CA'}


class SmilesCoordsTokenizer(MolTokenizer):
    def __init__(self, random: bool, smiles_voc_dir: str, h: bool):
        self.random = random
        self.smi_tokenizer = StringTokenizer2(f"{WORKDIR}/cplm/src/data/vocs/{smiles_voc_dir}")
        with open(f"{WORKDIR}/cplm/src/data/vocs/{smiles_voc_dir}/non_atom_tokens.txt") as f:
            self.non_atom_tokens = set(f.read().splitlines())
        if not h:
            self.non_atom_tokens |= {'H', '[H]'}
        self.float_tokenizer = FloatTokenizer(type(self).__name__, -250, 250)
    
    def encode(self, mol):
        smi, orders = get_smi_orders(mol, self.random)
        coords = get_coords(mol)
        tokens = self.smi_tokenizer.tokenize(smi)+['[XYZ]']+self.float_tokenizer.tokenize_array(coords[orders].ravel())
        positions = list(range(len(tokens)))
        return tokens, positions, orders
    
    @wrap_decode_stream(add_end_token=True, add_range_pos=True)
    def decode_stream(self, cls):
        smi_token_range = set(self.smi_tokenizer.vocs())|{'[XYZ]'}
        smi_tokens = []
        while True:
            token = yield smi_token_range
            if token == '[XYZ]':
                break
            smi_tokens.append(token)
        smi = ''.join(smi_tokens)
        raw_atom_poss = np.array([i for i, token in enumerate(smi_tokens) if token not in self.non_atom_tokens])

        mol, orders_inv = smi_to_mol_orders_inv(smi, cls)
        n_atom = len(orders_inv)
        
        coords = yield from coord_stream(n_atom)
        raw_coord_posss = (np.arange(n_atom*6)+len(smi_tokens)+1).reshape(-1, 6)
        atom_poss = raw_atom_poss[orders_inv].tolist()
        coord_posss = raw_coord_posss[orders_inv].tolist()
        set_coords(mol, coords[orders_inv])
        return mol, atom_poss, coord_posss

    def vocs(self):
        return self.smi_tokenizer.vocs() | self.float_tokenizer.vocs() | {'[XYZ]'}


class SmileCoordsTokenizer(MolTokenizer):
    def __init__(self, random: bool, smiles_voc_dir: str, h: bool):
        self.random = random
        self.h = h
        self.smi_tokenizer = StringTokenizer2(f"{WORKDIR}/cplm/src/data/vocs/{smiles_voc_dir}")
        with open(f"{WORKDIR}/cplm/src/data/vocs/{smiles_voc_dir}/non_atom_tokens.txt") as f:
            self.non_atom_tokens = set(f.read().splitlines())
        if not self.h:
            self.non_atom_tokens |= {'[H]', 'H'}
        self.float_tokenizer = FloatTokenizer(type(self).__name__, -250, 250)
    
    def encode(self, mol):
        smi, orders = get_smi_orders(mol, self.random)

        coords = get_coords(mol)
        smi_tokens = self.smi_tokenizer.tokenize(smi)            
        tokens = []
        i = 0
        for smi_token in smi_tokens:
            tokens.append(smi_token)
            if smi_token not in self.non_atom_tokens:
                tokens += self.float_tokenizer.tokenize_array(coords[orders[i]])
                i += 1
        positions = list(range(len(tokens)))
        return tokens, positions, orders
    
    @wrap_decode_stream(add_end_token=False, add_range_pos=True)
    def decode_stream(self, end_token, cls):
        smi_token_range = self.smi_tokenizer.vocs()|{end_token}

        # smiles
        smi_tokens = []
        raw_atom_poss = []
        raw_coord_posss = []
        coords = []
        pos = 0
        while True:
            token = yield smi_token_range
            pos += 1
            if token == end_token:
                break
            smi_tokens.append(token)
            if token not in self.non_atom_tokens:
                raw_atom_poss.append(pos-1)
                raw_coord_posss.append(np.arange(pos, pos+6))
                coord = yield from coord_stream(1)
                coords.append(coord)
                pos += 6
        coords = np.concatenate(coords, axis=0)
        smi = ''.join(smi_tokens)

        mol, orders_inv = smi_to_mol_orders_inv(smi, cls)
        atom_poss = np.array(raw_atom_poss)[orders_inv].tolist()
        coord_posss = np.stack(raw_coord_posss)[orders_inv].tolist()
        set_coords(mol, coords[orders_inv])
        return mol, atom_poss, coord_posss

    def vocs(self):
        return self.smi_tokenizer.vocs() | self.float_tokenizer.vocs()


def get_mol_tokenizer(
    format: Literal['atoms_coords', 'atom_coords', 'atom_valence_coords', 'ordered_atoms_coords', 'smiles_coords', 'smile_coords'], 
    order: Literal['residue', 'ran', 'can'], 
    smiles_voc_dir: str, 
    h: bool,
) -> MolTokenizer:
    assert order in ['residue', 'ran', 'can']
    match format:
        case 'atoms_coords':
            return AtomsCoordsTokenizer(order)
        case 'atom_coords':
            return AtomCoordsTokenizer(order)
        case 'atom_valence_coords':
            return AtomValenceCoordsTokenizer(order)
        case 'ordered_atoms_coords':
            return OrderedAtomsCoordsTokenizer(order)
        case _:
            assert order != 'residue'
            random = order == 'ran'
            if format == 'smiles_coords':
                return SmilesCoordsTokenizer(random, smiles_voc_dir, h)
            elif format == 'smile_coords':
                return SmileCoordsTokenizer(random, smiles_voc_dir, h)
            else:
                raise ValueError

class MolTokenizerDataset(WrapTupleDataset[tuple[list[str], list[int], list[int]]]):
    def __init__(self, mol_data: Dataset[Chem.Mol|ob.OBMol], format, order, smiles_voc_dir, h: bool):
        super().__init__(mol_data, 2)
        self.mol_tokenizer = get_mol_tokenizer(format, order, smiles_voc_dir, h)
    def __getitem__(self, idx: int):
        tokens, positions, orders = self.mol_tokenizer.encode(self.dataset[idx])
        return (tokens, positions), orders

    def vocs(self):
        return self.mol_tokenizer.vocs()