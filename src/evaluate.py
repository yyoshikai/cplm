import sys, os
from logging import getLogger
import itertools as itr, random
from typing import Any, TypeVar
import numpy as np
from rdkit import Chem
from rdkit.Chem import Conformer
from rdkit.Geometry import Point3D
from vina import Vina
from openbabel import pybel
logger = getLogger(__name__)

T = TypeVar('T')

def split_list(l: list[T], sep: T) -> list[list[T]]:
    splits = []
    split = []
    for x in l:
        if x == sep:
            splits.append(split)
            split = []
        else:
            split.append(x)
    splits.append(split)
    return splits

def parse_mol_tokens(tokens: list[str]) -> tuple[str, str, np.ndarray|None]:
    """
    Parameters
    ----------
    tokens: 
        ([POCKET], ..., [XYZ], ...,) [LIGAND], ..., [XYZ], ...(, [END], ...)

    Returns
    -------
    coord_error:
        '': No error
        'LONG_PREFIX': No '[LIGAND]' token
        'NOT_FLOAT': [XYZ]~[END] tokens do not fit `xx .xxx`+
        'SIZE': coord_len % 3 != 0
    smiles: str
    coords: np.ndarray[*, 3]
    """
    tokens = itr.dropwhile(lambda x: x != '[LIGAND]', tokens)
    clauses = split_list(tokens, '[LIGAND]')
    if len(clauses) <= 1:
        return 'LONG_PREFIX', '', None
    tokens = split_list(clauses[1], '[END]')[0].__iter__()

    smiles = ''.join(itr.takewhile(lambda x: x != '[XYZ]', tokens))
    coords = []
    while True:
        try:
            coord = tokens.__next__()+tokens.__next__()
            coord = float(coord)
            coords.append(coord)
        except StopIteration:
            break
        except ValueError:
            return 'NOT_FLOAT', smiles, None
    coords = np.array(coords)
    if len(coords) % 3 != 0:
        return 'SIZE', smiles, None
    coords = coords.reshape(-1, 3)
    return '', smiles, coords

def parse_mol(smiles: str, coords: np.ndarray) -> tuple[str, Chem.Mol|None]:
    """
    Parameters
    ----------
    smiles: str
    coord: np.ndarray[*, 3]

    Returns
    -------
    error: str 
    mol: Chem.Mol
    
    """

    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)

    if mol is None or smiles == '':
        return 'SMILES', None
    if mol.GetNumAtoms() != len(coords): 
        return 'COORD_MISMATCH', None

    smi_out = Chem.MolToSmiles(mol, canonical=False)
    if smi_out != smiles:
        return 'SMILES_MISMATCH', None
    
    atom_idxs = mol.GetProp('_smilesAtomOutputOrder', autoConvert=True)
    natom = mol.GetNumAtoms()
    if not np.all(np.array(atom_idxs) == np.arange(natom)):
        return 'INDEX_MISMATCH', None
    
    conf = Conformer()
    for iatom in range(natom):
        conf.SetAtomPosition(iatom, Point3D(*coords[iatom]))
    mol.AddConformer(conf)
    return '', mol

def parse_pocket_tokens(tokens: list[str], end_token: str, 
        pocket_coord_heavy: bool, coord_follow_atom: bool) \
        -> tuple[str, list[str], np.ndarray|None]:
    """
    Parameters
    ----------
    tokens: list[str]
        ... [POCKET] .... <end_token> ...
    end_token: str
        [LIGAND], [END], etc.

    Returns
    -------
    error: name of error. '' if no error was found.
    atoms: list[str]
    coords: np.ndarray[float]
    
    """
    pocket_tokens = {'CA', 'C', 'H', 'O', 'N', 'S', 'P', 'F', 'ZN', 'BR', 'MG', '[UNK]'}

    token_iter = iter(tokens)
    list(itr.takewhile(lambda x: x != '[POCKET]', token_iter))

    if coord_follow_atom:
        atoms = []
        coords = []
        error = ''

        atom = None
        coord = []
        coord_str = None
        for token in itr.takewhile(lambda x: x!= end_token, token_iter):
            if atom is None:
                atom = token
                continue
            if coord_str is None:
                coord_str = token
                continue
            coord_str += token
            try:
                coord_v = float(coord_str)
                coord_str = None
            except ValueError:
                error = 'coord_format'
                break
            coord.append(coord_v)
            if len(coord) == 3:
                atoms.append(atom)
                coords.append(coord)
                atom = None
                coord = []
        if not set(atoms) <= pocket_tokens:
            error = 'atom'
        elif coord_str is not None:
            error = 'coord_format'
        elif atom is not None:
            error = 'coord_count'
        if error != "":
            return error, atoms, None
        else:
            return error, atoms, np.array(coords)

    else:
        # parse atoms
        atoms = list(itr.takewhile(lambda x: x!='[XYZ]', token_iter))
        if not set(atoms) <= pocket_tokens:
            return 'atom', atoms, None

        # check coords
        if end_token is None:
            coord_tokens = list(token_iter)
        else:
            coord_tokens = list(itr.takewhile(lambda x: x != end_token, token_iter))

        error = ''
        coords = []
        coord_str = ""
        for t in coord_tokens:
            if coord_str == "":
                coord_str = t
            else:
                if t[0] != '.':
                    error = 'coord_format'
                    break
                coord_str += t
                try:
                    coord = float(coord_str)
                except ValueError:
                    error = 'coord_format'
                    break
                coords.append(coord)
                coord_str = ''
        if error != '': 
            return error, atoms, None
        coords = np.array(coords)

        if pocket_coord_heavy:
            n_coord = len(atoms)
        else:
            n_coord = np.sum(np.array(atoms) == 'CA')
        if len(coords) != n_coord*3:
            error = 'coord_count'
            return 'coord_count', atoms, None
        return '', atoms, coords.reshape(-1, 3)

def parse_pocket(atoms: list[str], coords: np.ndarray) -> str:
    """
    Parameters
    ----------
    atoms: list[str](N)
    coords: np.ndarray(N, 3)

    Returns
    -------
    pdb: PDB string    
    """
    assert len(atoms) == len(coords)

    pdb = ''
    for ia in range(len(atoms)):
        c = coords[ia]
        atom = atoms[ia]
        if atom == '[UNK]': continue # OK?
        elem = 'C' if atom == 'CA' else atom
        pdb += f"ATOM    {ia+1:>3}  {atom:<2}  UNK A   1     {c[0]:>7.03f} {c[1]:>7.03f} {c[2]:>7.03f}  1.00 40.00          {elem:2>}  \n"
    return pdb

def eval_vina_dummy(lig_path: str, rec_path: str, out_dir: str) -> tuple[float, float]:
    if random.random() < 0.1:
        return None, None
    else:
        return random.random(), random.random()

def eval_vina(lig_path: str, rec_path: str, out_dir: str) -> tuple[float, float]:
    try:
        os.makedirs(out_dir, exist_ok=True)
        
        # Prepare ligand
        mol = next(pybel.readfile('sdf', lig_path))
        mol.addh()
        mol.write('pdbqt', f"{out_dir}/lig_h.pdbqt", overwrite=True, opt={'c': None})
        mol.write('sdf', f"{out_dir}/lig_h.sdf", overwrite=True)

        # Prepare receptor
        rec = next(pybel.readfile('pdb', rec_path))
        rec.addh()
        rec.write('pdbqt', f"{out_dir}/rec.pdbqt", overwrite=True, opt={'r': None, 'c': None})

        # Vina
        v = Vina(sf_name='vina', verbosity=0)
        v.set_receptor(f"{out_dir}/rec.pdbqt")
        mol = Chem.SDMolSupplier(f"{out_dir}/lig_h.sdf", removeHs=False).__next__()
        if mol is None:
            logger.warning(f"Error: {lig_path=} is None.")
            return None, None
        center = mol.GetConformer().GetPositions().mean(axis=0)
        v.set_ligand_from_file(f"{out_dir}/lig_h.pdbqt")

        v.compute_vina_maps(center=center.tolist(), box_size=[20, 20, 20])
        score = v.score()[0]
        min_score = v.optimize()[0]

    except Exception as e:
        logger.warning(f"Error in {lig_path=}, {rec_path=}: {e}")
        score = min_score = None
    return score, min_score
