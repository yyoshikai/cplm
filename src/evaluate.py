import sys, os
from logging import getLogger
import itertools, subprocess, random
import numpy as np
from rdkit import Chem
from rdkit.Chem import Conformer
from rdkit.Geometry import Point3D
from vina import Vina
from openbabel import pybel
logger = getLogger(__name__)

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
    tokens = itertools.dropwhile(lambda x: x != '[LIGAND]', tokens)
    try:
        tokens.__next__()
    except StopIteration:
        return 'LONG_PREFIX', '', None
    smiles = ''.join(itertools.takewhile(lambda x: x != '[XYZ]', tokens))
    tokens = itertools.takewhile(lambda x: x != '[END]', tokens)
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
