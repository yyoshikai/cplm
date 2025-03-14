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

def eval_vina0(lig_path: str, rec_path: str, out_dir: str) -> tuple[float, float]:
    """
    Parameters
    ----------
    lig_path: ~.sdf
    rec_path: ~.pdb
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Prepare ligand
    subprocess.run(f"obabel -i sdf {lig_path} -o pdbqt -O {out_dir}/lig_h.pdbqt -h -xc --errorlevel 1", shell=True, capture_output=True)
    subprocess.run(f"obabel -i sdf {lig_path} -o sdf -O {out_dir}/lig_h.sdf -h --errorlevel 1", shell=True, capture_output=True)

    # Prepare receptor
    subprocess.run(f"obabel -i pdb {rec_path} -o pdbqt -O {out_dir}/rec.pdbqt -h -xr -xc", shell=True, capture_output=True)

    try:
        v = Vina(sf_name='vina', verbosity=0)
        v.set_receptor(f"{out_dir}/rec.pdbqt")
        v.set_ligand_from_file(f"{out_dir}/lig_h.pdbqt")

        mol = Chem.SDMolSupplier(f"{out_dir}/lig_h.sdf", removeHs=False).__next__()
        center = mol.GetConformer().GetPositions().mean(axis=0)
        v.compute_vina_maps(center=center.tolist(), box_size=[20, 20, 20])
        score = v.score()[0]
        min_score = v.optimize()[0]
    except Exception as e:
        logger.warning(f"Error in Vina {lig_path=}, {rec_path=}: {e}")
        score = min_score = None
    return score, min_score

def eval_vina_dummy(lig_path: str, rec_path: str, out_dir: str) -> tuple[float, float]:
    if random.random() < 0.1:
        return None, None
    else:
        return random.random(), random.random()

logger = getLogger('eval_vina')
def eval_vina(lig_path: str, rec_path: str, out_dir: str) -> tuple[float, float]:
    try:
        logger.warning('started func eval_vina')
        os.makedirs(out_dir, exist_ok=True)
        
        # Prepare ligand
        logger.warning('started prepare ligand')
        mol = next(pybel.readfile('sdf', lig_path))
        logger.warning('end readfile')
        mol.addh()
        logger.warning('end addh')
        mol.write('pdbqt', f"{out_dir}/lig_h.pdbqt", overwrite=True, opt={'c': None})
        logger.warning('end pdbqt')
        mol.write('sdf', f"{out_dir}/lig_h.sdf", overwrite=True)
        logger.warning('end sdf')

        # Prepare receptor
        logger.warning('started prepare receptor')
        rec = next(pybel.readfile('pdb', rec_path))
        logger.warning('end readifle')
        rec.addh()
        logger.warning('end addh')
        rec.write('pdbqt', f"{out_dir}/rec.pdbqt", overwrite=True, opt={'r': None, 'c': None})
        logger.warning('end pdbqt')

        logger.warning('start vina') 
        v = Vina(sf_name='vina', verbosity=0)
        logger.warning('end Vina')
        v.set_receptor(f"{out_dir}/rec.pdbqt")
        logger.warning('end set_receptor')
        v.set_ligand_from_file(f"{out_dir}/lig_h.pdbqt")
        logger.warning('end set_ligand')

        mol = Chem.SDMolSupplier(f"{out_dir}/lig_h.sdf", removeHs=False).__next__()
        logger.warning('end SDMolSupplier')
        center = mol.GetConformer().GetPositions().mean(axis=0)
        logger.warning('end center')
        v.compute_vina_maps(center=center.tolist(), box_size=[20, 20, 20])
        logger.warning('end compute_vina_maps')
        score = v.score()[0]
        logger.warning('end score')
        min_score = v.optimize()[0]
        logger.warning('end min_score')
    except Exception as e:
        logger.warning(f"Error in Vina {lig_path=}, {rec_path=}: {e}")
        score = min_score = None
    logger.warning('end func eval_vina')
    return score, min_score
