import sys, os
from logging import getLogger
import itertools as itr, random, subprocess
from typing import Any, TypeVar, Optional
from time import time
import numpy as np
from rdkit import Chem
from rdkit.Chem import Conformer
from rdkit.Geometry import Point3D
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
from rdkit.Chem.rdMolAlign import CalcRMS
from vina import Vina
from openbabel import pybel
from AutoDockTools.Utilities24 import prepare_receptor4
from .prepare_receptor4 import main as prepare_receptor4_func
from .utils.time import wtqdm
logger = getLogger(__name__)
WORKDIR = os.environ.get('WORKDIR', "/workspace")

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
        'COORD_LONG_PREFIX': No '[LIGAND]' token
        'COORD_NOT_FLOAT': [XYZ]~[END] tokens do not match `xx .xxx`+
        'COORD_SIZE': coord_len % 3 != 0
    smiles: str
    coords: np.ndarray[*, 3]
    """
    tokens = itr.dropwhile(lambda x: x != '[LIGAND]', tokens)
    clauses = split_list(tokens, '[LIGAND]')
    if len(clauses) <= 1:
        return 'COORD_LONG_PREFIX', '', None
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
            return 'COORD_NOT_FLOAT', smiles, None
    coords = np.array(coords)
    if len(coords) % 3 != 0:
        return 'COORD_SIZE', smiles, None
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
    
    atom_idxs = eval(mol.GetProp('_smilesAtomOutputOrder'))
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
                error = 'COORD_FORMAT'
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
            error = 'COORD_FORMAT'
        elif atom is not None:
            error = 'COORD_COUNT'
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
                    error = 'COORD_FORMAT'
                    break
                coord_str += t
                try:
                    coord = float(coord_str)
                except ValueError:
                    error = 'COORD_FORMAT'
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
            error = 'COORD_COUNT'
            return 'COORD_COUNT', atoms, None
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


def parse_qvina_outputs(docked_sdf_path, ref_mol):

    suppl = Chem.SDMolSupplier(docked_sdf_path)
    results = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        line = mol.GetProp('REMARK').splitlines()[0].split()[2:]
        try:
            rmsd = CalcRMS(ref_mol, mol)
        except:
            rmsd = np.nan
        results.append({
            'rdmol': mol,
            'mode_id': i,
            'affinity': float(line[0]),
            'rmsd_lb': float(line[1]),
            'rmsd_ub': float(line[2]),
            'rmsd_ref': rmsd
        })

    return results

def eval_qvina(lig_path, rec_path, out_dir, lig_idx=0, conda_env='adt', use_uff=True, center=None, exhaustiveness=16, timeout: Optional[float]=None):
    """
    Pocket2Molの実装を再現
    """
    logger = getLogger('eval_qvina')
    try:
        out_dir = os.path.realpath(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        rec_path = os.path.realpath(rec_path)

        mol = list(Chem.SDMolSupplier(lig_path))[lig_idx]
        mol = Chem.AddHs(mol, addCoords=True)
        if use_uff:
            try:
                not_converge = 10
                while not_converge > 0:
                    flag = UFFOptimizeMolecule(mol)
                    not_converge = min(not_converge - 1, flag * 10)
            except RuntimeError:
                pass
        sdf_writer = Chem.SDWriter(f"{out_dir}/lig.sdf")
        sdf_writer.write(mol)
        sdf_writer.close()
        noH_rdmol = Chem.RemoveHs(mol)


        pos = mol.GetConformer(0).GetPositions()
        if center is None:
            center = (pos.max(0) + pos.min(0)) / 2
        else:
            center = center

        proc = None
        results = None
        docked_sdf_path = None


        mol = next(pybel.readfile('sdf', f"{out_dir}/lig.sdf"))
        mol.write('pdbqt', f"{out_dir}/lig.pdbqt", overwrite=True)


        commands = f"""
source /workspace/envs/cu124/3.11.9/conda/.bashrc
conda activate {conda_env}
cd {out_dir}
# Prepare receptor (PDB->PDBQT)
prepare_receptor4.py -r {rec_path} -o rec.pdbqt
/workspace/github/Pocket2Mol/experiments/qvina/qvina02 \
--receptor rec.pdbqt \
--ligand lig.pdbqt \
--center_x {center[0]:.4f} \
--center_y {center[1]:.4f} \
--center_z {center[2]:.4f} \
--size_x 20 --size_y 20 --size_z 20 \
--exhaustiveness {exhaustiveness}
        """

        docked_sdf_path = os.path.join(out_dir, 'lig_out.sdf')

        proc = subprocess.Popen(
            '/bin/bash', 
            shell=False, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )

        proc.stdin.write(commands.encode('utf-8'))
        proc.stdin.close()
        start = time()
        while proc is None:
            if timeout is not None and time()-start > timeout:
                logger.warning(f"qvina subprocess reached timeout({timeout})", flush=True)
                return None
        while proc.poll() is None:
            if timeout is not None and time()-start > timeout:
                logger.warning(f"qvina subprocess reached timeout({timeout})", flush=True)
                return None
            
        mol = next(pybel.readfile('pdbqt', F"{out_dir}/lig_out.pdbqt"))
        mol.addh()
        mol.write('sdf', f"{out_dir}/lig_out.sdf", overwrite=True)    


        results = parse_qvina_outputs(docked_sdf_path, noH_rdmol)
        return results[0]['affinity']
    except:
        logger.info('[Error] Vina error: %s' % docked_sdf_path)
        logger.info(f"output: ")
        logger.info(proc.stdout.read().decode())
        logger.info(f"stderr:")
        logger.info(proc.stderr.read().decode())
        return None


def eval_qvina2(lig_path, rec_path, out_dir, lig_idx=0, use_uff=True, center=None, exhaustiveness=16, timeout: Optional[float]=None):
    """
    Pocket2Molの実装から変更
    """
    logger = getLogger('eval_qvina')
    try:
        out_dir = os.path.realpath(out_dir)
        os.makedirs(out_dir, exist_ok=True)


        rec_path = os.path.realpath(rec_path)

        mol = list(Chem.SDMolSupplier(lig_path))[lig_idx]
        mol = Chem.AddHs(mol, addCoords=True)
        if use_uff:
            try:
                not_converge = 10
                while not_converge > 0:
                    flag = UFFOptimizeMolecule(mol)
                    not_converge = min(not_converge - 1, flag * 10)
            except RuntimeError:
                pass
        sdf_writer = Chem.SDWriter(f"{out_dir}/lig.sdf")
        sdf_writer.write(mol)
        sdf_writer.close()
        noH_rdmol = Chem.RemoveHs(mol)


        pos = mol.GetConformer(0).GetPositions()
        if center is None:
            center = (pos.max(0) + pos.min(0)) / 2
        else:
            center = center

        proc = None
        results = None
        docked_sdf_path = None


        mol = next(pybel.readfile('sdf', f"{out_dir}/lig.sdf"))
        mol.write('pdbqt', f"{out_dir}/lig.pdbqt", overwrite=True)

        path_to_qvina = os.environ.get('QVINA_PATH', f"{WORKDIR}/github/qvina/bin/qvina02")
        commands = f"""
cd {out_dir}
echo $PATH
# Prepare receptor (PDB->PDBQT)
python {prepare_receptor4.__file__} -r {rec_path} -o rec.pdbqt
{path_to_qvina} \
--receptor rec.pdbqt \
--ligand lig.pdbqt \
--center_x {center[0]:.4f} \
--center_y {center[1]:.4f} \
--center_z {center[2]:.4f} \
--size_x 20 --size_y 20 --size_z 20 \
--exhaustiveness {exhaustiveness}
        """

        docked_sdf_path = os.path.join(out_dir, 'lig_out.sdf')

        proc = subprocess.Popen(
            '/bin/bash', 
            shell=False, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )

        proc.stdin.write(commands.encode('utf-8'))
        proc.stdin.close()
        start = time()
        while proc.poll() is None:
            if timeout is not None and time()-start > timeout:
                print(f"qvina subprocess reached timeout({timeout})", flush=True)
                return None
            
        mol = next(pybel.readfile('pdbqt', F"{out_dir}/lig_out.pdbqt"))
        mol.addh()
        mol.write('sdf', f"{out_dir}/lig_out.sdf", overwrite=True)    


        results = parse_qvina_outputs(docked_sdf_path, noH_rdmol)
        return results[0]['affinity']
    except:
        logger.info('[Error] Vina error: %s' % docked_sdf_path)
        logger.info(f"output: ")
        logger.info(proc.stdout.read().decode())
        logger.info(f"stderr:")
        logger.info(proc.stderr.read().decode())
        return None


def eval_qvina3(lig_path, rec_path, out_dir, lig_idx=0, use_uff=True, center=None, exhaustiveness=16, timeout: Optional[float]=None, pbar: Optional[wtqdm] = None, verbose: bool=False, cpu: int|None = None):
    """
    Pocket2Molの実装から変更
    """
    def log(name):
        if pbar is not None:
            pbar.start(name)
        if verbose:
            print(f"---{name}---", flush=True)
    logger = getLogger('eval_qvina')
    log('qvina_prep_mol1')
    proc = None
    docked_sdf_path = None
    try:
        out_dir = os.path.realpath(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        rec_path = os.path.realpath(rec_path)

        mol = list(Chem.SDMolSupplier(lig_path))[lig_idx]
        mol = Chem.AddHs(mol, addCoords=True)
        if use_uff:
            log('qvina_uff')
            try:
                not_converge = 10
                while not_converge > 0:
                    flag = UFFOptimizeMolecule(mol)
                    not_converge = min(not_converge - 1, flag * 10)
            except RuntimeError:
                pass
        log('qvina_prep_mol2')
        sdf_writer = Chem.SDWriter(f"{out_dir}/lig.sdf")
        sdf_writer.write(mol)
        sdf_writer.close()
        noH_rdmol = Chem.RemoveHs(mol)


        pos = mol.GetConformer(0).GetPositions()
        if center is None:
            center = (pos.max(0) + pos.min(0)) / 2
        else:
            center = center

        mol = next(pybel.readfile('sdf', f"{out_dir}/lig.sdf"))
        mol.write('pdbqt', f"{out_dir}/lig.pdbqt", overwrite=True)
        log('qvina_prep_rec')
        prepare_receptor4_func(['-r', rec_path, '-o', f'{out_dir}/rec.pdbqt'])

        log('qvina_command')

        docked_sdf_path = os.path.join(out_dir, 'lig_out.sdf')

        proc = subprocess.Popen(
            '/bin/bash', 
            shell=False, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        path_to_qvina = os.environ.get('QVINA_PATH', f"{WORKDIR}/github/qvina/bin/qvina02")
        commands = f"cd {out_dir} && {path_to_qvina} --receptor rec.pdbqt --ligand lig.pdbqt --center_x {center[0]:.4f} --center_y {center[1]:.4f} --center_z {center[2]:.4f} --size_x 20 --size_y 20 --size_z 20 --exhaustiveness {exhaustiveness}"
        if cpu is not None:
            commands += f" --cpu {cpu}"
        proc.stdin.write(commands.encode('utf-8'))
        proc.stdin.close()
        start = time()
        while proc.poll() is None:
            if timeout is not None and time()-start > timeout:
                print(f"qvina subprocess reached timeout({timeout})", flush=True)
                return None
        
        log('qvina_parse')
        mol = next(pybel.readfile('pdbqt', F"{out_dir}/lig_out.pdbqt"))
        mol.addh()
        mol.write('sdf', f"{out_dir}/lig_out.sdf", overwrite=True)    

        results = parse_qvina_outputs(docked_sdf_path, noH_rdmol)
        return results[0]['affinity']
    except:
        log('qvina_error')
        logger.info('[Error] Vina error: %s' % docked_sdf_path)
        if proc is not None:
            logger.info(f"output: ")
            logger.info(proc.stdout.read().decode())
            logger.info(f"stderr:")
            logger.info(proc.stderr.read().decode())
        return None
