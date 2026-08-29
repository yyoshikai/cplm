import os, subprocess
from logging import getLogger
from pathlib import Path
import subprocess
from typing import TypeVar, Optional
from time import time
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
from vina import Vina
from openbabel import openbabel as ob
from openbabel.openbabel import OBMol, OBConversion
from ..prepare_receptor4 import main as prepare_receptor4_func
from ..utils import silence_print
from ..utils.path import make_pardir, WORKDIR
from ..chem import sdf2obmol, pdb2obmol, rdmol2obmol, get_coords
logger = getLogger(__name__)
root_dir = Path(__file__).parents[2]

T = TypeVar('T')

DELIM = "__REC_LIG_DELIM__"

def _eval_vina(lig_sdf, rec_pdb, rec_pdbqt_path):
    ligand = sdf2obmol(lig_sdf)
    obc = OBConversion()
    obc.SetOutFormat('pdbqt')
    obc.AddOption('c', obc.OUTOPTIONS)
    ligand.AddHydrogens()
    ligand_str = obc.WriteString(ligand)
    lig_center = get_coords(ligand).mean(axis=0)
    make_pardir(rec_pdbqt_path)
    rec = pdb2obmol(rec_pdb)
    rec.AddHydrogens()
    obc.AddOption('r', obc.OUTOPTIONS)
    obc.WriteFile(rec, rec_pdbqt_path)
    obc.CloseOutFile()
    v = Vina(verbosity=0)
    v.set_receptor(rec_pdbqt_path)
    v.set_ligand_from_string(ligand_str)
    v.compute_vina_maps(lig_center.tolist(), [20, 20, 20])
    score = v.score()[0]
    min_score = v.optimize()[0]
    return score, min_score

def eval_vina(lig_sdf: str, rec_pdb: str, rec_pdbqt_path: str) -> tuple[float, float]:
    """
    処理の本体は _eval_vina_worker
    普通に実行するとVinaでC++のエラー(pythonで捕捉できず処理が止まる) が出ることがあるので, multiprocessingで処理を分離

    Parameters
    ----------
    lig_sdf: str
        SDF string
    rec_pdb: str
        PDB string
    """
    r = subprocess.run(
        f"python -m src.evaluate._eval_vina_worker {rec_pdbqt_path}".split(' '),
        cwd=root_dir,
        input = f"{lig_sdf}{DELIM}{rec_pdb}", 
        capture_output=True, text=True
    )
    if r.returncode == 0:
        score, min_score = map(float, r.stdout.strip().split(' '))
        return score, min_score, None
    else:
        return None, None, r.stderr

def parse_qvina_out(path: str) -> float:
    obc = OBConversion()
    obc.SetInFormat('pdbqt')
    lig_out_obmol = OBMol()
    obc.ReadFile(lig_out_obmol, path)
    affinity = float(ob.toPairData(lig_out_obmol.GetData('REMARK')).GetValue().split()[2])
    return affinity


def eval_qvina(ligand: Chem.Mol|str, rec_pdb_path: str, out_dir: str, use_uff=True, center=None, exhaustiveness=16, timeout: Optional[float]=None, cpu: int|None = None, print_prepare: bool=True):
    """
    Returns
    -------
    affinity: float|None
        If Error, None is returned.
    error: None|'timeout'|tuple[e, str, str]
        if no error, None is returned.
    stdout: str|None
        stdout of qvina command
    stderr: str|None
        stderr of qvina command
    """
    if isinstance(ligand, str):
        ligand = Chem.MolFromMolBlock(ligand)
    if ligand is None:
        return None, 'input_ligand_is_invalid', '', ''

    stdout = stderr = affinity = None
    obc = OBConversion()
    out_dir = os.path.realpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    mol = Chem.AddHs(ligand, addCoords=True)
    if use_uff:
        try:
            not_converge = 10
            while not_converge > 0:
                flag = UFFOptimizeMolecule(mol)
                not_converge = min(not_converge - 1, flag * 10)
        except RuntimeError:
            pass
    pos = mol.GetConformer(0).GetPositions()
    if center is None:
        center = (pos.max(0) + pos.min(0)) / 2
    
    lig_obmol = rdmol2obmol(mol)
    obc.SetOutFormat('pdbqt')
    obc.WriteFile(lig_obmol, f"{out_dir}/lig.pdbqt")
    
    with silence_print(not print_prepare):
        prepare_receptor4_func(['-r', rec_pdb_path, '-o', f'{out_dir}/rec.pdbqt'])

    path_to_qvina = os.environ.get('QVINA_PATH', f"{WORKDIR}/github/qvina/qvina02")

    args = [
        path_to_qvina,
        '--receptor', 'rec.pdbqt',
        '--ligand', 'lig.pdbqt',
        '--center_x', f'{center[0]:.4f}',
        '--center_y', f'{center[1]:.4f}',
        '--center_z', f'{center[2]:.4f}',
        '--size_x', '20',
        '--size_y', '20',
        '--size_z', '20',
        '--exhaustiveness', str(exhaustiveness),
    ]

    if cpu is not None:
        args += ['--cpu', str(cpu)]

    proc = subprocess.Popen(
        args,
        cwd=out_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"qvina subprocess reached timeout({timeout})", flush=True)
        proc.kill()
        stdout, stderr = proc.communicate()
        return None, 'timeout', stdout, stderr

    try:
        affinity = parse_qvina_out(f"{out_dir}/lig_out.pdbqt")
        return affinity, None, stdout, stderr
    except AttributeError: # /lig_out.pdbqt が存在しないなど
        return None, 'qvina', stdout, stderr
