import os
from logging import getLogger
import subprocess
from typing import TypeVar, Optional
from time import time
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
from vina import Vina
from openbabel import openbabel as ob
from openbabel.openbabel import OBMol, OBConversion
from .prepare_receptor4 import main as prepare_receptor4_func
from .utils.path import make_pardir, WORKDIR
from .chem import sdf2obmol, pdb2obmol, rdmol2obmol, get_coord_from_mol
logger = getLogger(__name__)

T = TypeVar('T')

def eval_vina(ligand: OBMol|str, rec: OBMol|str, rec_pdbqt_path: str) -> tuple[float, float]:
    """
    cf.ProcessPoolExecutorに投げられるよう, strも受け付ける。

    Parameters
    ----------
    ligand: OBMol|str
        OBMol object or SDF string
    protein: OBMol|str
        OBMol object or PDB string
    """
    score = min_score = error = None
    try:
        if isinstance(ligand, str):
            ligand = sdf2obmol(ligand)
        obc = OBConversion()
        obc.SetOutFormat('pdbqt')
        obc.AddOption('c', obc.OUTOPTIONS)
        ligand.AddHydrogens()
        ligand_str = obc.WriteString(ligand)
        lig_center = get_coord_from_mol(ligand).mean(axis=0)

        make_pardir(rec_pdbqt_path)
        if isinstance(rec, str):
            rec = pdb2obmol(rec)
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
    except Exception as e:
        error = e
    return score, min_score, error

def parse_qvina_out(path: str) -> float:
    obc = OBConversion()
    obc.SetInFormat('pdbqt')
    lig_out_obmol = OBMol()
    obc.ReadFile(lig_out_obmol, path)
    affinity = float(ob.toPairData(lig_out_obmol.GetData('REMARK')).GetValue().split()[2])
    return affinity


def eval_qvina(ligand: Chem.Mol|str, rec_pdb_path: str, out_dir: str, use_uff=True, center=None, exhaustiveness=16, timeout: Optional[float]=None, cpu: int|None = None):
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

    stdout = stderr = affinity = None
    obc = OBConversion()
    try:
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
        
        prepare_receptor4_func(['-r', rec_pdb_path, '-o', f'{out_dir}/rec.pdbqt'])

        proc = subprocess.Popen('/bin/bash', shell=False, stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        path_to_qvina = os.environ.get('QVINA_PATH', f"{WORKDIR}/github/qvina/bin/qvina02")
        command = f"cd {out_dir} && {path_to_qvina} --receptor rec.pdbqt --ligand lig.pdbqt --center_x {center[0]:.4f} --center_y {center[1]:.4f} --center_z {center[2]:.4f} --size_x 20 --size_y 20 --size_z 20 --exhaustiveness {exhaustiveness}"
        if cpu is not None:
            command += f" --cpu {cpu}"
        proc.stdin.write(command.encode('utf-8'))
        proc.stdin.close()
        start = time()
        while proc.poll() is None:
            if timeout is not None and time()-start > timeout:
                print(f"qvina subprocess reached timeout({timeout})", flush=True)
                return None, 'timeout', stdout, stderr
        stdout = proc.stdout.read().decode()
        stderr = proc.stderr.read().decode()
        
        affinity = parse_qvina_out(f"{out_dir}/lig_out.pdbqt")
        return affinity, None, stdout, stderr
    except Exception as e:
        return affinity, e, stdout, stderr
