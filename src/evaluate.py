import os
from logging import getLogger
import subprocess
from typing import TypeVar, Optional
from time import time
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
from vina import Vina
from openbabel.openbabel import OBMol, OBConversion
from .prepare_receptor4 import main as prepare_receptor4_func
from .utils.time import wtqdm
logger = getLogger(__name__)
WORKDIR = os.environ.get('WORKDIR', "/workspace")

T = TypeVar('T')

def rdmol2obmol(rdmol: Chem.Mol) -> OBMol:
    sdf = Chem.MolToMolBlock(rdmol) # hydrogens remain
    obc = OBConversion()
    obc.SetInFormat('sdf')
    obmol = OBMol()
    obc.ReadString(obmol, sdf)
    return obmol

def obmol2rdmol(obmol: OBMol) -> Chem.Mol:
    obc = OBConversion()
    obc.SetOutFormat('sdf')
    sdf = obc.WriteString(obmol)
    return Chem.MolFromMolBlock(sdf, removeHs=False)

def eval_vina(ligand: OBMol, protein: OBMol, protein_path: str) -> tuple[float, float]:
    obc = OBConversion()
    obc.SetOutFormat('pdbqt')
    
    ligand.AddHydrogens()
    ligand_str = obc.WriteString(ligand)
    protein.AddHydrogens()
    obc.WriteFile(protein, protein_path)

    v = Vina(verbosity=0)
    v.set_receptor(protein_path)
    v.set_ligand_from_string(ligand_str)
    v.compute_vina_maps()
    score = v.score()[0]
    min_score = v.optimize()[0]
    return score, min_score

def eval_qvina(ligand: Chem.Mol, rec_path: str, out_dir: str, use_uff=True, center=None, exhaustiveness=16, timeout: Optional[float]=None, pbar: Optional[wtqdm] = None, verbose: bool=False, cpu: int|None = None):
    def log(name):
        if pbar is not None:
            pbar.start(name)
        if verbose:
            print(f"---{name}---", flush=True)
    logger = getLogger('eval_qvina')
    log('qvina_prep_mol1')
    proc = None
    obc = OBConversion()
    try:
        out_dir = os.path.realpath(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        mol = Chem.AddHs(ligand, addCoords=True)
        if use_uff:
            log('qvina_uff')
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
        
        log('qvina_prep_mol2')
        lig_obmol = rdmol2obmol(mol)
        obc.SetOutFormat('pdbqt')
        obc.WriteFile(lig_obmol, f"{out_dir}/lig.pdbqt")
        
        log('qvina_prep_rec')
        prepare_receptor4_func(['-r', rec_path, '-o', f'{out_dir}/rec.pdbqt'])

        log('qvina_command')
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
                return None
        
        log('qvina_parse')
        lig_out_obmol = OBMol()
        obc.SetInFormat('pdbqt')
        obc.ReadFile(lig_out_obmol, f"{out_dir}/lig_out.pdbqt")
        lig_out_rdmol = obmol2rdmol(lig_out_obmol)
        affinity = lig_out_rdmol.GetProp('REMARK').splitlines()[0].split()[2]
        return affinity
    except:
        log('qvina_error')
        logger.info(f'[Error] Vina error at {out_dir}')
        if proc is not None:
            logger.info(f"output: ")
            logger.info(proc.stdout.read().decode())
            logger.info(f"stderr:")
            logger.info(proc.stderr.read().decode())
        return None
