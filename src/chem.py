from ctypes import c_double
import numpy as np
from rdkit import Chem
from rdkit.Chem import Conformer
from rdkit.Geometry import Point3D
from openbabel.openbabel import OBMol, OBConversion

def rdmol2obmol(rdmol: Chem.Mol) -> OBMol:
    sdf = Chem.MolToMolBlock(rdmol) # hydrogens remain
    obc = OBConversion()
    obc.SetInFormat('sdf')
    obmol = OBMol()
    obc.ReadString(obmol, sdf)
    return obmol

def obmol2rdmol(obmol: OBMol) -> Chem.Mol:
    """
    MolFromMolBlockだとPropが無視される。
    
    """
    obc = OBConversion()
    obc.SetOutFormat('sdf')
    sdf = obc.WriteString(obmol)
    ms = Chem.SDMolSupplier()
    ms.SetData(sdf, removeHs=False)
    return next(ms)

def pdb2obmol(pdb: str) -> OBMol:
    obc = OBConversion()
    obc.SetInFormat('pdb')
    obmol = OBMol()
    obc.ReadString(obmol, pdb)
    return obmol

def sdf2obmol(sdf: str) -> OBMol:
    obc = OBConversion()
    obc.SetInFormat('sdf')
    obmol = OBMol()
    obc.ReadString(obmol, sdf)
    return obmol

def obmol2pdb(obmol: OBMol) -> str:
    obc = OBConversion()
    obc.SetOutFormat('pdb')
    return obc.WriteString(obmol)

def sdf_path2obmol(sdf_path: str) -> OBMol:
    with open(sdf_path) as f:
        return sdf2obmol(f.read())
    
def pdb_path2obmol(pdb_path: str) -> OBMol:
    with open(pdb_path) as f:
        return pdb2obmol(f.read())

def set_conf(conf: Conformer, coord: np.ndarray):
    for i in range(len(coord)):
        conf.SetAtomPosition(i, Point3D(*coord[i].tolist()))

def array_to_conf(coord: np.ndarray) -> Conformer:
    conf = Conformer()
    set_conf(conf, coord)
    return conf

def get_coord_from_mol(mol: OBMol) -> np.ndarray:
    coord = mol.GetCoordinates()
    return np.array((c_double * (mol.NumAtoms()*3)).from_address(int(coord))).reshape(-1, 3)