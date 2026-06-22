from rdkit import Chem
from openbabel.openbabel import OBMol, OBConversion

def rdmol2obmol(rdmol: Chem.Mol) -> OBMol:
    sdf = Chem.MolToMolBlock(rdmol) # hydrogens remain
    obc = OBConversion()
    obc.SetInFormat('sdf')
    obmol = OBMol()
    obc.ReadString(obmol, sdf)
    return obmol

def obmol2rdmol(obmol: OBMol, sanitize: bool=True) -> Chem.Mol:
    """
    MolFromMolBlockだとPropが無視される。
    
    """
    obc = OBConversion()
    obc.SetOutFormat('sdf')
    sdf = obc.WriteString(obmol)
    ms = Chem.SDMolSupplier()
    ms.SetData(sdf, removeHs=False, sanitize=sanitize)
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
