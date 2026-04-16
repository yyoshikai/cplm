import re
from ctypes import c_double
from logging import getLogger
from typing import Literal
import numpy as np
from rdkit import Chem
from rdkit.Chem import Conformer
from rdkit.Geometry import Point3D
from openbabel.openbabel import OBMol, OBConversion

logger = getLogger(__name__)

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

def _randomize(mol: Chem.Mol, rng: np.random.Generator):
    for i in range(100): # ランダムに失敗する場合がある
        try:
            idxs = np.arange(mol.GetNumAtoms(), dtype=int)
            rng.shuffle(idxs)
            mol = Chem.RenumberAtoms(mol, idxs.tolist())
            return mol, Chem.MolToSmiles(mol, canonical=False)
        except Exception as e:
            print(e)
            pass
    else:
        logger.warning("randomize_smiles failed for 100 times.")
        return mol, ""

def randomize_smiles(mol: Chem.Mol, rng: np.random.Generator):
    return _randomize(mol, rng)[1]

# randomize/canonicalize
# refer /workspace/cplm/experiments/tests/source.ipynb "260204 canonical"
def set_atom_order(mol: Chem.Mol, random: bool, rng: np.random.Generator) -> Chem.Mol:
    if random:
        mol, ran = _randomize(mol, rng)
    else:
        can = Chem.MolToSmiles(mol, canonical=True)
    mol = Chem.RenumberAtoms(mol, eval(mol.GetProp('_smilesAtomOutputOrder')))
    return mol

def mol_from_atoms_coords(atoms: list[str], coords: np.ndarray) -> tuple[Chem.Mol|None, str|None]:
    """
    原子の元素記号と座標からChem.Molを作成
    
    Returns:
        Chem.Mol|None: 分子、エラーの場合None
        str|None: エラーの場合その内容、エラーでない場合None
    """

def read_pdb_path(path: str, out_cls: Literal['ob', 'rdkit', 'text']):
    if out_cls == 'ob':
        mol = OBMol()
        obc = OBConversion()
        obc.SetInFormat('pdb')
        obc.ReadFile(mol, path)
    elif out_cls == 'rdkit':
        mol = Chem.MolFromPDBFile(path, sanitize=False)
        if mol is None:
            # Error process at read_pdb_text
            with open(path) as f:
                mol = read_pdb_text(f.read(), out_cls)
        
        params = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        Chem.SanitizeMol(mol, sanitizeOps=params) # このようにしておくことで, エラーを回避しつつ水素付加などができる
    elif out_cls == 'text':
        with open(path) as f:
            mol = f.read()
    else:
        raise ValueError(f"Unknown {out_cls=}")
    return mol

def read_pdb_text(text: str, out_cls: Literal['ob', 'rdkit', 'text']):
    if out_cls == 'ob':
        mol = OBMol()
        obc = OBConversion()
        obc.SetInFormat('pdb')
        obc.ReadString(mol, text)
    elif out_cls == 'rdkit':
        mol = Chem.MolFromPDBBlock(text, sanitize=False, removeHs=True)
        if mol is None:
            # 'GLX' という, GLNかGLUか分からないアミノ酸があると読み込みエラーになる。
            # ので, とりあえずGLNに置換する(GLUは電荷がある)。
            text = re.sub(
                r"(ATOM  "
                r"[ 0-9]{5} "
                r" )XE1("
                r"."
                r")GLX( "
                r"."
                r"[ 0-9]{4}"
                r".   "
                r"[ \-0-9]{4}\.[0-9]{3}"
                r"[ \-0-9]{4}\.[0-9]{3}"
                r"[ \-0-9]{4}\.[0-9]{3}"
                r"[ \-0-9.]{6}"
                r"[ \-0-9.]{6}          "
                r" )X("
                r"[ +\-0-9]{2}\n)", 
                r"\1OE1\2GLX\3O\4", text
            )
            text = re.sub(
                r"(ATOM  "
                r"[ 0-9]{5} "
                r" )XE2("
                r"."
                r")GLX( "
                r"."
                r"[ 0-9]{4}"
                r".   "
                r"[ \-0-9]{4}\.[0-9]{3}"
                r"[ \-0-9]{4}\.[0-9]{3}"
                r"[ \-0-9]{4}\.[0-9]{3}"
                r"[ \-0-9.]{6}"
                r"[ \-0-9.]{6}          "
                r" )X("
                r"[ +\-0-9]{2}\n)", 
                r"\1NE2\2GLX\3N\4", text
            )
            # ASX も同様
            text = re.sub(
                r"(ATOM  "
                r"[ 0-9]{5} "
                r" )XD1("
                r"."
                r")ASX( "
                r"."
                r"[ 0-9]{4}"
                r".   "
                r"[ \-0-9]{4}\.[0-9]{3}"
                r"[ \-0-9]{4}\.[0-9]{3}"
                r"[ \-0-9]{4}\.[0-9]{3}"
                r"[ \-0-9.]{6}"
                r"[ \-0-9.]{6}          "
                r" )X("
                r"[ +\-0-9]{2}\n)", 
                r"\1OD1\2ASX\3O\4", text
            )
            text = re.sub(
                r"(ATOM  "
                r"[ 0-9]{5} "
                r" )XD2("
                r"."
                r")ASX( "
                r"."
                r"[ 0-9]{4}"
                r".   "
                r"[ \-0-9]{4}\.[0-9]{3}"
                r"[ \-0-9]{4}\.[0-9]{3}"
                r"[ \-0-9]{4}\.[0-9]{3}"
                r"[ \-0-9.]{6}"
                r"[ \-0-9.]{6}          "
                r" )X("
                r"[ +\-0-9]{2}\n)", 
                r"\1ND2\2ASX\3N\4", text
            )

            # HETATM ... X という, 何か分からない原子があると読み込みエラーになる。
            # ので, それらは削除する
            text = re.sub(
                r"HETATM"
                r"[ 0-9]{5} "
                r"( UNK|UNK ).UNX ."
                r"[ 0-9]{4}"
                r".   "
                r"[ \-0-9]{4}\.[0-9]{3}"
                r"[ \-0-9]{4}\.[0-9]{3}"
                r"[ \-0-9]{4}\.[0-9]{3}"
                r"[ \-0-9.]{6}"
                r"[ \-0-9.]{6}          "
                r" X"
                r"[ +\-0-9]{2}\n", 
                r"", text
            )
            mol = Chem.MolFromPDBBlock(text, sanitize=False, removeHs=True)

        params = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
        Chem.SanitizeMol(mol, sanitizeOps=params) # このようにしておくことで, エラーを回避しつつ水素付加などができる

    elif out_cls == 'text':
        mol = text
    else:
        raise ValueError(f"Unknown {out_cls=}")
    return mol

def element_symbols() -> list[str]:
    table = Chem.GetPeriodicTable()
    return [table.GetElementSymbol(i) for i in range(1, 119)]
ELEMENT_SYMBOLS = element_symbols()