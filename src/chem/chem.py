import re
from ctypes import c_double
from dataclasses import dataclass
from logging import getLogger
from typing import Literal
import numpy as np
from rdkit import Chem
from rdkit.Chem import Conformer
from rdkit.Geometry import Point3D
from openbabel.openbabel import OBMol, OBConversion
from openbabel import openbabel as ob

@dataclass
class Pocket:
    atoms: np.ndarray
    coord: np.ndarray

    def __post_init__(self):
        assert len(self.atoms) == len(self.coord)
        assert self.coord.ndim == 2 and self.coord.shape[1] == 3

logger = getLogger(__name__)

def set_conf(conf: Conformer, coord: np.ndarray):
    for i in range(len(coord)):
        conf.SetAtomPosition(i, Point3D(*coord[i].tolist()))

def array_to_conf(coord: np.ndarray) -> Conformer:
    conf = Conformer()
    set_conf(conf, coord)
    return conf

def get_coords(mol: OBMol|Chem.Mol|Pocket) -> np.ndarray:
    if isinstance(mol, OBMol):
        coord = mol.GetCoordinates()
        return np.array((c_double * (mol.NumAtoms()*3)).from_address(int(coord))).reshape(-1, 3)
    elif isinstance(mol, Chem.Mol):
        return mol.GetConformer().GetPositions()
    elif isinstance(mol, Pocket):
        return mol.coord
    else:
        raise ValueError(f"Unknown {type(mol)=}")

def get_atoms(mol: Chem.Mol|ob.OBMol) -> list[str]:
    """
    Calpha炭素はCA
    それ以外は元素記号 (2文字目は小文字 Mg等)
    """
    if isinstance(mol, ob.OBMol):
        atoms = []
        for atom in ob.OBMolAtomIter(mol):
            res = atom.GetResidue()
            if res is not None and res.GetAtomID(atom) == ' CA ':
                atom = 'CA'
            else:
                atom = ELEMENT_SYMBOLS[atom.GetAtomicNum()-1]
            atoms.append(atom)
    elif isinstance(mol, Chem.Mol):
        atoms = []
        for atom in mol.GetAtoms():
            rinfo = atom.GetPDBResidueInfo()
            if rinfo is not None and rinfo.GetName() == ' CA ':
                atom = 'CA'
            else:
                atom = atom.GetSymbol()
            atoms.append(atom)
    else:
        raise ValueError
    return atoms

def set_coords(mol: OBMol|Chem.Mol|Pocket, coords: np.ndarray) -> None:
    if isinstance(mol, Chem.Mol):
        if mol.GetNumConformers() > 0:
            # confへの代入のみで元の分子も変更されることを確認 @tests/test.ipynb
            set_conf(mol.GetConformer(), coords)
        else:
            mol.AddConformer(array_to_conf(coords))
    elif isinstance(mol, Pocket):
        mol.coord = coords
    else:
        for i, atom in enumerate(ob.OBMolAtomIter(mol)):
            atom.SetVector(coords[i].tolist())

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