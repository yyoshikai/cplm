import logging, gzip
from collections.abc import Iterator
from rdkit import Chem, rdBase, RDLogger
import numpy as np

def ignore_warning():
    RDLogger.DisableLog("rdApp.*")

def set_rdkit_logger(propagate=True):
    rdBase.LogToPythonLogger()
    rdkit_logger = logging.getLogger('rdkit')
    rdkit_logger.propagate = propagate
    for handler in rdkit_logger.handlers: 
        rdkit_logger.removeHandler(handler)
    rdBase.LogToPythonLogger()

sanitize_ops = 0
for k,v in Chem.rdmolops.SanitizeFlags.values.items():
    if v not in [Chem.rdmolops.SanitizeFlags.SANITIZE_CLEANUP,
                Chem.rdmolops.SanitizeFlags.SANITIZE_ALL]:
        sanitize_ops |= v

def randomize(can, mol, rstate: np.random.RandomState):
    nums = np.arange(mol.GetNumAtoms())
    for i_trial in range(100):
        rstate.shuffle(nums)
        mol = Chem.RenumberAtoms(mol, nums.tolist())
        ran = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)
        if Chem.MolToSmiles(Chem.MolFromSmiles(ran)) == can: break
    else:
        raise ValueError("Randomization failed.")
    return ran

def load_sdf_gz(path: str) -> list[Chem.Mol]:
    with gzip.open(path) as f:
        return list(Chem.ForwardSDMolSupplier(f))

