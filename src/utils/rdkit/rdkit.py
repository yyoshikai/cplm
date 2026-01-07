import logging, gzip
from collections import defaultdict
from collections.abc import Iterator
from rdkit import Chem, rdBase, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np

def ignore_rdkit_warning():
    RDLogger.DisableLog("rdApp.*")

def set_rdkit_logger():
    rdkit_logger = logging.getLogger('rdkit')
    rdBase.LogToPythonLogger()
    for handler in rdkit_logger.handlers: 
        rdkit_logger.removeHandler(handler)
    return rdkit_logger

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

# From /workspace/cheminfodata/molnet/preprocess/240408_split_gem/source.ipynb
def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles

    Args:
        smiles: smiles sequence
        include_chirality: Default=False
    
    Return: 
        the scaffold of the given smiles.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

def split(dataset, 
        frac_train=None, 
        frac_valid=None, 
        frac_test=None,
        seed=None):
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    N = len(dataset)

    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind in range(N):
        scaffold = generate_scaffold(dataset[ind], include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))

    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    return train_idx, valid_idx, test_idx