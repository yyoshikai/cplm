import numpy as np
from rdkit import Chem

import math
from rdkit import Chem

# XS (X-Score) Atom Types as defined in src/lib/atom_constants.h (lines 80-112)
XS_TYPE_C_H   = 0
XS_TYPE_C_P   = 1
XS_TYPE_N_P   = 2
XS_TYPE_N_D   = 3
XS_TYPE_N_A   = 4
XS_TYPE_N_DA  = 5
XS_TYPE_O_P   = 6
XS_TYPE_O_D   = 7
XS_TYPE_O_A   = 8
XS_TYPE_O_DA  = 9
XS_TYPE_S_P   = 10
XS_TYPE_P_P   = 11
XS_TYPE_F_H   = 12
XS_TYPE_Cl_H  = 13
XS_TYPE_Br_H  = 14
XS_TYPE_I_H   = 15
XS_TYPE_Si    = 16
XS_TYPE_At    = 17
XS_TYPE_Met_D = 18

# XS van der Waals radii as defined in src/lib/atom_constants.h (lines 263-296)
XS_RADII = [
    1.9, # 0: C_H
    1.9, # 1: C_P
    1.8, # 2: N_P
    1.8, # 3: N_D
    1.8, # 4: N_A
    1.8, # 5: N_DA
    1.7, # 6: O_P
    1.7, # 7: O_D
    1.7, # 8: O_A
    1.7, # 9: O_DA
    2.0, # 10: S_P
    2.1, # 11: P_P
    1.5, # 12: F_H
    1.8, # 13: Cl_H
    2.0, # 14: Br_H
    2.2, # 15: I_H
    2.2, # 16: Si
    2.3, # 17: At
    1.2, # 18: Met_D
]

# Standard metal elements in AutoDock Vina (src/lib/atom_constants.h and parse_pdbqt.cpp)
METALS = {
    'LI', 'BE', 'NA', 'MG', 'AL', 'K', 'CA', 'SC', 'TI', 'V', 'CR', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN',
    'GA', 'RB', 'SR', 'Y', 'ZR', 'NB', 'MO', 'TC', 'RU', 'RH', 'PD', 'AG', 'CD', 'IN', 'SN', 'CS', 'BA',
    'LA', 'CE', 'PR', 'ND', 'PM', 'SM', 'EU', 'GD', 'TB', 'DY', 'HO', 'ER', 'TM', 'YB', 'LU', 'HF', 'TA',
    'W', 'RE', 'OS', 'IR', 'PT', 'AU', 'HG', 'TL', 'PB', 'BI', 'U'
}

def get_xs_type(mol, atom):
    """
    Assigns the XS atom type to an RDKit atom object.
    Matches the logic in model::assign_types() in src/lib/model.cpp (lines 409-453).
    """
    elem = atom.GetSymbol().upper()
    
    if elem == 'H' or elem == 'D':
        return None  # Hydrogen is not scored directly in Vina's heavy-atom potential
        
    if elem in METALS:
        return XS_TYPE_Met_D
        
    if elem == 'C':
        # Polar carbon if bonded to any heteroatom (not C, not H)
        has_hetero = False
        for nbr in atom.GetNeighbors():
            if nbr.GetSymbol().upper() not in ('C', 'H'):
                has_hetero = True
                break
        return XS_TYPE_C_P if has_hetero else XS_TYPE_C_H
        
    elif elem == 'N':
        donor = atom.GetTotalNumHs() > 0
        acceptor = True
        
        # Non-acceptors (quaternary N, amide, sulfonamide, pyrrole)
        if atom.GetFormalCharge() > 0:
            acceptor = False
        else:
            # Check for amide or sulfonamide: N bonded to C(=O) or S(=O)
            for nbr in atom.GetNeighbors():
                if nbr.GetSymbol().upper() == 'C':
                    for gnbr in nbr.GetNeighbors():
                        if gnbr.GetSymbol().upper() in ('O', 'S'):
                            bond = mol.GetBondBetweenAtoms(nbr.GetIdx(), gnbr.GetIdx())
                            if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                                acceptor = False
                                break
                elif nbr.GetSymbol().upper() == 'S':
                    for gnbr in nbr.GetNeighbors():
                        if gnbr.GetSymbol().upper() == 'O':
                            bond = mol.GetBondBetweenAtoms(nbr.GetIdx(), gnbr.GetIdx())
                            if bond and bond.GetBondType() == Chem.BondType.DOUBLE:
                                acceptor = False
                                break
                if not acceptor:
                    break
            
            # Aromatic pyrrole-like nitrogens (lone pair conjugated in the ring)
            if acceptor and atom.GetIsAromatic():
                # Pyridine-like: neutral, degree 2 in aromatic ring, 0 Hs
                # Pyrrole-like: neutral, degree 3 in aromatic ring, or has H
                num_aromatic_bonds = sum(1 for b in atom.GetBonds() if b.GetIsAromatic())
                if donor or num_aromatic_bonds > 2:
                    acceptor = False
                    
        if acceptor and donor:
            return XS_TYPE_N_DA
        elif acceptor:
            return XS_TYPE_N_A
        elif donor:
            return XS_TYPE_N_D
        else:
            return XS_TYPE_N_P
            
    elif elem == 'O':
        donor = atom.GetTotalNumHs() > 0
        # Oxygens are always hydrogen bond acceptors in AD4/Vina
        return XS_TYPE_O_DA if donor else XS_TYPE_O_A
        
    elif elem == 'S':
        return XS_TYPE_S_P
        
    elif elem == 'P':
        return XS_TYPE_P_P
        
    elif elem == 'F':
        return XS_TYPE_F_H
        
    elif elem == 'CL':
        return XS_TYPE_Cl_H
        
    elif elem == 'BR':
        return XS_TYPE_Br_H
        
    elif elem == 'I':
        return XS_TYPE_I_H
        
    elif elem == 'SI':
        return XS_TYPE_Si
        
    elif elem == 'AT':
        return XS_TYPE_At
        
    return None

# =====================================================================
# Vina Potentials as defined in src/lib/potentials.h
# =====================================================================

def eval_gauss1(r, d_ij):
    # vina_gaussian(offset=0, width=0.5, cutoff=8.0)
    if r >= 8.0:
        return 0.0
    return math.exp(-((r - d_ij) / 0.5) ** 2)

def eval_gauss2(r, d_ij):
    # vina_gaussian(offset=3.0, width=2.0, cutoff=8.0)
    if r >= 8.0:
        return 0.0
    return math.exp(-((r - (d_ij + 3.0)) / 2.0) ** 2)

def eval_repulsion(r, d_ij):
    # vina_repulsion(offset=0.0, cutoff=8.0)
    if r >= 8.0:
        return 0.0
    d = r - d_ij
    if d > 0.0:
        return 0.0
    return d * d

def eval_hydrophobic(r, d_ij, t1, t2):
    # vina_hydrophobic(good=0.5, bad=1.5, cutoff=8.0)
    if r >= 8.0:
        return 0.0
    
    is_h1 = (t1 in (XS_TYPE_C_H, XS_TYPE_F_H, XS_TYPE_Cl_H, XS_TYPE_Br_H, XS_TYPE_I_H))
    is_h2 = (t2 in (XS_TYPE_C_H, XS_TYPE_F_H, XS_TYPE_Cl_H, XS_TYPE_Br_H, XS_TYPE_I_H))
    
    if is_h1 and is_h2:
        # slope_step(bad=1.5, good=0.5, r - d_ij)
        x = r - d_ij
        if x <= 0.5:
            return 1.0
        elif x >= 1.5:
            return 0.0
        else:
            return 1.5 - x
    return 0.0

def eval_hbond(r, d_ij, t1, t2):
    # vina_non_dir_h_bond(good=-0.7, bad=0.0, cutoff=8.0)
    if r >= 8.0:
        return 0.0
    
    donors = {XS_TYPE_N_D, XS_TYPE_N_DA, XS_TYPE_O_D, XS_TYPE_O_DA, XS_TYPE_Met_D}
    acceptors = {XS_TYPE_N_A, XS_TYPE_N_DA, XS_TYPE_O_A, XS_TYPE_O_DA}
    
    is_hb = (t1 in donors and t2 in acceptors) or (t2 in donors and t1 in acceptors)
    
    if is_hb:
        # slope_step(bad=0.0, good=-0.7, r - d_ij)
        x = r - d_ij
        if x <= -0.7:
            return 1.0
        elif x >= 0.0:
            return 0.0
        else:
            return -x / 0.7
    return 0.0

def curl(e, v=1000.0):
    """
    Soft capping function for positive clash energies to prevent infinity.
    Matches curl() in src/lib/curl.h (lines 29-44).
    """
    if e > 0 and v is not None:
        return e * (v / (v + e))
    return e

# =====================================================================
# Connectivity and Torsion Analysis
# =====================================================================

def is_amide_bond(mol, bond):
    a1 = bond.GetBeginAtom()
    a2 = bond.GetEndAtom()
    if {a1.GetSymbol().upper(), a2.GetSymbol().upper()} == {'C', 'N'}:
        c_atom = a1 if a1.GetSymbol().upper() == 'C' else a2
        # Check if carbon is bonded to double-bonded oxygen
        for nbr in c_atom.GetNeighbors():
            if nbr.GetSymbol().upper() == 'O':
                b = mol.GetBondBetweenAtoms(c_atom.GetIdx(), nbr.GetIdx())
                if b and b.GetBondType() == Chem.BondType.DOUBLE:
                    return True
    return False

def get_rotatable_bonds(mol):
    """
    Finds rotatable bonds excluding terminal rotors and amide bonds.
    Matches standard Vina/Meeko rotatable bond definition.
    """
    rot_bonds = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if bond.IsInRing():
            continue
            
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        
        if a1.GetSymbol().upper() == 'H' or a2.GetSymbol().upper() == 'H':
            continue
            
        # Count heavy neighbors
        n1 = sum(1 for nbr in a1.GetNeighbors() if nbr.GetSymbol().upper() != 'H')
        n2 = sum(1 for nbr in a2.GetNeighbors() if nbr.GetSymbol().upper() != 'H')
        
        # Exclude terminal rotors (like -CH3, -NH2, -OH)
        if n1 > 1 and n2 > 1:
            if is_amide_bond(mol, bond):
                continue
            rot_bonds.append(bond)
    return rot_bonds

def get_rigid_components(mol, rotatable_bonds):
    """
    Identifies rigid segments of the molecule by finding connected components
    after removing all rotatable bonds. Used to exclude fixed intramolecular interactions.
    """
    adj = {atom.GetIdx(): [] for atom in mol.GetAtoms() if atom.GetSymbol().upper() != 'H'}
    
    rot_set = set()
    for bond in rotatable_bonds:
        idx1 = bond.GetBeginAtom().GetIdx()
        idx2 = bond.GetEndAtom().GetIdx()
        rot_set.add(tuple(sorted((idx1, idx2))))
        
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if a1.GetSymbol().upper() == 'H' or a2.GetSymbol().upper() == 'H':
            continue
            
        idx1 = a1.GetIdx()
        idx2 = a2.GetIdx()
        
        if tuple(sorted((idx1, idx2))) not in rot_set:
            adj[idx1].append(idx2)
            adj[idx2].append(idx1)
            
    visited = set()
    components = {}
    comp_id = 0
    for idx in adj:
        if idx not in visited:
            stack = [idx]
            visited.add(idx)
            components[idx] = comp_id
            while stack:
                curr = stack.pop()
                for nbr in adj[curr]:
                    if nbr not in visited:
                        visited.add(nbr)
                        components[nbr] = comp_id
                        stack.append(nbr)
            comp_id += 1
    return components

# =====================================================================
# Main Scoring Functions
# =====================================================================

def calculate_pair_energy(r, t1, t2, weights):
    """
    Calculates raw un-curled interaction energy between a pair of atoms.
    """
    d_ij = XS_RADII[t1] + XS_RADII[t2]
    
    g1 = eval_gauss1(r, d_ij)
    g2 = eval_gauss2(r, d_ij)
    rep = eval_repulsion(r, d_ij)
    hydro = eval_hydrophobic(r, d_ij, t1, t2)
    hbond = eval_hbond(r, d_ij, t1, t2)
    
    e = (weights[0] * g1 +
         weights[1] * g2 +
         weights[2] * rep +
         weights[3] * hydro +
         weights[4] * hbond)
    return e

def eval_vina_atom(molecule, protein):
    """
    Replicates the AutoDock Vina scoring function.
    
    Parameters:
    - molecule: SDF file content (str) or rdkit.Chem.Mol object.
    - protein: PDB file content (str).
    - weights: List of weights [w_gauss1, w_gauss2, w_repulsion, w_hydrophobic, w_hbond, w_glue, w_rot]
               Defaults to Vina values if None.
    - E_intra_unbound: Unbound ligand intramolecular energy. Defaults to E_intra if None.
    
    Returns:
    - A dictionary containing:
        - 'free_energy': Estimated Free Energy of Binding (kcal/mol)
        - 'inter_energy': Intermolecular energy (ligand-receptor)
        - 'intra_energy': Intramolecular energy (ligand-ligand)
        - 'num_rotatable_bonds': Active rotatable bonds count (N_rot)
    """
    
    # Default AutoDock Vina weights as defined in src/lib/vina.h (lines 123-126)
    weights = [-0.035579, -0.005156, 0.840245, -0.035069, -0.587439, 50.0, 0.05846]
    
    # 1. Parse Molecule
    if isinstance(molecule, str):
        mol = Chem.MolFromMolBlock(molecule, removeHs=False)
        if mol is None:
            raise ValueError("Failed to parse molecule SDF string.")
    else:
        mol = molecule
        
    # 2. Parse Protein
    # Try sanitizing first; if it fails, parse without strict validation (e.g. for partial receptors)
    rec_mol = Chem.MolFromPDBBlock(protein, removeHs=False)
    if rec_mol is None:
        rec_mol = Chem.MolFromPDBBlock(protein, removeHs=False, sanitize=False)
        if rec_mol is None:
            raise ValueError("Failed to parse protein PDB string.")
            
    # 3. Parameterize Ligand (Molecule) Atoms
    lig_atom_mask = []
    lig_atoms = []
    lig_conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        t = get_xs_type(mol, atom)
        lig_atom_mask.append(t is not None)
        if t is not None:
            pos = lig_conf.GetAtomPosition(idx)
            lig_atoms.append({
                'idx': idx,
                'type': t,
                'pos': (pos.x, pos.y, pos.z)
            })
    lig_atom_mask = np.array(lig_atom_mask)
            
    # 4. Parameterize Protein Atoms
    rec_atoms = []
    rec_conf = rec_mol.GetConformer()
    for atom in rec_mol.GetAtoms():
        idx = atom.GetIdx()
        t = get_xs_type(rec_mol, atom)
        if t is not None:
            pos = rec_conf.GetAtomPosition(idx)
            rec_atoms.append({
                'idx': idx,
                'type': t,
                'pos': (pos.x, pos.y, pos.z)
            })
            
    # 5. Connectivity and Rotatable Bonds of the Ligand
    rot_bonds = get_rotatable_bonds(mol)
    N_rot = len(rot_bonds)
    
    # 6. Intermolecular Energy (Ligand-Receptor)
    # Curing is applied to the sum of interactions of a single ligand atom with all receptor atoms.
    # Matches non_cache::eval in src/lib/non_cache.cpp.
    E_inter = np.zeros(len(lig_atoms), dtype=float)

    for ia, latom in enumerate(lig_atoms):
        sum_e = 0.0
        lx, ly, lz = latom['pos']
        lt = latom['type']
        
        for ratom in rec_atoms:
            rx, ry, rz = ratom['pos']
            rt = ratom['type']
            
            # Distance check
            dist_sq = (lx - rx)**2 + (ly - ry)**2 + (lz - rz)**2
            if dist_sq < 64.0: # 8.0^2 Å
                r = math.sqrt(dist_sq)
                sum_e += calculate_pair_energy(r, lt, rt, weights)
                
        E_inter[ia] = curl(sum_e, v=1000.0)
        
    # 7. Intramolecular Energy (Ligand-Ligand)
    # Curing is applied to each individual pair interaction.
    # Matches model::eval_intramolecular in src/lib/model.cpp (lines 856-879).
    E_intra = np.zeros((len(lig_atoms), len(lig_atoms)), dtype=float)
    topo_dist = Chem.GetDistanceMatrix(mol)
    rigid_comps = get_rigid_components(mol, rot_bonds)
    
    for i in range(len(lig_atoms)):
        la1 = lig_atoms[i]
        idx1 = la1['idx']
        t1 = la1['type']
        x1, y1, z1 = la1['pos']
        
        for j in range(i + 1, len(lig_atoms)):
            la2 = lig_atoms[j]
            idx2 = la2['idx']
            t2 = la2['type']
            x2, y2, z2 = la2['pos']
            
            # Exclude pairs in the same rigid component (distance is fixed)
            if rigid_comps.get(idx1) == rigid_comps.get(idx2):
                continue
                
            # Exclude 1-2, 1-3, 1-4 topological neighbors (bonds path length <= 3)
            if topo_dist[idx1][idx2] <= 3:
                continue
                
            dist_sq = (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2
            if dist_sq < 64.0:
                r = math.sqrt(dist_sq)
                pair_e = calculate_pair_energy(r, t1, t2, weights)
                E_intra[i][j] = curl(pair_e, v=1000.0)
        
    w_rot = weights[6]

    E_inter = E_inter / (1.0 + w_rot * N_rot)
    E_intra = E_intra / (1.0 + w_rot * N_rot)

    # Fill hydrogens
    n_atom_all = mol.GetNumAtoms()
    E_inter_all = np.zeros(n_atom_all, dtype=float)
    E_intra_all = np.zeros((n_atom_all, n_atom_all), dtype=float)
    E_inter_all[lig_atom_mask] = E_inter
    E_intra_all[lig_atom_mask][:,lig_atom_mask] = E_intra

    free_energy = E_inter.sum()

    # Distance penalty
    penalized_all = np.zeros(n_atom_all, dtype=bool)
    
    # 1. Determine which ligand heavy atoms are close to any protein atom (< 4Å)
    is_close = {}
    for latom in lig_atoms:
        idx = latom['idx']
        lx, ly, lz = latom['pos']
        close = False
        for ratom in rec_atoms:
            rx, ry, rz = ratom['pos']
            dist_sq = (lx - rx)**2 + (ly - ry)**2 + (lz - rz)**2
            if dist_sq < 16.0: # 4.0^2
                close = True
                break
        is_close[idx] = close

    # 2. Group ligand heavy atom indices by rigid component ID
    comp_to_atoms = {}
    for latom in lig_atoms:
        idx = latom['idx']
        comp_id = rigid_comps.get(idx)
        if comp_id is not None:
            if comp_id not in comp_to_atoms:
                comp_to_atoms[comp_id] = []
            comp_to_atoms[comp_id].append(idx)
            
    # For each rigid component, check if any of its atoms is close (< 4Å)
    comp_close = {}
    for comp_id, atom_idxs in comp_to_atoms.items():
        comp_close[comp_id] = any(is_close[idx] for idx in atom_idxs)

    # 3. Check penalty conditions for each heavy atom
    for latom in lig_atoms:
        idx = latom['idx']
        
        # If the atom itself is close, it is not penalized
        if is_close[idx]:
            continue
            
        # If any adjacent atom is close, it is not penalized
        nbrs = mol.GetAtomWithIdx(idx).GetNeighbors()
        if any(is_close.get(nbr.GetIdx(), False) for nbr in nbrs):
            continue
            
        # If any atom in the same rigid component is close, it is not penalized
        comp_id = rigid_comps.get(idx)
        if comp_id is not None and comp_close.get(comp_id, False):
            continue
            
        # If we get here, the atom is penalized
        penalized_all[idx] = True

    return free_energy, E_inter_all, E_intra_all, penalized_all