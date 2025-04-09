from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Fragments


def count_hydroxylic(mol):
    return Fragments.fr_Al_OH(mol)


def count_carboxylic(mol):
    return Fragments.fr_Al_COO(mol)


def count_primary_secondary_amine(mol):
    return Fragments.fr_NH1(mol) + Fragments.fr_NH2(mol)


def count_amide(mol):
    return Fragments.fr_amide(mol)


def count_sulfoxide_bond(mol):
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 16:  # Sulfur atom
            for bond in atom.GetBonds():
                if bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetOtherAtom(atom).GetAtomicNum() == 8:
                    count += 1
    return count


def count_rings(mol):
    ring_info = mol.GetRingInfo()

    fused_rings = 0
    unfused_rings = 0

    for ring in ring_info.AtomRings():
        fused = False
        for a in ring:
            if ring_info.NumAtomRings(a) > 1:
                fused = True
                break
        if fused:
            fused_rings += 1
        else:
            unfused_rings += 1
    return unfused_rings, fused_rings


def count_frag(mol):
    num_hydroxylic = count_hydroxylic(mol)
    num_carboxylic = count_carboxylic(mol)
    num_amine = count_primary_secondary_amine(mol)
    num_amide = count_amide(mol)
    num_sulfoxide = count_sulfoxide_bond(mol)
    num_unfused_rings, num_fused_rings = count_rings(mol)
    return num_hydroxylic, num_carboxylic, num_amine, num_amide, num_sulfoxide, num_unfused_rings, num_fused_rings


def estimate_density(smiles: str) -> float:
    """ Estimate the density of organic materials from the SMILES Girolami method

    Args:
        smiles (str): The SMILES string of organic molecules

    Returns:
        density (float): The density of the organic molecules
    
    """
    mol = Chem.MolFromSmiles(smiles)
    frag = count_frag(mol)
    correction = frag[0] * 0.1  + frag[1] * 0.1  + frag[2] * 0.1  + frag[3] * 0.1  + frag[4] * 0.1  + frag[5] * 0.1  + frag[6] * 0.075 
    if correction > 0.3:
        correction = 0.3
    mol = Chem.AddHs(mol)
    M = sum([a.GetMass() for a in mol.GetAtoms()])
    numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    Vs = 0.0
    for n in numbers:
        if n <= 2:
            Vs += 1
        elif n >= 3 and n <= 10:
            Vs += 2
        elif n >= 11 and n <= 18:
            Vs += 4
        elif n >= 19 and n <= 36:
            Vs += 5
        elif n >= 37 and n <= 54:
            Vs += 7.5
        else:
            Vs += 9
    # print("correction: ", correction)
    # print("M: ", M)
    # print("Vs: ", Vs)
    return (1 + correction) * M / Vs / 5.0
    