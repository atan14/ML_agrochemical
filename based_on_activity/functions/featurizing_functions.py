import numpy as np


def get_rdk(mol):
    from rdkit.Chem.rdmolops import RDKFingerprint
    bitstring = RDKFingerprint(mol, 1, fpSize=2048).ToBitString()
    return np.array(list(bitstring))


def get_ecfp(mol):
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    bitstring = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).ToBitString()
    return np.array(list(bitstring))


def generate_scaffold_set(smiles_series):
    from deepchem.splits.splitters import generate_scaffold

    scaffolds = [generate_scaffold(i) for i in smiles_series]

    scaffolds_dict = {}
    for ind, scaffold in enumerate(scaffolds):
        if scaffold not in scaffolds_dict:
            scaffolds_dict[scaffold] = [ind]
        else:
            scaffolds_dict[scaffold].append(ind)

    scaffolds_dict = {key: sorted(value) for key, value in scaffolds_dict.items()}
    scaffold_sets = [scaffold_set for (scaffold, scaffold_set) in sorted(
        scaffolds_dict.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)]
    return scaffold_sets
