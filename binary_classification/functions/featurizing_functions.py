import numpy as np


def fingerprint(mol):
    from rdkit.Chem.Fingerprints import FingerprintMols
    return np.ndarray.flatten(np.array(FingerprintMols.FingerprintMol(mol)))


def molecular_fingerprint_padding(x):
    result = np.zeros((2048,))
    result[:x.shape[0]] = x
    return result


def get_ecfp(mol):
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    bitstring = GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).ToBitString()
    return np.array(list(bitstring))
