from rdkit.Chem import AllChem, DataStructs, rdMHFPFingerprint
from mapchiral import mapchiral
from map4 import MAP4Calculator
import tmap as tm

# Encoders for original MAP4
map2_2048 = MAP4Calculator(radius=1, dimensions=2048)
map4_2048 = MAP4Calculator(radius=2, dimensions=2048)
map6_2048 = MAP4Calculator(radius=3, dimensions=2048)
tm_minhash_2048 = tm.Minhash(d=2048)

# Dictionary of fingerprint functions
fingerprint_functions = {
    'ecfp4': lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048),
    'ecfp4c': lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=True),
    'ecfp6': lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048),
    'ecfp6c': lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048, useChirality=True),
    'ap': lambda mol: AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048),
    'apc': lambda mol: AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048, includeChirality=True),
    'map2': lambda mol: map2_2048.calculate(mol),
    'map2c': lambda mol: mapchiral.encode(mol, 1, n_permutations=2048),
    'map4': lambda mol: map4_2048.calculate(mol),
    'map4c': lambda mol: mapchiral.encode(mol, 2, n_permutations=2048),
    'map6': lambda mol: map6_2048.calculate(mol),
    'map6c': lambda mol: mapchiral.encode(mol, 3, n_permutations=2048)
}

# Dictionary of distance functions (use the same name as for fingerprint)
distance_functions = {
    'ecfp4': lambda query_fingerprint, compound_fingerprint: 1 - DataStructs.DiceSimilarity(query_fingerprint, compound_fingerprint),
    'ecfp4c': lambda query_fingerprint, compound_fingerprint: 1 - DataStructs.DiceSimilarity(query_fingerprint, compound_fingerprint),
    'ecfp6': lambda query_fingerprint, compound_fingerprint: 1 - DataStructs.DiceSimilarity(query_fingerprint, compound_fingerprint),
    'ecfp6c': lambda query_fingerprint, compound_fingerprint: 1 - DataStructs.DiceSimilarity(query_fingerprint, compound_fingerprint),
    'ap': lambda query_fingerprint, compound_fingerprint: 1 - DataStructs.DiceSimilarity(query_fingerprint, compound_fingerprint),
    'apc': lambda query_fingerprint, compound_fingerprint: 1 - DataStructs.DiceSimilarity(query_fingerprint, compound_fingerprint),
    'map2': lambda query_fingerprint, compound_fingerprint: tm_minhash_2048.get_distance(query_fingerprint, compound_fingerprint),
    'map2c': lambda query_fingerprint, compound_fingerprint: 1 - mapchiral.jaccard_similarity(query_fingerprint, compound_fingerprint),
    'map4': lambda query_fingerprint, compound_fingerprint: tm_minhash_2048.get_distance(query_fingerprint, compound_fingerprint),
    'map4c': lambda query_fingerprint, compound_fingerprint: 1 - mapchiral.jaccard_similarity(query_fingerprint, compound_fingerprint),
    'map6': lambda query_fingerprint, compound_fingerprint: tm_minhash_2048.get_distance(query_fingerprint, compound_fingerprint),
    'map6c': lambda query_fingerprint, compound_fingerprint: 1 - mapchiral.jaccard_similarity(query_fingerprint, compound_fingerprint),
}