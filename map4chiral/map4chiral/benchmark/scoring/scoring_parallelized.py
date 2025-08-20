import os
import pandas as pd
from rdkit import Chem
from multiprocessing import Pool

from fingerprints import fingerprint_functions, distance_functions

# Set file directory as working directory 
os.chdir(os.path.dirname(os.path.abspath(__file__)))

fingerprint_names = ['map2c', 'map2', 'map4c', 'map4', 'map6c', 'map6', 'ecfp4c', 'ecfp4', 'ecfp6c', 'ecfp6', 'apc', 'ap']

# Choose for which datasets to compute the benchmark
datasets = ['ChEMBL', 'DUD', 'MUV', 'PeptidesM', 'Peptides',]

# Iterate over all datasets
def scoring(dataset):
    # Set the paths to the dataset and the queries
    dataset_path = f'../compounds/{dataset}'
    queries_path = f'../queries/{dataset}'

    # Iterate over all files in a dataset folder
    for file in os.listdir(dataset_path):

        # Load the dataset and the queries
        file_dataset = pd.read_csv(os.path.join(dataset_path, file))
        file_queries = pd.read_csv(os.path.join(queries_path, f'{file.split(".")[0]}_queries.csv'))
        output_df = file_dataset.copy()

        # Compute the fingerprints for the dataset and the queries
        for fingerprint in fingerprint_names:
            set_fingerprints = [fingerprint_functions[fingerprint](Chem.MolFromSmiles(smiles)) for smiles in file_dataset['smiles']]
            query_fingerprints = [fingerprint_functions[fingerprint](Chem.MolFromSmiles(smiles)) for smiles in file_queries['smiles']]

            # Determine the distance function to use for each fingerprint and compute the distances
            distance_function = distance_functions[fingerprint]
            for idx, query in enumerate(query_fingerprints):
                distances = [distance_function(query, compound) for compound in set_fingerprints]
            
                # Add the distances to the output DataFrame
                output_df[f'{fingerprint}_{idx}'] = distances

        # Save the output DataFrame to a CSV file
        if not os.path.exists(dataset):
            os.mkdir(dataset)
    
        filename = file.split('.')[0]
        output_df.to_csv(f'{dataset}/{filename}_scored.csv', index=False)

# Parallelize the scoring
pool = Pool(processes=14)
pool.map(scoring, datasets)