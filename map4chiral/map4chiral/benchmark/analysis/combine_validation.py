import os
import pandas as pd

# Set file directory as working directory 
os.chdir(os.path.dirname(os.path.abspath(__file__)))

metrics = ['auc_mean', 'ef1_mean', 'ef5_mean', 'bedroc20_mean', 'bedroc100_mean', 'rie20_mean', 'rie100_mean']
datasets = ['ChEMBL', 'DUD', 'MUV', 'PeptidesM', 'Peptides']
fingerprint_names = ['map2c', 'map2', 'map4c', 'map4', 'map6c', 'map6', 'ecfp4c', 'ecfp4', 'ecfp6c', 'ecfp6', 'apc', 'ap']

# Iterate over metrics
for metric in metrics:

    # Initialize output dataframe
    output_df = pd.DataFrame(columns=fingerprint_names)

    # Iterate over datasets
    for dataset in datasets:
        dataset_path = f'../validation/{dataset}'

        # Iterate over files
        for file in os.listdir(dataset_path):
                data = pd.read_csv(os.path.join(dataset_path, file))[metric].values
                output_df.loc[len(output_df)] = data
    
    # Create output directory
    if not os.path.exists('combined_sets'):
        os.makedirs('combined_sets')
    
    # Save output dataframe
    output_df.to_csv(f'combined_sets/{metric}.csv', index=False)