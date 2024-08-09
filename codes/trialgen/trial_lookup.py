import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

"""
This function reads through all the trial params and saves them as csv file
"""

# read all the params.pickle
# starting from 240606_0.6_5
import os
import pickle
import pandas as pd
from tqdm import tqdm

def load_params_to_dataframe(base_path):
    # dataframe to store all the data
    params_df = pd.DataFrame()

    # walk through the directory structure
    subfolders = [ f.path for f in os.scandir(base_path) if f.is_dir() ]
    
    for trial_subfolder in tqdm(subfolders):
        # get all the files in trial subfolder
        trial_name = trial_subfolder.split('/')[-1]
        files = [f.path for f in os.scandir(trial_subfolder)]
        if np.any(['params.pickle' in f for f in files]):  # if the folder has a pickle
            params_file_path = os.path.join(trial_subfolder, 'params.pickle')
            print(f'Processed {params_file_path}')
            # read the pickle
            try:
                with open(params_file_path, 'rb') as f:
                    params = pickle.load(f)
                    temp_df = pd.DataFrame(params)
                    temp_df['trial_name'] = trial_name
                    # append the DataFrame to params_df
                    params_df = pd.concat([params_df, temp_df], ignore_index=True)
            except (pickle.UnpicklingError, IOError) as e:
                print(f"Could not read {params_file_path}: {e}")
                continue
    return params_df

# path of data directory
base_path = '/ptmp/mpa/wuze/multiphase_turb/data'
params_df = load_params_to_dataframe(base_path)

# process
params_df = params_df.drop_duplicates()
params_df = params_df.sort_values(by='trial')
params_df = params_df.set_index('trial_name')

params_df.to_csv('/ptmp/mpa/wuze/multiphase_turb/saves/trial_params.csv', encoding='utf-8', index=True, header=True)
print(f"Loaded data from {len(params_df)} params.pickle files.")