import numpy as np
import pandas as pd

import pickle
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath('/freya/ptmp/mpa/wuze/multiphase_turb'))
import codes
from codes.funcs import *  # import everything in functions
from codes.timescales import get_stop_time  # grab get_stop_time() function, which has all the criteria

sys.path.append(os.path.abspath('/freya/ptmp/mpa/wuze/athena_pp/vis/python'))
from athena_read import athdf
import h5py


# read the csv
csvpath = "/ptmp/mpa/wuze/multiphase_turb/saves/cloud_8e2_new.csv"
    
df = pd.read_csv(csvpath, comment='#')
df.sort_values(by='trial', inplace=True) # do the processing by date / trial name

# update with each row
for irow, row in tqdm(df.iterrows()):
    # grab the files
    trial = row['trial']
    print(f'\n\ntrial = {trial}')
    datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'
    dataf = get_hst(trial = trial)
    rp = get_rp(trial = trial)

    # grab pfloor
    try:
        pfloor = rp['pfloor']
    except KeyError:
        pfloor = 1e-9
    print(f'pfloor = {pfloor}')

    # get the stop time
    stop_time, stop_ind =\
    get_stop_time(trial=trial, rp=rp, pfloor=pfloor, T_cold=rp['T_cold'], hst_data=dataf,
                  perc_cell_press=0.5, perc_mass_hot=20, perc_mass_cold=5)
    
    # update it in the csv
    df.at[irow, 'stop_time'] = stop_time / rp['t_cc'] # time in tcc

# save the saved dataframe
df.to_csv(csvpath, index=False)