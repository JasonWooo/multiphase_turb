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
for _, row in tqdm(df.iterrows()):
    # grab the files
    trial = row['trial']
    print('\n\ntrial')
    datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'
    dataf = get_hst(trial = trial)
    rp = get_rp(trial = trial)

    # get the stop time
    stop_time, stop_ind =\
    get_stop_time(trial=trial, rp=rp, pfloor=rp['pfloor'], T_cold=rp['T_cold'], hst_data=dataf,
                  perc_cell_press=0.5, perc_mass_hot=20, perc_mass_cold=5)
    
    # update it in the csv
    row['stop_time'] = stop_time # time in tcc