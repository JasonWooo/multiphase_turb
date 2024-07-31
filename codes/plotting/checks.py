"""

A notebook containing some of the sanity checks to perform on runs

"""

### import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from codes.funcs import *  # import everything in functions
mpl.rcParams.update(mpl.rcParamsDefault)
from codes.jason import plotting_def, plot_prettier
plotting_def()


"""
----------------------------------------
Check .turb runs
----------------------------------------
"""

def check_vturb_mach(trial=''):
    """
    Checks that the Mach number in the .turb runs are behaving as they should
    """
    datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'
    fname = f'{datapath}/turb/Turb.hst'

    # load run properties
    with open(f'{datapath}/params.pickle', 'rb') as handle:
        rp = pickle.load(handle)

    # load hst file
    with open(fname, 'r') as file: keys_raw = file.readlines()[1]
    keys = [a.split('=')[1] for a in keys_raw.split()[1:]]
    data = np.loadtxt(fname).T
    dataf = {keys[i]: data[i] for i in range(len(keys))}

    time = dataf['time'] / rp['t_eddy']
    e_k = dataf['1-KE'] + dataf['2-KE'] + dataf['3-KE']

    # plot the time evolution
    fig, axs = plt.subplots(1, 2, figsize=(6, 2))

    # get vturb from the run
    v_turb_run = np.sqrt(2 * e_k / dataf['mass'])
    axs[0].plot(time, v_turb_run)
    axs[0].axhline(rp['v_turb'], alpha=0.5, lw=1, ls='--')
    axs[0].set_xlabel('time'); axs[0].set_ylabel('Turbulence Velocity')

    # mach number
    mach_run_hst = v_turb_run / calc_cs(rp['T_hot']) #dataf['c_s_sum'] * rp['grid_dim']**3
    axs[1].plot(time, mach_run_hst)
    axs[1].axhline(rp['mach'], alpha=0.5, lw=1, ls='--')
    axs[1].set_xlabel('time'); axs[1].set_ylabel('Mach')
    axs[1].legend()

    print(np.mean(mach_run_hst[int(len(mach_run_hst)/2):]))

    plt.show()