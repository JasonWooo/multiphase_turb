"""

The growth rate plots correspond to the parameter_func_8e2.ipynb notebook

The growth fraction plots correspond to visualize.ipynb notebook

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
Evolution
"""

def plot_evol_both_trial(csvpath = '/freya/ptmp/mpa/wuze/multiphase_turb/data/saves/cloud_8e2_new.csv', trial = None):
    """
    Plot the evolution of COLD and WARM gas for a particular trial
    -----
    mach: only plot the points for a certain mach number
    """
    from matplotlib.collections import LineCollection
    import pandas as pd
    import pickle
    df = pd.read_csv(csvpath, comment='#')
    df['yval'] = df["r_cl"] / df["l_shat"]
    df.sort_values(by='yval', inplace=True)
    xys, cs = [], []
    fig, ax = plt.subplots(figsize=(5,1))

    x_lims = []
    
    for _, row in df.iterrows():
        if trial != row['trial']:  # check for this particular trial
            continue
        datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'
    
        """Load run parameters"""
        with open(f'{datapath}/params.pickle', 'rb') as handle:
            rp = pickle.load(handle)
        """Load the hst file"""
        fname = f'{datapath}/cloud/Turb.hst'
        with open(fname, 'r') as file: keys_raw = file.readlines()[1]
        keys = [a.split('=')[1] for a in keys_raw.split()[1:]]
        fname = f'{datapath}/cloud/Turb.hst'
        data = np.loadtxt(fname).T
        hst_data = {keys[i]: data[i] for i in range(len(keys))}

        """Retrieve mass fractions"""
        stop_time = row['stop_time']
        stop_ind = int(np.ceil(stop_time * rp['t_cc'] / (rp['dt_hdf5'] / 100)))  # find the index to stop
        x = hst_data['time'] / rp['t_eddy']  # in units of t_eddy
        x = x[:stop_ind]
        x_lims.append(stop_time)  # append the limit

        # gas masses
        cg = hst_data['cold_gas']
        wg = hst_data['warm_gas']
        # get cold and warm gas mass
        y_cg = cg[:stop_ind] / cg[0]
        y_wg = wg[:stop_ind] / cg[0]

        ax.plot(x, y_cg, lw=1, ls='-', color='blue', alpha=0.5, label='cold')
        ax.plot(x, y_wg, lw=1, ls='-', color='red', alpha=0.5, label='warm')

        break
        
    # y axis
    ax.set_ylim(-0.1, 1)
    ax.set_ylabel(r'$M_{\rm phase} / M_{\rm cold, ini}$')
    
    # x axis
    # plot to 1/3 of the box mass, whichever trial has the minimum
    ax.set_xlim(0, np.min(x_lims))
    ax.set_xlabel(r"Time $t / t_{\rm eddy}$")
    
    # ax.text(0.5, 2e0, fr'$T_{{\rm cold}} = {T_cold:.0f}$', ha='center')
    # ax.legend(loc='lower right', bbox_to_anchor=(1.8,0.1), fontsize=10, alignment='center')
    plt.title(fr'$R/l_{{\rm shatter}} = {row["yval"]:.0f}$')

    plt.show()




"""
Growth rates
"""

def plot_evol_all_both(csvpath = '/freya/ptmp/mpa/wuze/multiphase_turb/data/saves/cloud_8e3.csv', mach = 0.3, plot_wg = True):
    """
    Plots evolution of WARM & COLD in separate panels
    Plots ALL runs for a single Mach number
    -----
    mach: only plot the points for a certain mach number
    """
    from matplotlib.collections import LineCollection
    import pandas as pd
    import pickle
    df = pd.read_csv(csvpath, comment='#')
    df['yval'] = df["r_cl"] / df["l_shat"]
    df.sort_values(by='yval', inplace=True)
    xys, cs = [], []
    fig, ax1 = plt.subplots(figsize=(5,3))
    if plot_wg:
        fig, ax2 = plt.subplots(figsize=(5,3))
    cm = cmr.get_sub_cmap('cmr.iceburn', .1, .9)  # colormap for the gas fractions

    x_lims = []
    
    for _, row in df.iterrows():
        if f'{mach:.1f}' not in row['trial']:  # check for mach number
            continue
        datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{row['trial']}'
    
        """Load run parameters"""
        with open(f'{datapath}/params.pickle', 'rb') as handle:
            rp = pickle.load(handle)
        """Load the hst file"""
        fname = f'{datapath}/cloud/Turb.hst'
        with open(fname, 'r') as file: keys_raw = file.readlines()[1]
        keys = [a.split('=')[1] for a in keys_raw.split()[1:]]
        fname = f'{datapath}/cloud/Turb.hst'
        data = np.loadtxt(fname).T
        hst_data = {keys[i]: data[i] for i in range(len(keys))}

        """Retrieve mass fractions"""
        stop_time = row['stop_time']
        stop_ind = int(np.ceil(stop_time * rp['t_cc'] / (rp['dt_hdf5'] / 100)))  # find the index to stop
        x = hst_data['time'] / rp['t_eddy']  # in units of t_eddy
        x = x[:stop_ind]
        x_lims.append(stop_time)  # append the limit

        # gas masses
        cg = hst_data['cold_gas']
        wg = hst_data['warm_gas']
        # get cold and warm gas mass
        y_cg = cg[:stop_ind] / cg[0]
        y_wg = wg[:stop_ind] / cg[0]
        print(y_cg)
        color = cm(plt.Normalize(1, 7)(np.log10(row["yval"]))) # color of the line

        ax1.plot(x, y_cg, lw=1, ls='-', color=color, alpha=0.5, label=fr'$R/l_{{\rm shatter}} = {row["yval"]:.0f}$')
        if plot_wg:
            ax2.plot(x, y_wg, lw=1, ls='-', color=color, alpha=0.5, label=fr'$R/l_{{\rm shatter}} = {row["yval"]:.0f}$')
    
    # y axis
    ax1.set_ylim(1/3, 3)
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$M_{\rm cold} / M_{\rm cold, ini}$')
    
    # x axis
    # plot to 1/3 of the box mass, whichever trial has the minimum
    ax1.set_xlim(0, np.min(x_lims))
    ax1.set_xlabel(r"Time $t / t_{\rm eddy}$")
    
    # ax1.text(0.5, 2e0, fr'$T_{{\rm cold}} = {T_cold:.0f}$', ha='center')
    # ax1.legend(loc='lower right', bbox_to_anchor=(1.8,0.1), fontsize=10, alignment='center')
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 3))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.ax.set_ylabel(r'$\log_{10} \frac{M_{\rm cold}}{M_{\rm cold, 0}}$', rotation=90, labelpad=10)
    # plt.title(fr'${{\mathcal{{M}} = {mach}}}$, resolution: $128^3$')

    if plot_wg:
        ax2.set_ylim(1e-3, 1e1)
        ax2.set_yscale('log')
        ax2.set_ylabel(r'$M_{\rm warm} / M_{\rm cold, ini}$')
        ax2.set_xlim(0, np.min(x_lims))
        ax2.set_xlabel(r"Time $t / t_{\rm eddy}$")
        # ax2.legend(loc='lower right', bbox_to_anchor=(1.3,0.1), fontsize=10, alignment='center')
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 3))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2)
        cbar.ax.set_ylabel(r'$\log_{10} \frac{M_{\rm cold}}{M_{\rm cold, 0}}$', rotation=90, labelpad=10)
        # plt.title(fr'${{\mathcal{{M}} = {mach}}}$, resolution: $128^3$')

    plt.show()


def plot_evol_all_cold(csvpath = '/freya/ptmp/mpa/wuze/multiphase_turb/data/saves/cloud_8e3.csv', mach = 0.3, cm = None, alpha = 0.5, verbose = False, plot_growth = False):
    """
    Plots evolution of only COLD
    Plots ALL runs for a single Mach number

    Corresponds to Figure 5 in the paper
    -----
    mach: only plot the points for a certain mach number
    """
    from matplotlib.collections import LineCollection
    import pandas as pd
    df = pd.read_csv(csvpath, comment='#')
    df['yval'] = df["r_cl"] / df["l_shat"]
    df.sort_values(by='yval', inplace=True)
    xys, cs = [], []
    
    fig, ax = plt.subplots(figsize=(5,3))
    
    for _, row in df.iterrows():
        if (row['x_mach'] < mach - 0.05) or (row['x_mach'] > mach + 0.05):  # if not within range
            continue
        datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{row['trial']}'
        
        """Load history file"""
        # try
        try:
            fname = f'{datapath}/turb/Turb.hst'
            with open(fname, 'r') as file: keys_raw = file.readlines()[1]
            keys = [a.split('=')[1] for a in keys_raw.split()[1:]]
        except:
            fname = f'{datapath}/cloud/Turb.hst'
            with open(fname, 'r') as file: keys_raw = file.readlines()[1]
            keys = [a.split('=')[1] for a in keys_raw.split()[1:]]
        
        fname = f'{datapath}/cloud/Turb.hst'
        data = np.loadtxt(fname).T
        dataf = {keys[i]: data[i] for i in range(len(keys))}
        # cold gas
        cg = dataf['cold_gas']
        cg_st_epoch = (cg != 0).argmax()
        
        log_mass_frac = row['log_cold_mass']#np.log10(cg[-1] / cg[cg_st_epoch])
    
        """Load run parameters"""
        with open(f'{datapath}/params.pickle', 'rb') as handle:
            rp = pickle.load(handle)
    
        x = dataf['time'] / rp['t_eddy']
        y = dataf['cold_gas'] / dataf['cold_gas'][cg_st_epoch]
        color = cm(plt.Normalize(0, 8)(np.log10(row["yval"])))

        # get the normalized cloud size
        coeff, expo = s_n(row["yval"])
        ax.plot(x, y, lw=1, ls='-', color=color, alpha=0.5, label=fr'${coeff:.0f}\times 10^{{{expo:.0f}}}$')  #R/l_{{\rm shatter}} = 


        """Plot all expected growths"""
        if plot_growth:
            if (plot_growth == -1) and (row["yval"] < 1e8):
                continue
            # load t_cool_min
            l_shatter_min, t_cool_mix, t_cool_min, [t_cool_func, T_tcoolmin, T_mix] = load_lshat(rp=rp, verbose=verbose)
    
            # growth time
            t_grow = alpha * rp['chi'] *\
            (rp['mach'] ** (-1/2)) *\
            (row['yval'] ** (1/2)) *\
            ((rp['box_size'] / rp['cloud_radius']) ** (1/6)) *\
            t_cool_min
    
            # use actual time, plot eddie time
            print(alpha, rp['chi'], rp['mach'], row['yval'], rp['box_size'] / rp['cloud_radius'])
            print(t_grow, rp['t_eddy'], t_cool_min)
            cold_frac = np.exp(dataf['time'] / t_grow)
            ax.plot(x, cold_frac, lw=1, ls=':', alpha=0.5, color=color)

    # y axis
    ax.set_ylim(1/3, 3)
    # yticks = np.logspace(np.log10(1/5), np.log10(5), 10)
    # ax.set_yticks(yticks)
    # ax.set_yticklabels(yticks)
    ax.set_yscale('log')
    ax.set_ylabel(r'$M_{\rm cold} / M_{\rm cold, ini}$')
    
    # x axis
    ax.set_xlim(0, 2)
    ax.set_xlabel(r"Time $t / t_{\rm eddy}$")
    
    ax.legend(loc='lower right', bbox_to_anchor=(1.17, 0.5), fontsize=5, alignment='left')
    
    # add colorbar
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 8))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.915, 0.10, 0.02, 0.4])  # [left, bottom, width, height]
    cbar = plt.colorbar(sm, cax=cbar_ax, ax=ax)#, extend='both')
    cbar.set_ticks([0, 2, 4, 6, 8])
    cbar.set_ticklabels([0, 2, 4, 6, 8])
    cbar.ax.set_ylabel(r'$\log_{10} \frac{R_{\rm cl}}{l_{\rm shatter}}$', rotation=90, labelpad=10)
    ax.set_title(fr'${{\mathcal{{M}} = {mach}}}$, resolution: ${rp['grid_dim']}^3$')
    plt.show()