"""

The growth rate plots correspond to the parameter_func_8e2.ipynb notebook

The growth fraction plots correspond to visualize.ipynb notebook

"""

### import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from codes.funcs import *  # import everything in functions
from codes.timescales import *
mpl.rcParams.update(mpl.rcParamsDefault)
from codes.jason import plotting_def, plot_prettier
import cmasher as cmr
plotting_def()


"""
Evolution
"""

def plot_evol_stack(trial = '240613_0.1_10', return_cropped = True):
    """
    Plot the evolution of different phases of gas in the run
    Make a stacked plot
    """
    # grab the rp
    rp = get_rp(trial)
    # grab the hst file data
    datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'
    # try, for different file structures
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

    # mass fractions
    cg_st_epoch = 0
    cold_frac_all = dataf['cold_gas'] / dataf['cold_gas'][cg_st_epoch]  # cold gas fraction
    warm_frac_all = dataf['warm_gas'] / dataf['cold_gas'][cg_st_epoch]  # warm gas fraction

    """Crop the mass at the stop time"""
    if return_cropped:
        stop_time, stop_ind = get_stop_time(trial=trial, rp=rp, pfloor=rp['pfloor'], T_cold=rp['T_cold'], hst_data=dataf)
        # mass fractions, cropped to where pressure does not hit pfloor
    else:
        stop_time = dataf['time'][-1]  # the last time in hst
        stop_ind = len(cold_frac_all)  # the last hst
    cold_frac_cropped = cold_frac_all[:stop_ind]
    warm_frac_cropped = warm_frac_all[:stop_ind]
    stop_time_cc = stop_time / rp['t_cc']
    
    # hot gas mass from total mass
    tot_mass = np.sum(get_datamd(fname=f'{datapath}/cloud/Turb.out2.00001.athdf', key='rho', verbose=False)) * ((rp['box_size'] / rp['grid_dim']) ** 3)
    cg_mass = dataf['cold_gas']
    wg_mass = dataf['warm_gas']

    # crop the mass
    time_cropped = dataf['time'][:stop_ind]
    wg_mass_cropped = wg_mass[:stop_ind]
    cg_mass_cropped = cg_mass[:stop_ind]
    hg_mass_cropped = tot_mass - cg_mass_cropped - wg_mass_cropped

    # stop when the hot gas mass reach 10%
    if np.any((hg_mass_cropped / tot_mass) < 0.2):  # if the hot gas disappeears
        print('Cropped for hot mass')
        hot_fill_ind = np.argmin(np.abs(hg_mass_cropped / tot_mass - 0.05))
    else:
        hot_fill_ind = -1
    stop_time_cc = time_cropped[hot_fill_ind] / rp['t_cc']

    """Make the plot"""
    fig, ax1 = plt.subplots(figsize=(5,3))
    # plot a stack plot
    ax1.stackplot(time_cropped[:hot_fill_ind] / rp['dt_hdf5'],  # time
                    wg_mass_cropped[:hot_fill_ind] / tot_mass,  # c, w, and h
                    cg_mass_cropped[:hot_fill_ind] / tot_mass,
                    hg_mass_cropped[:hot_fill_ind] / tot_mass,
                    labels=['warm', 'cold', 'hot'],
                    colors=['orange', 'blue', 'red'])
    # ax1.set_xlim(0, 2.5 * rp['t_cc'] / rp['dt_hdf5'])
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Gas mass [ini cold mass]')
    ax1.set_xlabel('Time [epochs]')
    
    # alternative t_cc axis
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    t_cc_ticks = np.linspace(0, ax1.get_xlim()[1], 5)
    t_cc_labels = (t_cc_ticks * rp['dt_hdf5']) / rp['t_cc']
    ax2.set_xticks(t_cc_ticks)
    ax2.set_xticklabels(f'{x:.2f}' for x in t_cc_labels)
    ax2.set_xlabel(r"Time [$t_{\rm cc}$]")
    
    # ax1.legend()
    plt.show()
        
    return np.log10(cg_mass_cropped[:hot_fill_ind] / cg_mass_cropped[cg_st_epoch]), np.log10(wg_mass_cropped[:hot_fill_ind] / cg_mass_cropped[cg_st_epoch]), stop_time_cc



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
    # ax.set_ylim(-0.1, 2)
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

def plot_evol_all_both(csvpath = '/freya/ptmp/mpa/wuze/multiphase_turb/data/saves/cloud_8e3.csv', mach = 0.3, plot_wg = True, xlim_min = True):
    """
    Plots evolution of WARM & COLD in separate panels
    Plots ALL runs for a single Mach number
    -----
    mach: only plot the points for a certain mach number
    xlim_min: limit x axis (time) to the minimum of all simulations
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
    cm = cmr.get_sub_cmap('cmr.guppy', 0, 1)#cmr.get_sub_cmap('cmr.iceburn', .1, .9)  # colormap for the gas fractions

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
    ax1.set_xlim(0, np.min(x_lims) if xlim_min else np.max(x_lims))
    ax1.set_xlabel(r"Time $t / t_{\rm eddy}$")
    
    # ax1.text(0.5, 2e0, fr'$T_{{\rm cold}} = {T_cold:.0f}$', ha='center')
    # ax1.legend(loc='lower right', bbox_to_anchor=(1.8,0.1), fontsize=10, alignment='center')
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 8))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.ax.set_ylabel(r'$\log_{10} \frac{R_{\rm cl}}{l_{\rm shatter}}$', rotation=90, labelpad=10)
    # plt.title(fr'${{\mathcal{{M}} = {mach}}}$, resolution: $128^3$')

    if plot_wg:
        ax2.set_ylim(1e-3, 1e1)
        ax2.set_yscale('log')
        ax2.set_ylabel(r'$M_{\rm warm} / M_{\rm cold, ini}$')
        ax2.set_xlim(0, np.min(x_lims) if xlim_min else np.max(x_lims))
        ax2.set_xlabel(r"Time $t / t_{\rm eddy}$")
        # ax2.legend(loc='lower right', bbox_to_anchor=(1.3,0.1), fontsize=10, alignment='center')
        sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 3))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2)
        cbar.ax.set_ylabel(r'$\log_{10} \frac{M_{\rm cold}}{M_{\rm cold, 0}}$', rotation=90, labelpad=10)
        # plt.title(fr'${{\mathcal{{M}} = {mach}}}$, resolution: $128^3$')

    plt.show()


def plot_evol_all_cold(csvpath = '/freya/ptmp/mpa/wuze/multiphase_turb/data/saves/cloud_8e3.csv', mach = 0.3, cm = None, alpha = 0.5,
                       verbose = False, plot_growth = False, growth_temps = [8e3, 8e2]):
    """
    Plots evolution of only COLD
    Plots ALL runs for a single Mach number

    Corresponds to Figure 5 in the paper
    -----
    mach: only plot the points for a certain mach number
    plot_growth: whether or not the expected growth is plotted
        growth_temps: provided for t_grow calculations
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
        rp = get_rp(trial=row['trial'])
    
        x = dataf['time'] / rp['t_eddy']
        y = dataf['cold_gas'] / dataf['cold_gas'][cg_st_epoch]
        color = cm(plt.Normalize(0, 8)(np.log10(row["yval"])))

        # get the normalized cloud size
        coeff, expo = s_n(row["yval"])
        ax.plot(x, y, lw=1, ls='-', color=color, alpha=1, label=fr'${coeff:.0f}\times 10^{{{expo:.0f}}}$')  #R/l_{{\rm shatter}} = 


        """Plot all expected growths"""
        if plot_growth:
            # filter out all the runs with small radii
            if (plot_growth == -1) and (row["yval"] < 1e4):
                continue
            # load t_cool_min
            l_shatter_min, t_cool_mix, t_cool_min, [t_cool_func, T_tcoolmin, T_mix] = load_lshat(rp=rp, verbose=verbose)

            [T_high, T_low] = growth_temps
            T_peak = 8.57e+03
            chi_growth = T_high / T_low
    
            # growth time
            t_grow = alpha * chi_growth *\
            (rp['mach'] ** (-1/2)) *\
            (row['yval'] ** (1/2)) *\
            ((rp['box_size'] / rp['cloud_radius']) ** (1/6)) *\
            t_cool_func(T_peak)  # the t_cool for the lower temperature
    
            # use actual time, plot eddie time
            print(t_grow, rp['t_eddy'], t_cool_min)
            cold_frac = np.exp(dataf['time'] / t_grow)
            ax.plot(x, cold_frac, lw=1, ls=':', alpha=0.5, color=color)

    # y axis
    ax.set_ylim(1/3, 10)
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
    # plt.show()
    return ax, x, dataf['time']



def plot_evol_all_ratio(csvpath = '/freya/ptmp/mpa/wuze/multiphase_turb/data/saves/cloud_8e3.csv', mach = 0.3, cold_cloud_ratio = 2, plot_wg = True):
    """
    Plots evolution of only COLD
    Plots ALL runs for a single Mach number
    Checks that all 

    -----
    mach: only plot the points for a certain mach number
    cold_cloud_ratio: the ratio between cloud temperature and cold gas temperature criteria
        Choose within cold_cloud_ratio_list = [2, 3, 5, 10]
    """
    from matplotlib.collections import LineCollection
    import pandas as pd
    import pickle
    df = pd.read_csv(csvpath, comment='#')
    df['yval'] = df["r_cl"] / df["l_shat"]
    df.sort_values(by='yval', inplace=True)
    xys, cs = [], []
    fig, ax1 = plt.subplots(figsize=(5,3))
    fig, ax2 = plt.subplots(figsize=(5,3))
    cm = plt.colormaps['BrBG_r']  # colormap for the gas fractions

    x_lims = []
    for _, row in df.iterrows():
        if row['x_mach'] != mach:
            continue
        datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{row['trial']}'
        
        """Load history and temperatures"""
        with open(f'{datapath}/cloud/time_temperature', 'rb') as handle:
            time_athdf, temperature_athdf, mass_athdf = pickle.load(handle)
    
        """Load run parameters"""
        with open(f'{datapath}/params.pickle', 'rb') as handle:
            rp = pickle.load(handle)

        """Gas masses"""
        # cold gas mass
        T_cold = rp['T_cloud'] * cold_cloud_ratio
        cg = []  # cold gas mass array
        # warm gas mass
        T_warm = rp['T_warm']
        wg = []
        
        for temp_arr, mass_arr in zip(temperature_athdf, mass_athdf):
            cg_epoch = np.sum(mass_arr[temp_arr <= T_cold])
            cg.append(cg_epoch)  # append the cold gas mass for one time save
            if plot_wg: wg.append(np.sum(mass_arr[np.logical_and(temp_arr > T_cold, temp_arr <= T_warm)]))

            # check for 1/3 box size
            if cg_epoch > np.sum(mass_arr) / 3:
                break
                
        cg = np.array(cg); wg = np.array(wg)
        cg_st_epoch = (cg != 0).argmax()
        # calculate mass fractional manually
        log_mass_frac = np.log10(cg[-1] / cg[cg_st_epoch])

        # mass evolution
        x = time_athdf / rp['t_eddy']  # in units of t_eddy
        y_cg = cg / cg[cg_st_epoch]
        y_wg = wg / cg[cg_st_epoch]
        color = cm(plt.Normalize(0, 3)(np.log10(row["yval"])))

        # limit to 1/3 box mass
        x = x[:len(y_cg)]
        x_lims.append(np.max(x))
        ax1.plot(x, y_cg, lw=1, ls='-', color=color, alpha=0.5, label=fr'$R/l_{{\rm shatter}} = {row["yval"]:.0f}$')
        if plot_wg: ax2.plot(x, y_wg, lw=1, ls='-', color=color, alpha=0.5, label=fr'$R/l_{{\rm shatter}} = {row["yval"]:.0f}$')
    
    # y axis
    ax1.set_ylim(1/3, 3)
    ax1.set_yscale('log')
    ax1.set_ylabel(r'$M_{\rm cold} / M_{\rm cold, ini}$')
    
    # x axis
    # plot to 1/3 of the box mass, whichever trial has the minimum
    ax1.set_xlim(0, np.min(x_lims))
    ax1.set_xlabel(r"Time $t / t_{\rm eddy}$")
    
    ax1.text(0.5, 2e0, fr'$T_{{\rm cold}} = {T_cold:.0f}$', ha='center')
    
    ax1.legend(loc='lower right', bbox_to_anchor=(1.8,0.1), fontsize=10, alignment='center')
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 3))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.ax.set_ylabel(r'$\log_{10} \frac{R_{\rm cl}}{l_{\rm shatter}}$', rotation=90, labelpad=10)
    plt.title(fr'${{\mathcal{{M}} = {mach}}}$, resolution: $128^3$')

    if plot_wg:
        ax2.set_ylim(1e-4, 1)
        ax2.set_yscale('log')
        ax2.set_ylabel(r'$M_{\rm warm} / M_{\rm cold, ini}$')
        ax2.set_xlim(0, np.min(x_lims))
        ax2.set_xlabel(r"Time $t / t_{\rm eddy}$")
        ax2.legend(loc='lower right', bbox_to_anchor=(1.3,0.1), fontsize=10, alignment='center')

    plt.show()


# grab the growth rates of all the runs

def growth_rate_calc(csvpath = '/freya/ptmp/mpa/wuze/multiphase_turb/saves/cloud_8e3.csv',
                    mach = 0.4, alpha = 0.5):
    """
    Returns a table of calculated and actual growth rates of all runs in the csv
    -----
    mach: only plot the points for a certain mach number
    """
    import pandas as pd
    df = pd.read_csv(csvpath, comment='#')
    df['yval'] = df["r_cl"] / df["l_shat"]
    df.sort_values(by='yval', inplace=True)
    xys, cs = [], []
    
    # for each box
    for _, row in df.iterrows():
        """Select trials with mach and growth"""
        # mach
        if (row['x_mach'] < mach - 0.05) or (row['x_mach'] > mach + 0.05):  # if not within range
            continue
        # growth
        print(f'cold {row['log_cold_mass']:<10.3f} warm {row['log_warm_mass']:<10.3f}')
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
        
        """Load masses"""
        # gas masses
        cg = dataf['cold_gas']
        wg = dataf['warm_gas']
        cg_st_epoch = (cg != 0).argmax()
    
        rp = get_rp(trial=row['trial'])
        x = dataf['time'] / rp['t_cc']

        cg_norm = cg / cg[cg_st_epoch]
        wg_norm = wg / cg[cg_st_epoch]
        # smoothen the lines
        from scipy.ndimage.filters import gaussian_filter1d
        cg_smoothed = gaussian_filter1d(cg_norm, sigma=10)
        wg_smoothed = gaussian_filter1d(wg_norm, sigma=10)

        # the x and y values
        time = x[-100:]
        m_cold = cg_smoothed[-100:]
        m_warm = wg_smoothed[-100:]
        y_cold = np.log(m_cold / m_cold[0])
        y_warm = np.log(m_warm / m_warm[0])
        
        """Run growths"""
        tgrow_cold_run = 1 / np.polyfit(time, y_cold, deg=1, full=False)[0]  # slope of the fit
        tgrow_warm_run = 1 / np.polyfit(time, y_warm, deg=1, full=False)[0]  # slope of the fit


        """Calculated growths"""
        # load t_cool_min
        l_shatter_min, t_cool_mix, t_cool_min, [t_cool_func, T_tcoolmin, T_mix] = load_lshat(rp=rp, verbose=False)

        # growth time for cold
        tgrow_cold_calc = alpha * rp['chi'] *\
        (rp['mach'] ** (-1/2)) *\
        (row['yval'] ** (1/2)) *\
        ((rp['box_size'] / rp['cloud_radius']) ** (1/6)) *\
        t_cool_min

        # for warm
        tgrow_warm_calc = alpha * (rp['chi'] / 10) *\
        (rp['mach'] ** (-1/2)) *\
        (row['yval'] ** (1/2)) *\
        ((rp['box_size'] / rp['cloud_radius']) ** (1/6)) *\
        t_cool_min

        print(row['trial'])
        print(f"{'cold, run:':<20} {tgrow_cold_run:<20.3f} {'cold, calc:':<20} {tgrow_cold_calc:<10.3f}")
        print(f"{'warm, run:':<20} {tgrow_warm_run:<20.3f} {'warm, calc:':<20} {tgrow_warm_calc:<10.3f}\n")  #{' ' * 20} 