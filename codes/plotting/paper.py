"""
Make plots for the paper
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
Figure 5
"""

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