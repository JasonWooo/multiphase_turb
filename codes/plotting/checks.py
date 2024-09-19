"""

A notebook containing some of the sanity checks to perform on runs

"""

### import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
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
    rp = get_rp(trial)

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

"""
----------------------------------------
Panel plots
----------------------------------------
"""

def temp_rcl_plot(data, tccs,
                  vmin = 800, vmax = 1e5,
                  cmap = 'viridis', lfs = 16, tfs = 18):
    num_runs, temp_data, rcls = data
    # Create the figure and axes
    fig, axes = plt.subplots(1, num_runs, figsize=(2 * num_runs, 2.5), dpi=300)
    
    # normalizations
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

    # define titles for the columns and rows
    tcc_titles = [fr'{_}$\ t_{{\rm cc}}$' for _ in tccs]
    # R_cl values in pc
    # rcl_titles = rcls
    rcl_titles = [20, 2000, 200000, 200000000]

    # plot each panel
    for i in range(num_runs):
        ax = axes[i]
        ax.set_xticks([])
        ax.set_yticks([])

        # make the plot
        data = temp_data[i]
        img = ax.imshow(data, cmap=cmap, norm=norm)
        
        # label the rcls
        coeff, expo = s_n(rcl_titles[i])
        ax.text(len(data)/2, len(data), fr'${coeff:.0f}\times 10^{{{expo:.0f}}}\ l_{{\rm shatter}}$', ha='center', va='bottom', fontsize=lfs)    

        # label the times
        ax.set_xlabel(tcc_titles[i], fontsize=lfs)

    # fig.supylabel(r'$R_{\rm cl} / l_{\rm shatter}$', x=0.08, fontsize=lfs)

    # colorbar
    cax = axes[-1].inset_axes([1.1, -0.2, 0.08, 1.4])
    cbar = fig.colorbar(img, cax=cax, orientation='vertical', location='right', pad=0.1, shrink=0.8, aspect=30, extend='max')
    cbar.set_label('Temperature [K]', fontsize=lfs)
    cax.axhline(1600, lw=0.5, ls='--', color='k', alpha=0.5)
    # cax.text(-1, 1700, r'$1600\ \rm{K}$', color='k', ha='center', va='center', alpha=0.5, rotation=90)
    cax.text(-0.5, 1200, 'cold', color='blue', ha='center', va='center', alpha=0.5, rotation=90, fontsize=lfs/2)
    cax.text(-0.5, 2500, 'warm', color='purple', ha='center', va='center', alpha=0.5, rotation=90, fontsize=lfs/2)

    # add the main title
    fig.suptitle(r'$T_{\rm cloud} = 8\times 10^2\ {\rm K},\ L_{\rm box}/R_{\rm cl} = 50$', va='top', fontsize=tfs, x=0.53, y=1.05)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.show()




"""
----------------------------------------
Scalar related plots
----------------------------------------
"""


"""
Makes temperature evolution for scalar in two brackets
"""

def tracer_temp_evol_load(trial='240711_0.4_16000', scalar_thres = 0.1, nbins_temp = 'auto'):
    
    """read parameters"""
    datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'

    """construct a list of files"""
    file_list = np.sort([f'{datapath}/cloud/{f}' for f in os.listdir(f'{datapath}/cloud') if f.startswith('Turb.out2') and f.endswith('.athdf')])

    # arrays to append to
    time_athdf = []  # time array for the athdfs
    temp_athdf = []  # temperature array for the athdfs
    scalar_athdf = []

    for fname in tqdm(file_list):
        t, press, rho, r = get_datamds(fname=fname, keys=['Time', 'press', 'rho', 'r0'], verbose=False)
        scalar = np.array(r) * np.array(rho)  # scalar is scaled by density
        temperature = calc_T(press, rho)  # calculate temperature from the two
        
        # cells with less and more scalar
        time_athdf.append(t)
        temp_athdf.append(temperature)
        scalar_athdf.append(scalar)

    """process the athdfs into image"""
    temp_large = []  # temperature for cells with large scalar values (CLOUD)
    temp_small = []  # temperature for cells with small scalar values (HOT)

    for temperature, scalar in tqdm(zip(temp_athdf, scalar_athdf)):
        temp_large.append(np.array(temperature[scalar > scalar_thres]).flatten())
        temp_small.append(np.array(temperature[scalar <= scalar_thres]).flatten())

    # temperature bins
    if nbins_temp == 'auto':
        actual_nbins_temp = len(time_athdf)
    else:
        actual_nbins_temp = nbins_temp
    bins_temp = np.power(10., np.linspace(np.log10(8e2), np.log10(1e6), actual_nbins_temp))
    temps = [temp_large, temp_small]  # large and small

    imgdata = [[], []]
    for i, temp_switch in enumerate(temps):  # for both larger and smaller
        for temp in temp_switch:  # for each snapshot
            hist, _ = np.histogram(temp, bins=bins_temp)
            imgdata[i].append(hist)
        
    return time_athdf, temp_athdf, scalar_athdf, temp_large, temp_small, imgdata


def tracer_temp_evol_plot(data = None, trial = '240711_0.4_16000',
                          cmaps = ['Blues', 'Reds'], second_axis_snapshots = False):
    # load params
    time_athdf, temp_athdf, scalar_athdf, temp_large, temp_small, imgdata = data
    rp = get_rp(trial=trial)
    # time in units of tcc
    bins_time = time_athdf/rp['t_cc']

    from matplotlib.ticker import FixedLocator
    from matplotlib.colors import LogNorm

    # make the plot
    fig, ax1 = plt.subplots(figsize=(3, 3), dpi=200)
    imgs = []

    # imgdata = [imgdata[0], [[0]]]

    for cmap, xyval in zip(cmaps, imgdata):
        img = \
        ax1.imshow(xyval,
                    cmap=cmap, alpha=0.8,
                    norm=LogNorm(vmin=1, vmax=1e5),
                    aspect="auto",
                    origin='lower',
                    extent=[bins_time.min(), bins_time.max(),
                            np.log10(8e2), np.log10(1e6)])
        imgs.append(img)

    # colorbars
    cbar_ax1 = fig.add_axes([0.925, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar1 = plt.colorbar(imgs[0], cax=cbar_ax1)
    cbar1.set_ticklabels([])
    cbar_ax2 = fig.add_axes([0.945, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar2 = plt.colorbar(imgs[1], cax=cbar_ax2)

    ax1.set_ylabel(r'$\log_{10}(T [\rm{K}])$')

    # set major ticks for tcc
    t_cc_labels = np.arange(bins_time.min(), bins_time.max(), 1)  # every 1 tccs
    t_cc_ticks = t_cc_labels
    ax1.set_xticks(t_cc_ticks)
    ax1.set_xticklabels([f'{x:.0f}' for x in t_cc_labels])
    ax1.set_xlabel(r"Time [$t_{\rm cc}$]")
    # add minor ticks without labels between 0 and 1
    minor_locator = FixedLocator(np.arange(0, 2, 0.2))
    ax1.xaxis.set_minor_locator(minor_locator)

    """Make a secondary axis on snapshots"""
    if second_axis_snapshots:
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        # set major ticks for snapshots
        t_cc_labels = np.linspace(0, ax2.get_xlim()[1] * rp['t_cc'] / rp['dt_hdf5'] // 10 * 10, 5)
        t_cc_ticks = t_cc_labels / rp['t_cc'] * rp['dt_hdf5']
        ax2.set_xticks(t_cc_ticks)
        ax2.set_xticklabels([f'{x:.0f}' for x in t_cc_labels])
        ax2.set_xlabel(r"Time [snapshots]")

    ax1.legend()
    # plt.savefig(figpath, format="pdf", bbox_inches="tight")
    plt.show()




"""
Makes average scalar value evolution for temperature distributions
"""


def tracer_avg_evol_load(trial = '240711_0.4_16000', nbins_temp = 'auto', ncores = [1, 1]):
    from scipy.stats import binned_statistic as bs
    
    """read parameters"""
    datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'

    """construct a list of files"""
    file_list = np.sort([f'{datapath}/cloud/{f}' for f in os.listdir(f'{datapath}/cloud') if f.startswith('Turb.out2') and f.endswith('.athdf')])

    # split with cores
    this_core, all_core = ncores
    file_list_core = file_list[int((this_core-1) / all_core * len(file_list)) :\
                               int(this_core / all_core * len(file_list))]

    # arrays to append to
    time_athdf = []  # time array for the athdfs
    temp_athdf = []  # temperature array for the athdfs
    scalar_athdf = []
    p_rho_athdf = []

    for fname in tqdm(file_list_core):
        print(fname)
        t, press, rho, r = get_datamds(fname=fname, keys=['Time', 'press', 'rho', 'r0'], verbose=False)
        scalar = np.array(r) * np.array(rho)# * (np.array(rho)) ** 2  # scalar is scaled by density
        temperature = calc_T(press, rho)  # calculate temperature from the two
        
        # cells with less and more scalar
        time_athdf.append(t)
        p_rho_athdf.append([press, rho])
        temp_athdf.append(temperature.flatten())
        scalar_athdf.append(scalar.flatten())

    """process the athdfs into image"""
    # temperature bins
    if nbins_temp == 'auto':
        actual_nbins_temp = len(time_athdf)
    else:
        actual_nbins_temp = nbins_temp
    bins_temp = np.power(10., np.linspace(np.log10(8e2), np.log10(1e6), actual_nbins_temp))

    imgdata = []
    for i, temperature in tqdm(enumerate(temp_athdf)):  # for each snapshot
        # avg_scalars, bin_temp_edges, _ = bs(temperature, scalar_athdf[i], statistic=np.sum, bins=bins_temp)  # mean
        avg_scalars, bin_temp_edges = np.histogram(temperature, weights=scalar_athdf[i], bins=bins_temp)  # mean
        # add values outside
        avg_scalars[0] += np.sum(scalar_athdf[i][temperature < bins_temp[0]])
        avg_scalars[-1] += np.sum(scalar_athdf[i][temperature >= bins_temp[-1]])
        imgdata.append(avg_scalars)

    # return the data
    # data = [time_athdf, p_rho_athdf, temp_athdf, scalar_athdf, imgdata, bin_temp_edges]
    
    # save the data
    save_data = [time_athdf, [list(a) for a in imgdata], bin_temp_edges.tolist()]

    # modify the save
    if this_core == all_core and all_core == 1:
        fsavename = f'/freya/ptmp/mpa/wuze/multiphase_turb/saves/tracer_avg_evol_{trial}.json'
    else:
        fsavename = f'/freya/ptmp/mpa/wuze/multiphase_turb/saves/tracer_avg_evol_{trial}_{this_core}.json'

    import json
    with open(fsavename, 'w') as f:
        json.dump(save_data, f, indent=4)
    print('Data written:')
    print(fsavename)
    
    # return data


def tracer_avg_evol_plot(data = None, trial = '240711_0.4_16000', tcc_lim = 3, vmin = 1e-8,
                         cmap = '', second_axis_snapshots = False, lfs = 12):
    # load params
    time_athdf, imgdata, bin_temp_edges = data
    # normalize and conserve
    imgdata = np.array(imgdata) / imgdata[0][0]
    imgdata[imgdata == 0.] = 1e-10

    rp = get_rp(trial=trial)
    # time in units of tcc
    bins_time = np.array(time_athdf)/float(rp['t_cc'])
    # temperature flipped
    bin_temp_edges = bin_temp_edges[:-1]
    bin_temp_edges = bin_temp_edges[:len(imgdata.T)]

    from matplotlib.ticker import FixedLocator
    from matplotlib.colors import LogNorm, Normalize

    # make the plot
    fig, ax1 = plt.subplots(figsize=(3, 3), dpi=200)
    
    # take the first however many
    tcc_lim_snapshots = int(rp['t_cc'] * tcc_lim / rp['dt_hdf5']) if tcc_lim else -1

    img = \
    ax1.pcolormesh(bins_time[:tcc_lim_snapshots], bin_temp_edges, imgdata[:tcc_lim_snapshots].T,
                   cmap=cmap, alpha=0.8,
                   norm=LogNorm(vmin=vmin, vmax=1),  #Normalize
                   shading='auto')


    # colorbars
    cbar_ax = fig.add_axes([0.925, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(img, cax=cbar_ax)
    cbar.set_label('normalized scalar sum', fontsize=lfs)

    # ax1.set_ylabel(r'$\log_{10}(T [\rm{K}])$')
    ax1.set_ylabel(r'$T$ [K]', fontsize=lfs)

    ax1.set_yscale('log')

    # set major ticks for tcc
    t_cc_labels = np.arange(bins_time.min(), bins_time.max(), 1)  # every 1 tccs
    t_cc_ticks = t_cc_labels
    ax1.set_xticks(t_cc_ticks)
    ax1.set_xticklabels([f'{x:.0f}' for x in t_cc_labels])
    ax1.set_xlabel(r"time [$t_{\rm cc}$]", fontsize=lfs)
    # add minor ticks without labels between 0 and 1
    minor_locator = FixedLocator(np.arange(0, 2, 0.2))
    ax1.xaxis.set_minor_locator(minor_locator)

    """Make a secondary axis on snapshots"""
    if second_axis_snapshots:
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        # set major ticks for snapshots
        t_cc_labels = np.linspace(0, ax2.get_xlim()[1] * rp['t_cc'] / rp['dt_hdf5'] // 10 * 10, 5)
        t_cc_ticks = t_cc_labels / rp['t_cc'] * rp['dt_hdf5']
        ax2.set_xticks(t_cc_ticks)
        ax2.set_xticklabels([f'{x:.0f}' for x in t_cc_labels])
        ax2.set_xlabel(r"time [snapshots]", fontsize=lfs)

    ax1.legend()

    # plt.savefig(figpath, format="pdf", bbox_inches="tight")
    plt.show()