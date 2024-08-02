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
Functions to load bulk quantities in a run
"""

import pickle

# trial = '240606_0.6_5'  # for 4e4 plot
# trial = '240610_0.6_2032'  # for 8e3 plot

def load_lshat(rp = None, verbose = False):
    # Import the cooling function from Hitesh's scripts
    import sys
    sys.path.append(os.path.abspath('/freya/ptmp/mpa/wuze/multiphase_turb/athena/cooling_scripts'))
    import cooling_fn as cf
    sys.path.append(os.path.abspath('/freya/ptmp/mpa/wuze/multiphase_turb/athena/helper_scripts'))
    import v_turb as vt

    # temperature range
    T_arr = np.logspace(np.log10(rp['T_floor']),
                        np.log10(rp['T_ceil']), 100)  # in kelvin
    rho_arr = rp['rho_hot'] * rp['T_hot'] / T_arr
    
    """
    Cooling & Heating functions
    """
    Gamma_n_arr = 1e-26 / rho_arr
    Lambda_arr = np.vectorize(cf.Lam_fn_powerlaw)(T_arr, Zsol=1.0, Lambda_fac=1.0)
    
    """
    Cooling & Heating rates
    """
    heating_rate = 1e-26 * rho_arr
    cooling_rate = Lambda_arr * rho_arr**2
    
    """
    Timescale
    """
    tcool_arr = np.vectorize(cf.tcool_calc)(
        rho_arr, T_arr, Zsol=1.0, Lambda_fac=1.0, fit_type="max"
    )

    """
    Shattering length
    """
    def calc_cs(T):
        m_to_cm = 100
        return np.sqrt(u.g * u.CONST_kB / (u.mu * u.CONST_amu) * T) / u.unit_velocity
    t_cool_func = lambda T : cf.tcool_calc(rp['rho_hot'] * rp['T_hot'] / T, T, Zsol=1.0, Lambda_fac=1.0, fit_type="max")
    l_shatter = np.vectorize(t_cool_func)(T_arr) * calc_cs(T_arr)

    """
    Minimum l_shatter
    """
    # calculate geometric mean T_mix
    # the two limits
    T_cold = rp['T_cloud']  # set cold temperature to that of cloud
    T_hot = rp['T_hot']

    """T_mix calculations"""
    # the params in range
    T_ind_low, T_ind_high = find_ind_l(T_arr, T_cold), find_ind_l(T_arr, T_hot)
    T_range = T_arr[T_ind_low : T_ind_high]
    tcool_range = tcool_arr[T_ind_low : T_ind_high]
    
    T_tcoolmin = T_range[np.argmin(tcool_range)]  # the temperature corresponding to the lowest t_cool
    T_mix = np.sqrt(T_tcoolmin * T_hot)  # use the temperature at which t_cool is lowest instead of this:  # T_mix = np.sqrt(T_cold * T_hot)  # use the cloud temperature

    t_cool_min = np.min(tcool_range)
    t_cool_mix = t_cool_func(T_mix)
    t_cool_cold = t_cool_func(T_cold)
    
    l_shatter_min = np.min(l_shatter[T_ind_low : T_ind_high])

    if verbose:
        print(f'T_cold = {T_cold}')
        print(f'T_hot = {T_hot:.2e}')
        print(f'T_mix = {T_mix:.2e}')
        print(f't_cool,mix,og = {t_cool_func(np.sqrt(T_cold * T_hot)):.2e}')
        print(f't_cool,mix = {t_cool_mix:.2e}')
        print(f't_cool,cold = {t_cool_cold:.2e}')
        print(f't_cool_min = {t_cool_min:.2e}')
        print(f'l_shatter_min = {l_shatter_min:.2e}')
    
    def plot_tvlines():
        plt.axvline(x=rp['T_floor'], lw=1, color="k", linestyle="--")
        plt.axvline(x=rp['T_cloud'], lw=1, color="slateblue", linestyle="--")
        plt.axvline(x=T_tcoolmin, lw=1, color="slateblue", linestyle="-.", alpha=0.2)
        plt.axvline(x=T_mix, lw=1, color="green", linestyle="-.", alpha=0.2)
        plt.axvline(x=rp['T_hot'], lw=1, color="orangered", linestyle="--")
        # plt.axvline(x=T_cut color="k", linestyle="")
    
    if verbose:
        """Cooling function"""
        plt.figure(figsize=(4, 3))
        plt.plot(T_arr, Lambda_arr, label=r"$\Lambda(T)$")
        
        plt.ylabel(r"$\Lambda(T)$  $[10^{-23}  \mathrm{ergs ~ cm}^3/\mathrm{s}]$", fontsize=14)
        plt.xlabel(r"$T$(K)", fontsize=14)
        
        plt.yscale("log")
        plt.xscale("log")
        
        plot_tvlines()
        plt.show()

        """Cooling rate"""
        plt.figure(figsize=(4, 3))
        plt.plot(T_arr, cooling_rate - heating_rate, color='green', label=r"$n^2\Lambda(T) - n\Gamma(T) $")
        plt.plot(T_arr, cooling_rate, color='blue', label=r"$n^2\Lambda(T)$")
        plt.plot(T_arr, heating_rate, color='red', label=r"$n\Gamma(T)$")
        
        plt.ylabel(r"Cooling/Heating rate", fontsize=14)
        plt.xlabel(r"$T$(K)", fontsize=14)
        
        # plt.yscale("log")
        plt.xscale("log")
        
        # plt.ylim(0.0, 5e-15)
        
        plt.legend()
        plot_tvlines()
        plt.show()

        """Cooling timescale"""
        plt.figure(figsize=(4, 3))
        plt.plot(T_arr, tcool_arr, label=r"$\Lambda(T)$")
        plt.ylabel(r"$t_{\rm cool} [u]$", fontsize=14)
        plt.xlabel(r"$T$(K)", fontsize=14)
        plt.yscale("log")
        plt.xscale("log")
        plot_tvlines()
        plt.show()
        
        """l_shatter"""
        plt.figure(figsize=(4, 3))
        # shatter length scale
        plt.plot(T_arr, l_shatter)
        plt.ylabel(r"$l_{\rm shatter}$", fontsize=14)
        plt.xlabel(r"$T$(K)", fontsize=14)
        plt.yscale("log")
        plt.xscale("log")
        plot_tvlines()
        plt.show()

    return l_shatter_min, t_cool_mix, t_cool_min, t_cool_cold  # return the minimum shattering length


def add_point(trial = '240613_0.1_10', verbose = True):
    # load function
    datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'
    rp = get_rp(trial)
    if verbose: print(rp)
    l_shatter_min, t_cool_mix, t_cool_min, t_cool_cold = load_lshat(rp = rp, verbose=verbose)
    vel_frac = rp['cloud_radius'] / l_shatter_min

    if verbose:
        print(f'X-value = {rp['mach']}')
        print(f'Y-value [log] = {np.log10(vel_frac):.3f}')
        print(f'Y-value = {vel_frac:.3f}')
    return rp, rp['mach'], vel_frac, t_cool_mix, t_cool_min, t_cool_cold

def plot_point_bw(trial = '240613_0.1_10', verbose = True):
    rp, x, y, t_cool_mix, t_cool_min, t_cool_cold = add_point(trial=trial, verbose=verbose)
    
    plt.subplots(figsize=(3,3))
    # point
    sc = plt.scatter(x, y, marker='o', vmin=-0.3, vmax=0.3, ec='k', fc='None')
    
    # analytical line
    ana_x = np.linspace(0, 1.2, 100)
    ana_y = ana_x * t_cool_mix / t_cool_min * 10. ** (0.6 * ana_x + 0.02)
    plt.plot(ana_x, ana_y, ls='-.', color='k')
    
    # axis
    plt.xlim(0, 1.2)
    plt.xticks(np.linspace(0, 1.2, 7))
    plt.ylim(np.power(10., np.log10(t_cool_mix / t_cool_min) - 2), np.power(10., np.log10(t_cool_mix / t_cool_min) + 2))
    plt.yscale('log')
    plt.xlabel(r'$\mathcal{M}_{\rm hot,\ turb}$')
    plt.ylabel(r'$\frac{R_{\rm cl}}{l_{\rm shatter}}$', rotation=0, labelpad=15)
    
    # color bar
    plt.legend()
    plt.grid()
    plt.show()

def add_color(rp = None, trial = '240613_0.1_10', verbose = True, cg_st_epoch = 0):
    # grab the hst file data
    datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'
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
    if verbose: print(dataf['cold_gas'])
    log_mass_frac = np.log10(dataf['cold_gas'] / dataf['cold_gas'][cg_st_epoch])

    if verbose:
        # plot athdf version
        fig, ax1 = plt.subplots(figsize=(5,3))
        
        # plot hst version
        cg_st_epoch = 0
        ax1.plot(dataf['time'] / rp['dt_hdf5'], dataf['cold_gas'] / dataf['cold_gas'][cg_st_epoch], ls='--', color='blue', label='Cold gas mass')
        ax1.plot(dataf['time'] / rp['dt_hdf5'], dataf['warm_gas'] / dataf['cold_gas'][cg_st_epoch], ls='--', color='orange', label='Warm gas mass')
        # ax1.set_yscale('log')
        
        # ax1.set_yscale('log')
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
        
        ax1.legend()
        plt.show()
        
    return log_mass_frac

def plot_point_color(trial = '240613_0.1_10', cg_st_epoch = 0, verbose = False):
    rp, x, y, t_cool_mix, t_cool_min, t_cool_cold = add_point(trial=trial, verbose=verbose)
    print(x,y)
    print(x,y)
    print(x,y)
    log_mass_frac = add_color(rp=rp, trial=trial, verbose=True, cg_st_epoch=cg_st_epoch)
    # normalize
    log_mass_frac = np.nan_to_num(log_mass_frac, posinf=1, neginf=-1)
    # make the plot
    plt.subplots(figsize=(5,4))
    cm = plt.colormaps['bwr_r']
    # scatter the points
    sc = plt.scatter(x, y, marker='o',
                     c=log_mass_frac[-1], vmin=-0.3, vmax=0.3, ec='k', cmap=cm)
    
    # analytical line
    ana_x = np.linspace(0, 1.2, 100)
    ana_y = ana_x * t_cool_mix / t_cool_min * 10. ** (0.6 * ana_x + 0.02)
    plt.plot(ana_x, ana_y, ls='-.', color='k')
    
    # axis
    plt.xlim(0, 1.2)
    plt.ylim(np.power(10., -0.5), np.power(10., 4.5))
    plt.yscale('log')
    plt.xlabel(r'$\mathcal{M}_{\rm hot,\ turb}$')
    plt.ylabel(r'$\frac{R_{\rm cl}}{l_{\rm shatter}}$', rotation=0)
    
    # color bar
    cbar = plt.colorbar(sc, extend='both')
    cbar.ax.set_ylabel(r'$\log_{10} \frac{M_{\rm cold}}{M_{\rm cold, 0}}$', rotation=90, labelpad=10)
    plt.legend()
    plt.grid()
    plt.show()
    return trial, x, y, rp, rp['cloud_radius'] / y, log_mass_frac, t_cool_mix, t_cool_min