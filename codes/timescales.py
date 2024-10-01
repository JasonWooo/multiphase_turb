### import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec

from codes.funcs import *  # import everything in functions
mpl.rcParams.update(mpl.rcParamsDefault)
from codes.jason import plotting_def, plot_prettier
plotting_def()

"""
Plot cooling function, rates, and timescales
"""

def plot_cooling(trial = '', shade = True, plot_which = [1, 1, 1]):
    """plots the cooling function, rate, and timescale from a trial
    plot_which = [bool, bool, bool]
        functions, rates, timescales
    """
    # retrieves run params
    rp = get_rp(trial=trial)

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

    """Temperatures"""
    # the two limits
    T_cold = rp['T_cloud']  # set cold temperature to that of cloud
    T_hot = rp['T_hot']
    T_mix = np.sqrt(T_cold * T_hot)

    def plot_tvlines(shade=False):
        if shade:
            y1, y2 = plt.ylim()
            plt.fill_between(x=[T_cold, 2 * T_cold], y1=y1, y2=y2, linestyle="--", color="slateblue", alpha=0.2)  # cold
            plt.fill_between(x=[2 * T_cold, T_mix], y1=y1, y2=y2, linestyle="--", color="green", alpha=0.2)  # warm
            plt.fill_between(x=[T_mix, rp['T_ceil']], y1=y1, y2=y2, linestyle="--", color="orangered", alpha=0.2)  # hot
        else:
            # only plot the definitions for cold, warm/mix, and hot
            plt.axvline(x=T_cold, lw=1, linestyle="--", color="slateblue", alpha=0.2)
            plt.axvline(x=T_mix, lw=1, linestyle="--", color="green", alpha=0.2)
            plt.axvline(x=T_hot, lw=1, linestyle="--", color="orangered", alpha=0.2)
    
    """Plot the [functions, rates, timescales]"""
    if plot_which[0]:
        """Cooling function"""
        plt.figure(figsize=(4, 2))
        plt.plot(T_arr, Lambda_arr, label=r"$\Lambda(T)$")
        
        plt.ylabel(r"$\Lambda(T)$  $[10^{-23}  \mathrm{ergs ~ cm}^3/\mathrm{s}]$", fontsize=12)
        plt.xlabel(r"$T$(K)", fontsize=12)
        
        plt.yscale("log")
        plt.xscale("log")
        
        plot_tvlines(shade=shade)
        plt.legend()
        plt.show()

    if plot_which[1]:
        """Cooling rate"""
        plt.figure(figsize=(4, 2))
        plt.plot(T_arr, cooling_rate - heating_rate, color='green', label=r"$n^2\Lambda(T) - n\Gamma(T) $")
        plt.plot(T_arr, cooling_rate, color='blue', label=r"$n^2\Lambda(T)$")
        plt.plot(T_arr, heating_rate, color='red', label=r"$n\Gamma(T)$")
        
        plt.ylabel(r"Cooling/Heating rate", fontsize=12)
        plt.xlabel(r"$T$(K)", fontsize=12)
        
        # plt.yscale("log")
        plt.xscale("log")
        plot_tvlines(shade=shade)
        plt.legend()
        plt.show()

    if plot_which[2]:
        """Cooling timescale"""
        plt.figure(figsize=(4, 2))
        plt.plot(T_arr, tcool_arr, label=r"$\Lambda(T)$")
        plt.ylabel(r"$t_{\rm cool} [u]$", fontsize=12)
        plt.xlabel(r"$T$(K)", fontsize=12)
        plt.yscale("log")
        plt.xscale("log")
        plot_tvlines(shade=shade)
        plt.legend()
        plt.show()



def plot_cooling_panel(trial = '', shade = True, fs = 12):
    """plots the cooling function and timescale from a trial
    Everything goes into the same panel
    """
    # retrieves run params
    rp = get_rp(trial=trial)

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
    t_cool_func = lambda T : cf.tcool_calc(rp['rho_hot'] * rp['T_hot'] / T, T, Zsol=1.0, Lambda_fac=1.0, fit_type="max")

    """Temperatures"""
    # the two limits
    T_cold = rp['T_cloud']  # set cold temperature to that of cloud
    T_hot = rp['T_hot']
    T_mix = np.sqrt(T_cold * T_hot)

    # make the plot
    fig, ax1 = plt.subplots(figsize=(4, 3))

    def plot_tvlines(ax, shade=False):
        if shade:
            y1, y2 = ax.get_ylim()
            ax.fill_between(x=[T_cold, 2 * T_cold], y1=y1, y2=y2, linestyle="--", color="slateblue", alpha=0.2)  # cold
            ax.fill_between(x=[2 * T_cold, T_mix], y1=y1, y2=y2, linestyle="--", color="green", alpha=0.2)  # warm
            ax.fill_between(x=[T_mix, rp['T_ceil']], y1=y1, y2=y2, linestyle="--", color="orangered", alpha=0.2)  # hot
        else:
            # only plot the definitions for cold, warm/mix, and hot
            ax.axvline(x=T_cold, lw=1, linestyle="--", color="slateblue", alpha=0.2)
            ax.axvline(x=T_mix, lw=1, linestyle="--", color="green", alpha=0.2)
            ax.axvline(x=T_hot, lw=1, linestyle="--", color="orangered", alpha=0.2)
    
    """
    Plot the [functions and timescales]
    """

    """Cooling function"""
    ax1.plot(T_arr, Lambda_arr, lw=1, ls='-', color='k', alpha=1, label=r"$\Lambda(T)$")
    
    ax1.set_ylabel(r"$\Lambda(T)$  $[\mathrm{ergs ~ cm}^3/\mathrm{s}]$", fontsize=fs)
    ax1.set_xlabel(r"$T$(K)", fontsize=fs)
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_ylim(1e-29, 1e-21)
    
    
    """Cooling timescale"""
    # plot the timescale on a twin axis
    ax2 = ax1.twinx()
    ax2.plot(T_arr, tcool_arr, lw=1, ls='--', color='k', alpha=1, label=r"$t_{\rm cool}$")

    # label the points
    # t_cool,min
    T_tcoolmin = T_arr[np.argmin(tcool_arr)]
    t_cool_min = t_cool_func(T_tcoolmin)
    ax2.scatter(T_tcoolmin, t_cool_min, marker='x', color='k', s=10, linewidths=1)
    ax2.text(T_tcoolmin, t_cool_min / 3, r'$t_{\rm cool, min}$', ha='center')
    # t_cool,peak
    T_peak = T_arr[np.argmax(tcool_arr[0 : find_ind_l(T_arr, 2e4)])]
    t_cool_peak = t_cool_func(T_peak)
    ax2.scatter(T_peak, t_cool_peak, marker='x', color='k', s=10, linewidths=1)

    ax2.text(T_peak, t_cool_peak * 3, r'$t_{\rm cool, peak}$', ha='center')

    ax2.set_ylabel(r"$t_{\rm cool}\ [{\rm Myr}]$", fontsize=fs)
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_xlim(rp['T_floor'], rp['T_ceil'])
    ax2.set_ylim(1e-5, 1e7)

    ax1.legend(loc='lower left', bbox_to_anchor=(0.6, 0), fontsize=fs, alignment='left')
    ax2.legend(loc='lower left', bbox_to_anchor=(0.6, 0.1), fontsize=fs, alignment='left')
    plot_tvlines(ax=ax2, shade=shade)
    
    plt.show()


def plot_cooling_panel_hist(trial = '', shade = True, fs = 12, snapshot = 20):
    """plots the cooling function and timescale from a trial
    Everything goes into the same panel
    Added top panel for plotting temperature hist of one of the snapshots
    """
    """Make the plot"""
    # use gridspec
    fig = plt.figure(figsize=(4, 3), dpi=300)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])  # ratio
    # main plot
    ax1 = fig.add_subplot(gs[1])

    # retrieves run params
    datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'
    rp = get_rp(trial=trial)

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
    t_cool_func = lambda T : cf.tcool_calc(rp['rho_hot'] * rp['T_hot'] / T, T, Zsol=1.0, Lambda_fac=1.0, fit_type="max")

    """Temperatures"""
    # the two limits
    T_cold = rp['T_cloud']  # set cold temperature to that of cloud
    T_hot = rp['T_hot']
    T_mix = np.sqrt(T_cold * T_hot)

    def plot_tvlines(ax, shade=False):
        if shade:
            y1, y2 = ax.get_ylim()
            ax.fill_between(x=[T_cold, 2 * T_cold], y1=y1, y2=y2, linestyle="None", color="slateblue", alpha=0.2)  # cold
            ax.fill_between(x=[2 * T_cold, T_mix], y1=y1, y2=y2, linestyle="None", color="green", alpha=0.2)  # warm
            ax.fill_between(x=[T_mix, rp['T_ceil']], y1=y1, y2=y2, linestyle="None", color="orangered", alpha=0.2)  # hot
        else:
            # only plot the definitions for cold, warm/mix, and hot
            ax.axvline(x=T_cold, lw=1, linestyle="--", color="slateblue", alpha=0.2)
            ax.axvline(x=T_mix, lw=1, linestyle="--", color="green", alpha=0.2)
            ax.axvline(x=T_hot, lw=1, linestyle="--", color="orangered", alpha=0.2)
    
    """
    Plot the [functions and timescales]
    """

    """Cooling function"""
    ax1.plot(T_arr, Lambda_arr, lw=1, ls='-', color='k', alpha=1, label=r"$\Lambda(T)$")
    
    ax1.set_ylabel(r"$\Lambda(T)$  $[\mathrm{ergs ~ cm}^3/\mathrm{s}]$", fontsize=fs)
    ax1.set_xlabel(r"$T$(K)", fontsize=fs)
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_ylim(1e-29, 1e-21)
    
    
    """Cooling timescale"""
    # plot the timescale on a twin axis
    ax2 = ax1.twinx()
    ax2.plot(T_arr, tcool_arr, lw=1, ls='--', color='k', alpha=1, label=r"$t_{\rm cool}$")

    # label the points
    # t_cool,min
    T_tcoolmin = T_arr[np.argmin(tcool_arr)]
    t_cool_min = t_cool_func(T_tcoolmin)
    ax2.scatter(T_tcoolmin, t_cool_min, marker='x', color='k', s=10, linewidths=1)
    ax2.text(T_tcoolmin, t_cool_min / 3, r'$t_{\rm cool, min}$', ha='center')
    # t_cool,peak
    T_peak = T_arr[np.argmax(tcool_arr[0 : find_ind_l(T_arr, 2e4)])]
    t_cool_peak = t_cool_func(T_peak)
    ax2.scatter(T_peak, t_cool_peak, marker='x', color='k', s=10, linewidths=1)

    ax2.text(T_peak, t_cool_peak * 3, r'$t_{\rm cool, peak}$', ha='center')

    ax2.set_ylabel(r"$t_{\rm cool}\ [{\rm Myr}]$", fontsize=fs)
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_xlim(rp['T_floor'], rp['T_ceil'])
    ax2.set_ylim(1e-5, 1e7)

    ax1.legend(loc='lower left', bbox_to_anchor=(0.6, 0), fontsize=fs, alignment='left')
    ax2.legend(loc='lower left', bbox_to_anchor=(0.6, 0.1), fontsize=fs, alignment='left')
    plot_tvlines(ax=ax2, shade=shade)


    """Shared x axis for 1d histogram"""
    ax_hist = fig.add_subplot(gs[0], sharex=ax1)

    fname=f'{datapath}/cloud/Turb.out2.{snapshot:05d}.athdf'
    rho, press = get_datamds(fname=fname, verbose=False, keys=['rho', 'press'])
    hist_data = calc_T(press, rho).flatten()
    hist_data = np.concatenate([hist_data, [800] * 10000, [1600] * 1000])

    bins_log = np.power(10., np.linspace(np.log10(rp['T_floor']), np.log10(rp['T_ceil']), 50))
    counts, bins, _ = ax_hist.hist(hist_data, bins=bins_log, alpha=1, color='grey', edgecolor='None')
    ax_hist.text(ax1.get_xlim()[0] / 10, 1e-6, 'frequency', ha='left', va='bottom', rotation=90, fontsize=fs)
    # ax_hist.set_yscale('log')
    ax_hist.set_ylim(0, np.max(counts))
    ax_hist.axis('off')
    
    plt.subplots_adjust(hspace=0)
    plt.show()

"""
Functions to load bulk quantities in a run
"""

# returns lshat of a run from the cooling times
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
    
    T_peak = T_range[np.argmax(tcool_arr[T_ind_low : find_ind_l(T_arr, 2e4)])]  # find the temperature at which the t_cool is peak
    t_cool_peak = t_cool_func(T_peak)
    
    l_shatter_min = np.min(l_shatter[T_ind_low : T_ind_high])

    if verbose:
        print(f'T_cold = {T_cold}')
        print(f'T_hot = {T_hot:.2e}')
        print(f'T_mix = {T_mix:.2e}')
        print(f'T_peak = {T_peak:.2e}')
        print(f't_cool,peak = {t_cool_peak:.2e}')
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
        plt.figure(figsize=(4, 2))
        plt.plot(T_arr, Lambda_arr, label=r"$\Lambda(T)$")
        
        plt.ylabel(r"$\Lambda(T)$  $[10^{-23}  \mathrm{ergs ~ cm}^3/\mathrm{s}]$", fontsize=12)
        plt.xlabel(r"$T$(K)", fontsize=12)
        
        plt.yscale("log")
        plt.xscale("log")
        
        plot_tvlines()
        plt.show()

        """Cooling rate"""
        plt.figure(figsize=(4, 2))
        plt.plot(T_arr, cooling_rate - heating_rate, color='green', label=r"$n^2\Lambda(T) - n\Gamma(T) $")
        plt.plot(T_arr, cooling_rate, color='blue', label=r"$n^2\Lambda(T)$")
        plt.plot(T_arr, heating_rate, color='red', label=r"$n\Gamma(T)$")
        
        plt.ylabel(r"Cooling/Heating rate", fontsize=12)
        plt.xlabel(r"$T$(K)", fontsize=12)
        
        # plt.yscale("log")
        plt.xscale("log")
        
        # plt.ylim(0.0, 5e-15)
        
        plt.legend()
        plot_tvlines()
        plt.show()

        """Cooling timescale"""
        plt.figure(figsize=(4, 2))
        plt.plot(T_arr, tcool_arr, label=r"$\Lambda(T)$")
        # plt.axhline(y=rp['t_cc'], color="k", linestyle="-", alpha=0.2, label=r"$t_{\rm cc}$")
        # plt.axhline(y=rp['t_eddy'], color="k", linestyle="-", alpha=0.2, label=r"$t_{\rm eddy}$")
        plt.ylabel(r"$t_{\rm cool} [u]$", fontsize=12)
        plt.xlabel(r"$T$(K)", fontsize=12)
        plt.yscale("log")
        plt.xscale("log")
        plt.legend()
        plot_tvlines()
        plt.show()
        
        """l_shatter"""
        plt.figure(figsize=(4, 2))
        # shatter length scale
        plt.plot(T_arr, l_shatter)
        plt.ylabel(r"$l_{\rm shatter}$", fontsize=12)
        plt.xlabel(r"$T$(K)", fontsize=12)
        plt.yscale("log")
        plt.xscale("log")
        plot_tvlines()
        plt.show()

    return l_shatter_min, t_cool_mix, t_cool_min, [t_cool_func, T_tcoolmin, T_mix]  # return the minimum shattering length

# returns x and y info on the Rcl / lshat & Mach parameter space
def add_point(trial = '240613_0.1_10', verbose = True):
    # load function
    datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'
    rp = get_rp(trial)
    if verbose: print(rp)
    l_shatter_min, t_cool_mix, t_cool_min, [t_cool_func, T_tcoolmin, T_mix] = load_lshat(rp = rp, verbose=verbose)
    vel_frac = rp['cloud_radius'] / l_shatter_min

    if verbose:
        print(f"X-value = {rp['mach']}")
        print(f'Y-value [log] = {np.log10(vel_frac):.3f}')
        print(f'Y-value = {vel_frac:.3f}')
    return rp, rp['mach'], vel_frac, t_cool_mix, t_cool_min, [t_cool_func, T_tcoolmin, T_mix]

# retrieve point without color in the parameter space
def plot_point_bw(trial = '240613_0.1_10', verbose = True):
    rp, x, y, t_cool_mix, t_cool_min, _ = add_point(trial=trial, verbose=verbose)
    
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
    # plt.ylim(np.power(10., np.log10(t_cool_mix / t_cool_min) - 2), np.power(10., np.log10(t_cool_mix / t_cool_min) + 2))
    plt.yscale('log')
    plt.xlabel(r'$\mathcal{M}_{\rm hot,\ turb}$')
    plt.ylabel(r'$\frac{R_{\rm cl}}{l_{\rm shatter}}$', rotation=0, labelpad=15)
    
    # color bar
    plt.legend()
    plt.grid()
    plt.show()

# returns cutoff time for a run
def get_stop_time(trial = '240613_0.1_10', rp = None, pfloor = 1e-9, T_cold = 1600, hst_data = None,
                  perc_cell_press = 0.5, perc_mass_hot = 20, perc_mass_cold = 5):  # criteria in percentage
    """
    Returns the stop time of a trial

    This is based on the criteria of the "turb_stop" executable
        >= 0.5% of all cells have pressure p < 2 * pfloor

    Loops through all saves in the folder and returns the time cutoff
    If there are none that satisfy this
        The run is probably done with "turb_stop" already
        The last time in the hst file will be returned
    """
    
    def press_condition(athdf_fname):
        """Takes in file name"""
        fname = f'{datapath}/cloud/{athdf_fname}'
        press = get_datamd(fname=fname, key='press', verbose=False)
        
        # condition
        crit = np.sum(press < 2 * pfloor) >= (perc_cell_press/100 * crit_num_cells)  # pressure condition
        return crit
    
    # grab all athdf files
    datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'
    athdf_fnames = [file for file in os.listdir(f'{datapath}/cloud') if file.startswith("Turb.out2.") and file.endswith(".athdf")]

    # grab the total mass of the first snapshot after init
    print('Retrieving box conditions...')

    # criteria based on total number of cells
    crit_num_cells = rp['grid_dim']**3
    # criteria based on percentage of mass
    crit_cell_mass = hst_data['rho_sum'][1] * ((rp['box_size']/rp['grid_dim']) ** 3)
    # null index for the last snapshot
    null_ind = len(hst_data['time']) - 1

    """
    Pressure conditions
    >= 0.5% of cells have pressure < 2 * pfloor
    """
    
    # get pressure from binary search
    stop_time_press = press_binary_search(datapath=datapath, arr=np.sort(athdf_fnames), condition=press_condition)
    if stop_time_press < hst_data['time'][-1]:
        stop_ind_press = find_ind_l(hst_data['time'], stop_time_press)
    else:
        stop_ind_press = null_ind

    """
    Mass conditions
    Hot gas mass < 20%
    Cold gas mass < 5%
    """

    # update stop_time with mass criteria
    hot_gas_mass = crit_cell_mass - hst_data['cold_gas'] - hst_data['warm_gas']

    crit_mass_hot = crit_cell_mass * (perc_mass_hot/100)
    crit_mass_cold = crit_cell_mass * (perc_mass_cold/100)
    
    # find where hot gas mass drops below critical
    if crit_mass_hot > hot_gas_mass[-1]: # if reached
        stop_ind_mass_hot = find_ind_s(hot_gas_mass, crit_mass_hot)
    else:
        stop_ind_mass_hot = null_ind

    # find where cold gas mass drops below critical
    if crit_mass_cold > hst_data['cold_gas'][-1]: # if reached
        stop_ind_mass_cold = find_ind_s(hst_data['cold_gas'], crit_mass_cold)
    else:
        stop_ind_mass_cold = null_ind

    # use the smallest of the three
    print(stop_ind_press, stop_ind_mass_hot, stop_ind_mass_cold)
    stop_ind = np.min([stop_ind_press, stop_ind_mass_hot, stop_ind_mass_cold])
    stop_time = hst_data['time'][stop_ind]

    return stop_time, stop_ind

# returns color information for a run
def add_color(rp = None, trial = '240613_0.1_10', verbose = True,
              cg_st_epoch = 0, return_cropped = True):
    # grab the hst file data
    datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'
    dataf = get_hst(trial = trial)

    # mass fractions
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

    
    # plot hst version
    cg_st_epoch = 0
    # ax1.plot(dataf['time'] / rp['dt_hdf5'], cold_frac_all, ls='--', color='blue', alpha=0.3)
    # ax1.plot(dataf['time'] / rp['dt_hdf5'], warm_frac_all, ls='--', color='orange', alpha=0.3)
    # # plot the cropped version in solid
    # ax1.plot(dataf['time'][:stop_ind] / rp['dt_hdf5'], cold_frac_cropped, ls='-', color='blue', label='Cold gas mass', alpha=1)
    # ax1.plot(dataf['time'][:stop_ind] / rp['dt_hdf5'], warm_frac_cropped, ls='-', color='orange', label='Warm gas mass', alpha=1)

    # hot gas mass from total mass
    tot_mass = np.sum(get_datamd(fname=f'{datapath}/cloud/Turb.out2.00001.athdf', key='rho', verbose=False)) * ((rp['box_size'] / rp['grid_dim']) ** 3)
    cg_mass = dataf['cold_gas']
    wg_mass = dataf['warm_gas']
    hg_mass = tot_mass - cg_mass - wg_mass

    # crop the mass
    print(tot_mass, cg_mass, wg_mass)
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

    if verbose:
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

# returns color and cutoff time for a run
def add_color_time(rp = None, trial = '240613_0.1_10', verbose = True,
                   cg_st_epoch = 0, return_cropped = True):
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

    # mass fractions
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

    
    # plot hst version
    cg_st_epoch = 0
    # ax1.plot(dataf['time'] / rp['dt_hdf5'], cold_frac_all, ls='--', color='blue', alpha=0.3)
    # ax1.plot(dataf['time'] / rp['dt_hdf5'], warm_frac_all, ls='--', color='orange', alpha=0.3)
    # # plot the cropped version in solid
    # ax1.plot(dataf['time'][:stop_ind] / rp['dt_hdf5'], cold_frac_cropped, ls='-', color='blue', label='Cold gas mass', alpha=1)
    # ax1.plot(dataf['time'][:stop_ind] / rp['dt_hdf5'], warm_frac_cropped, ls='-', color='orange', label='Warm gas mass', alpha=1)

    # hot gas mass from total mass
    tot_mass = np.sum(get_datamd(fname=f'{datapath}/cloud/Turb.out2.00001.athdf', key='rho', verbose=False)) * ((rp['box_size'] / rp['grid_dim']) ** 3)
    cg_mass = dataf['cold_gas']
    wg_mass = dataf['warm_gas']
    hg_mass = tot_mass - cg_mass - wg_mass

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

    return np.log10(cg_mass_cropped[:hot_fill_ind] / cg_mass_cropped[cg_st_epoch]), np.log10(wg_mass_cropped[:hot_fill_ind] / cg_mass_cropped[cg_st_epoch]), time_cropped[:stop_ind][:hot_fill_ind] / rp['t_cc']

# retrieve point with color
def plot_point_color(trial = '240613_0.1_10', cg_st_epoch = 0, verbose = False, return_cropped = True):
    rp, x, y, t_cool_mix, t_cool_min, _ = add_point(trial=trial, verbose=verbose)
    
    log_cold_frac_all, log_warm_frac_all, stop_time = add_color(rp=rp, trial=trial, verbose=True,
                                                                cg_st_epoch=cg_st_epoch, return_cropped=return_cropped)
    # normalize
    log_cold_frac = np.nan_to_num(log_cold_frac_all[-1], posinf=1, neginf=-1)
    log_warm_frac = np.nan_to_num(log_warm_frac_all[-1], posinf=1, neginf=-1)
    # make the plot
    fig, axs = plt.subplots(1, 2, figsize=(10,4))
    cm = plt.colormaps['bwr_r']

    """Plot both cold & hot gas plots"""
    for i, (ax, log_frac) in enumerate(zip(axs, [log_cold_frac, log_warm_frac])):
        # scatter the points
        sc = ax.scatter(x, y, marker='o',
                        c=log_frac, vmin=-0.3, vmax=0.3, ec='k', cmap=cm)
        
        # analytical line
        ana_x = np.linspace(0, 1.2, 100)
        ana_y = ana_x * t_cool_mix / t_cool_min * 10. ** (0.6 * ana_x + 0.02)
        ax.plot(ana_x, ana_y, ls='-.', color='k')
        ax.text(1, 1e4, 'Warm phase' if i else 'Cold phase', ha='center')
        
        # axis
        ax.set_xlim(0, 1.2)
        # ax.set_ylim(np.power(10., -0.5), np.power(10., 4.5))
        ax.set_yscale('log')
        ax.set_xlabel(r'$\mathcal{M}_{\rm hot,\ turb}$')
        ax.set_ylabel(r'$\frac{R_{\rm cl}}{l_{\rm shatter}}$', rotation=0)

        cbar = plt.colorbar(sc, extend='both')
        ax.legend()
        ax.grid()
        
    
    # color bar
    cbar.ax.set_ylabel(r'$\log_{10} \frac{M_{\rm cold}}{M_{\rm cold, 0}}$', rotation=90, labelpad=10)
    plt.show()
    return trial, x, y, rp, rp['cloud_radius'] / y, log_cold_frac, log_warm_frac, t_cool_mix, t_cool_min, rp['grid_dim'], stop_time

# add the params to file
def save_params(trial, trial_x_val, trial_y_val, rp, l_shatter_min, log_cold_frac, log_warm_frac, t_cool_mix, t_cool_min, grid_dim, stop_time,
                csvpath=f'/freya/ptmp/mpa/wuze/multiphase_turb/data/saves/cloud_8e3.csv'):
    import pandas as pd
    df = pd.read_csv(csvpath, index_col=False, comment='#')
    # display(df)
    
    params_dict = {
        'trial' : trial,
        'x_mach' : trial_x_val,
        'y' : trial_y_val,
        'r_cl' : rp['cloud_radius'],
        'l_shat' : l_shatter_min,
        'log_cold_mass' : log_cold_frac,
        'log_warm_mass' : log_warm_frac,
        'T_cloud' : rp['T_cloud'],
        'T_floor' : rp['T_floor'],
        'chi' : rp['chi'],
        't_cool_mix' : t_cool_mix,
        't_cool_min' : t_cool_min,
        'grid_dim' : grid_dim,
        'stop_time' : stop_time,  # the time at which simulation stops
    }
    
    # save trial in csv
    print(trial)
    if trial_df in list(df['trial']):  # if exists
        trial_ind = df.index[df['trial'] == trial_df][0]
        for col, val in params_dict.items():
            df.at[trial_ind, col] = val
    else:  # if no row with trial name, make one
        new_row = pd.DataFrame(params_dict, index=[0])
        df = pd.concat([df, new_row], ignore_index=True)
    
    # save updated csv
    display(df)
    df.to_csv(csvpath, index=False)


"""
Pressure floor
"""

def plot_pressure_floor(trial='240708_0.6_1600000', pfloor=0.0005, file_ind=-1):
    datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'
    # load properties
    rp = get_rp(trial)
    try:
        pfloor = rp['pfloor']
    except:
        pfloor = pfloor  # copy the pressure floor from input
    print(f'pfloor = {pfloor}')
    
    # get the LAST epoch
    athdf_files = [file for file in os.listdir(f'{datapath}/cloud') if file.startswith("Turb.out2.") and file.endswith(".athdf")]
    # middle_file = np.sort(athdf_files)[len(athdf_files)//2]
    ind_file = np.sort(athdf_files)[file_ind]
    
    fname = f'{datapath}/cloud/{ind_file}'
    press = get_datamd(fname=fname, verbose=False, key='press')
    
    fig, ax = plt.subplots(figsize=(4,3))
    if np.any(press < pfloor):
        print('!!!ERROR!!!')
        ax.hist(press.flatten(), alpha=0.5,
                bins=np.logspace(np.log10(np.min(press)), np.log10(np.max(press)), 100))
    else:
        ax.hist(press.flatten(), alpha=0.5,
                bins=np.logspace(np.log10(1e-10), np.log10(1e-1), 100) if pfloor < 1e-8 else np.logspace(np.log10(1e-8), np.log10(1e-0), 100) if pfloor < 0.0005 else np.logspace(np.log10(1e-4), np.log10(1e0), 100))
    # ax.text(1e4, 1000, f'Snapshot {epoch_num}')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e1, 1e7)
    ax.axvline(x=pfloor)
    ax.set_xlabel(r'$P$')
    # ax.axvline(rp['T_hot'], color='red', lw=1, ls='--')
    # ax.axvline(rp['T_cloud'], color='blue', lw=1, ls='--')
    
    plt.show()

"""
Calculate the timescales
"""

def print_timescales(trial):
    datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'
    rp = get_rp(trial)
    rp, x, y, t_cool_mix, t_cool_min, [t_cool_func, T_tcoolmin, T_mix] = add_point(trial=trial, verbose=False)
        
    # calculate t_cc of warm/cold gas
    T_cold = rp['T_cloud']
    T_hot = rp['T_hot']
    T_warm = 8000#np.sqrt(T_cold * T_hot)
    T_mix = np.sqrt(T_cold * T_hot)
    
    # calculate other quantities
    chi_hc = T_hot / T_cold
    chi_wc = T_warm / T_cold
    chi_hw = T_hot / T_warm
    mach = rp['mach']
    R_cl = rp['cloud_radius']
    cloud_box_ratio = 1/50
    
    # load other values
    t_cool_min = 0.0001217175017901
    l_shat = 2.4989679118099205e-06
    vturb = rp['mach'] * calc_cs(T_hot)
    
    # calculate the four timescales
    t_cc_hc = (chi_hc ** (1/2) * R_cl / vturb)
    t_cc_wc = (chi_wc ** (1/2) * R_cl / vturb)
    t_cc_hw = (chi_hw ** (1/2) * R_cl / vturb)
    
    t_grow_hc = chi_hc * mach**(-1/2) * (R_cl / l_shat)**(1/2) * cloud_box_ratio**(-1/6) * t_cool_func(T_warm)
    t_grow_wc = chi_wc * mach**(-1/2) * (R_cl / l_shat)**(1/2) * cloud_box_ratio**(-1/6) * t_cool_func(T_cold)
    t_grow_hw = chi_hw * mach**(-1/2) * (R_cl / l_shat)**(1/2) * cloud_box_ratio**(-1/6) * t_cool_min
    
    print(f't_cc_hc   = {t_cc_hc:>10.3f}; rate = {1/t_cc_hc:>10.3e}')  # add
    print(f't_cc_wc   = {t_cc_wc:>10.3f}; rate = {1/t_cc_wc:>10.3e}')  # add
    print(f't_cc_hw   = {t_cc_hw:>10.3f}; rate = {1/t_cc_hw:>10.3e}')  # subtract
    print(f't_grow_hc = {t_grow_hc:>10.3f}; rate = {1/t_grow_hc:>10.3e}')  # subtract
    print(f't_grow_wc = {t_grow_wc:>10.3f}; rate = {1/t_grow_wc:>10.3e}')  # subtract
    print(f't_grow_hw = {t_grow_hw:>10.3f}; rate = {1/t_grow_hw:>10.3e}')  # add
    print(f't_cool_wc = {t_cool_func(800):>10.3f}; rate = {1/t_cool_func(8000):>10.3e}')  # subtract
    print(f't_cool_hw = {t_cool_func(8000):>10.5f}; rate = {1/t_cool_func(8000):>10.3e}')  # add

    print(1/t_cc_wc + 1/t_grow_hw - 1/t_cc_hw - 1/t_grow_wc)
    print('\n' * 5)