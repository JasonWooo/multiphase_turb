"""
Make plots for the paper
"""

### import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.markers import MarkerStyle
import matplotlib.patheffects as pe
import cmasher as cmr
from tqdm import tqdm
from codes.funcs import *  # import everything in functions
from codes.timescales import *
import codes.timescales_8e3 as ts8e3
mpl.rcParams.update(mpl.rcParamsDefault)
from codes.jason import plotting_def, plot_prettier
plotting_def()


"""
Figure 1
----------------------------------------
Cooling function and cooling timescales
Shown in a single panel
"""

def cooling_8e2(figpath = '/ptmp/mpa/wuze/multiphase_turb/figures/cooling_8e2.pdf', trial = '', snapshot = 22,
                shade = True, fs = 12):
    # use gridspec
    fig = plt.figure(figsize=(4, 3), dpi=200)
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
            ax.fill_between(x=[T_cold, 2 * T_cold], y1=y1, y2=y2, linewidth=0, color="slateblue", alpha=0.2, zorder=-1)  # cold
            ax.fill_between(x=[2 * T_cold, T_mix], y1=y1, y2=y2, linewidth=0, color="green", alpha=0.2, zorder=-1)  # warm
            ax.fill_between(x=[T_mix, rp['T_ceil']], y1=y1, y2=y2, linewidth=0, color="orangered", alpha=0.2, zorder=-1)  # hot
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
    ax1.set_ylim(1e-29, 9e-22)
    
    
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
    ax_hist.set_yscale('log')
    ax_hist.set_ylim(3e4, np.max(counts))
    
    # ax_hist.text(ax1.get_xlim()[0] / 10, ax_hist.get_ylim()[0], 'frequency', ha='left', va='bottom', color='grey', rotation=90, fontsize=fs)
    ax_hist.text(1e7, ax_hist.get_ylim()[0], 'frequency', ha='center', va='bottom', color='grey', rotation=0, fontsize=fs)
    # ax_hist.axis('off')
    ax_hist.tick_params(axis='x',
                        which='both',
                        bottom=False,
                        top=False,
                        labelbottom=False)
    # ax_hist.spines['right'].set_visible(False)
    # ax_hist.spines['top'].set_visible(False)
    # ax_hist.spines['bottom'].set_visible(False)
    plot_tvlines(ax=ax_hist, shade=shade)
    
    plt.subplots_adjust(hspace=0)

    plt.savefig(figpath, format="pdf", bbox_inches="tight")
    plt.show()



"""
Figure 7
----------------------------------------
8e3 runs parameter space: R_cl / l_shatter & Mach number
Shown in 1 panel with the analytical fit
"""

def params_8e3(figpath = '/ptmp/mpa/wuze/multiphase_turb/figures/params_8e3.pdf',
               csvpath = f'/freya/ptmp/mpa/wuze/multiphase_turb/saves/cloud_8e3.csv',
               cm = None, cg_st_epoch = 0,
               show_text = False, shade = False, tfs = 7, lfs = 12,
               verbose = False):
    # load the trials
    df = pd.read_csv(csvpath, comment='#')
    trials = df['trial'].to_numpy()

    # initialize the plot
    fig, ax = plt.subplots(dpi=200, figsize=(4,4))
    
    for trial in trials:
        rp, x, y, t_cool_mix, t_cool_min, t_cool_cold = ts8e3.add_point(trial=trial, verbose=verbose)
        # correct for bad trials
        if trial in ['240612_0.5_1000', '240612_0.8_1200']:
            x = 0.3
        log_cold_frac = ts8e3.add_color(rp=rp, trial=trial, verbose=verbose, cg_st_epoch=cg_st_epoch)
        # normalize for all fractions
        log_frac = np.nan_to_num(log_cold_frac, posinf=1, neginf=-1)
        
        """Plot both cold gas plots"""
        # find the stable points
        t_cc_lim = 2  # when is the point stable
        stable_ind = int(np.ceil(t_cc_lim * rp['t_cc'] / rp['dt_hdf5']))
        log_frac_stable = log_frac[stable_ind:]  # select all points onward
        
        # take the mean and the std
        log_frac_mean = np.clip(np.mean(log_frac_stable), -0.5, 1)
        
        # scatter the points
        sc = ax.scatter(x, y, marker='o',
                        c=log_frac_mean, vmin=-0.3, vmax=0.3, ec='k', cmap=cm)

        if show_text:
            col = log_frac_mean
            if np.isclose(col, -.5): col = -np.inf  # catchs only the cold values
            col = f'{col:.2f}'.replace('-', '\N{MINUS SIGN}')
            plt.text(x + 0.05, y, col, fontsize=8, ha='left', va='center')
        
    # analytical line
    ana_x = np.linspace(0, 1.2, 100)
    ana_y = ana_x * (t_cool_cold) / t_cool_min * 10. ** (0.6 * ana_x) / 1.5

    # plot shaded region instead of a single line
    if shade:
        ax.fill_between(ana_x, y1=1e0, y2=ana_y, ls='-.', lw=1, color='pink', alpha=0.5, zorder=-1)
        ax.fill_between(ana_x, y1=ana_y, y2=1e5, ls='-.', lw=1, color='slateblue', alpha=0.5, zorder=-1)
    else:
        ax.plot(ana_x, ana_y, ls='-.', lw=1, color='k', alpha=0.5)
    # region labels
    ax.text(1, ana_y[-1]/4, s='destroyed', ha='right', va='center', rotation=10, fontsize=tfs)
    ax.text(1, ana_y[-1], s='survives', ha='right', va='center', rotation=10, fontsize=tfs)
    
    # axis
    ax.set_xlim(0.18, 1.02)
    ax.set_ylim(np.power(10., 0), np.power(10., 5))
    ax.set_yscale('log')
    ax.set_xlabel(r'$\mathcal{M}_{\rm hot,\ turb}$', fontsize=lfs)
    ax.set_ylabel(r'$\frac{R_{\rm cl}}{l_{\rm shatter}}$', rotation=0, labelpad=14, fontsize=lfs)
    # ax.legend()
    # ax.grid()

    # add colorbar
    cbar_ax = fig.add_axes([0.915, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(sc, cax=cbar_ax, extend='both')
    # cbar1.set_ticklabels([])
    cbar.ax.set_xlabel('cold', ha='left', labelpad=10, fontsize=tfs)
    cbar.ax.set_ylabel(r'$\log_{10} \frac{M_{\rm cold}}{M_{\rm cold, 0}}$', rotation=90, labelpad=3, fontsize=lfs)
    
    # save and show the figure
    plt.savefig(figpath, format="pdf", bbox_inches="tight")
    plt.show()
    return trial, x, y, rp, rp['cloud_radius'] / y, log_cold_frac, t_cool_mix, t_cool_min



"""
Figure 5
----------------------------------------
8e2 runs parameter space: R_cl / l_shatter & Mach number
Shown in 1 panel
WITH analytical fits

The pickle file is saved from plot_params_all_file_s() in parameter_func_8e2_sim.ipynb
"""

def params_8e2(figpath = '/ptmp/mpa/wuze/multiphase_turb/figures/params_8e2.pdf',
               pickle_path = f'/freya/ptmp/mpa/wuze/multiphase_turb/saves/cloud_8e2_new.pkl',
               T_cold = 800, T_hot = 4e6,
               log_ylimu = 9, ms = 30, plot_ana = False, shade = False, show_text = True, legend_on_plot = True,
               cm_cold = None, cm_warm = None, lfs = 10, tfs = 7):
    from matplotlib.gridspec import GridSpec

    with open(pickle_path, 'rb') as handle:
        plt_elements = pickle.load(handle)

    # initialize the plot
    fig, ax = plt.subplots(dpi=200, figsize=(4.2, 4))
    # make the markers
    lmarker = MarkerStyle("o", fillstyle="left")
    rmarker = MarkerStyle("o", fillstyle="right")
    
    """Plot both cold & hot gas plots from the pickle file"""
    i = 0
    scs = [None, None]
    for marker, cm in \
    zip([lmarker, rmarker], [cm_cold, cm_warm]):
        [x, y, pt_size, log_frac_mean] = plt_elements[i]

        # plot the point
        if i:  # warm, make linear
            scs[i] = ax.scatter(x, y, marker=marker, s=pt_size,
                                c=log_frac_mean, vmin=0, vmax=1, ec='k', cmap=cm)
        else:  # cold, colorbar in log
            scs[i] = ax.scatter(x, y, marker=marker, s=pt_size,
                                c=log_frac_mean, vmin=-0.3, vmax=0.3, ec='k', cmap=cm)
        i += 1  # update the savescounter

    if show_text:
        [x_c, y_c, pt_size_c, log_frac_mean_c] = plt_elements[0]
        [x_w, y_w, pt_size_w, log_frac_mean_w] = plt_elements[1]  # for warm / cold
        for j in range(len(x_w)):
            col = log_frac_mean_w[j]
            if np.isclose(col, -.5): col = -np.inf  # catchs only the cold values
            col = f'{col:.1f}'.replace('-', '\N{MINUS SIGN}')
            ax.text(x_w[j] + 0.3 * np.sqrt(pt_size_w[j])/ms, y_w[j], col, fontsize=6, ha='center', va='center', path_effects=[pe.withStroke(linewidth=1, foreground="white")])

            col = log_frac_mean_c[j]
            if np.isclose(col, -.5): col = -np.inf  # catchs only the cold values
            col = f'{col:.1f}'.replace('-', '\N{MINUS SIGN}')
            ax.text(x_c[j] - 0.3 * np.sqrt(pt_size_c[j])/ms, y_c[j], col, fontsize=6, ha='center', va='center', path_effects=[pe.withStroke(linewidth=1, foreground="white")])

    """Plot the analytic line"""
    ana_lines = []
    if plot_ana:
        plt.rcParams['text.latex.preamble'] = r'''
        \usepackage[utf8]{inputenc}
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{ulem}
        \usepackage{textcomp}'''

        ana_x = np.linspace(0, 1.2, 100)
        import sys
        sys.path.append(os.path.abspath('/freya/ptmp/mpa/wuze/multiphase_turb/athena/cooling_scripts'))
        import cooling_fn as cf
        t_cool_func = lambda T : cf.tcool_calc(T_hot * 0.1 / T, T, Zsol=1.0, Lambda_fac=1.0, fit_type="max")
        T_range = np.logspace(np.log10(800), np.log10(T_hot), 100)  # in kelvin
        tcool_range = np.vectorize(t_cool_func)(T_range)
        t_cool_min = np.min(tcool_range)
        T_tcoolmin = T_range[np.argmin(tcool_range)]
        # original analytical line
        # ana_y = ana_x * t_cool_mix / t_cool_min * 10. ** (0.6 * ana_x + 0.02) / 5
        # ax.plot(ana_x, ana_y, ls='-.', color='k', alpha=0.5, label='OG ana')
    
        """warm analytical line 1"""
        # temperatures
        T_cold = T_cold
        T_peak = 8.57e+03
        T_warm = 8e3
        T_hot = T_hot
        omega = T_cold / T_tcoolmin
        
        # timescale @ temperature of interest: where t_cool peaks, aka T_cold for 8000K runs
        t_cool_peak = t_cool_func(T_peak)

        # recalculate cooling function from hot to warm gas
        # ratio of t_cc(hot/cold) to t_cc(hot/warm), to get back to the original
        ratio_tcc = np.sqrt(T_warm / T_cold)  # warm / cold
        
        ana_y = ana_x * t_cool_peak / t_cool_min * ratio_tcc * (10. ** (0.6 * ana_x)) * np.sqrt(omega)
        ana_lines.append([ana_x, ana_y])

        """cold analytical line"""
        # temperatures
        T_cold = T_cold
        T_peak = 8.57e+03
        T_warm = 8e3
        T_hot = T_hot
        omega = T_cold / T_tcoolmin
        
        # timescale @ temperature of interest: where t_cool peaks, aka T_cold for 8000K runs
        t_cool_peak = t_cool_func(T_peak)

        # recalculate cooling function from warm to cold gas
        # ratio of t_cc(hot/cold) to t_cc(warm/cold), to get back to the original            
        ratio_tcc = np.sqrt(T_hot / T_warm)  # hot / warm
        
        ana_y = ana_x * t_cool_peak / t_cool_min * ratio_tcc * (10. ** (0.6 * ana_x)) * np.sqrt(omega)
        ana_lines.append([ana_x, ana_y])

        """warm analytical line 2"""
        # temperatures
        T_cold = T_cold
        T_peak = 8.57e+03
        T_warm = 8e3
        T_hot = T_hot
        omega = T_cold / T_tcoolmin

        # timescale of interest: growth time for the warm & cold gas
        chi_wc = T_warm / T_cold
        l_shat = 2.4989679118099205e-06
        cloud_box_ratio = 1/50
        cs_hot = calc_cs(T_hot)
        # t_cool of minimum temperature (800K)
        t_cool_low = t_cool_func(T_cold)

        # old
        # ana_y = ana_x *\
        # (cs_hot * l_shat / t_cool_low) ** (2/3) *\
        # (10. ** (0.6 * ana_x))**(2/3) *\
        # (cloud_box_ratio**(1/9)) *\
        # (chi_wc**(1/3)) *\
        # np.sqrt(omega)

        # new
        ana_y = ana_x *\
        (cs_hot * t_cool_low / l_shat) ** 2 *\
        (10. ** (0.6 * ana_x)) ** 2 *\
        cloud_box_ratio ** (-1/3) *\
        chi_wc *\
        np.sqrt(omega)

        ana_lines.append([ana_x, ana_y])


        """
        Make the shading or the line separating different regimes
        """
        if shade:
            # both destroyed
            ax.fill_between(ana_x, y1=np.power(10., 0.5),
                            y2=ana_lines[0][1], ls='-.', lw=1, ec='None', color='red', alpha=0.2, zorder=-1)
            # warm survived
            ax.fill_between(ana_x, y1=ana_lines[0][1],
                            y2=ana_lines[1][1], ls='-.', lw=1, ec='None', color='orange', alpha=0.2, zorder=-1)
            # both survived
            ax.fill_between(ana_x, y1=ana_lines[1][1],
                            y2=ana_lines[2][1], ls='-', lw=1, ec='None', color='green', alpha=0.2, zorder=-1)
            # cold survived
            ax.fill_between(ana_x, y1=ana_lines[2][1],
                            y2=np.power(10., log_ylimu), ls='-', lw=1, ec='None', color='teal', alpha=0.2, zorder=-1)
        
        # plot the lines only
        ax.plot(ana_lines[0][0], ana_lines[0][1], lw=1, ls='-.', color='orangered', alpha=0.5, label=r'$\frac {t_{\rm cc,hw}} {t_{\rm cool,peak}}$')
        ax.plot(ana_lines[1][0], ana_lines[1][1], lw=1, ls='--', color='blue', alpha=0.5, label=r'$\frac {t_{\rm cc,wc}} {t_{\rm cool,peak}}$')
        ax.plot(ana_lines[2][0], ana_lines[2][1], lw=1, ls='-.', color='brown', alpha=0.5, label=r'$\frac {t_{\rm cc,wc}} {t_{\rm grow,wc}}$')
        
        # region labels
        # ax.text(0.9, ana_lines[0][1][-1]/10, s=r'\textbf{\xout{c} \xout{w}}', ha='right', va='center', rotation=10, fontsize=7)  # the old fancy labels
        # ax.text(0.9, ana_lines[1][1][-1]/4, s=r'\textbf{\xout{c} \textcircled{w}}', ha='right', va='center', rotation=10, fontsize=7)
        # ax.text(0.9, ana_lines[2][1][-1]/1e2, s=r'\textbf{\textcircled{c}\textcircled{w}}', ha='right', va='center', rotation=10, fontsize=7)
        # ax.text(0.9, ana_lines[2][1][-1], s=r'\textbf{\textcircled{c} \xout{w}}', ha='right', va='center', rotation=10, fontsize=7)
        def find_ana_ind(which_ana, where_x):
            [ana_x, ana_y] = ana_lines[which_ana]
            return ana_y[find_ind_l(ana_x, where_x)]
        # from bottom to top
        where_x = 0.91
        rot = 7
        ax.text(where_x, find_ana_ind(0, where_x)/10, s=r'\texttt{cDwD}', ha='right', va='center', rotation=rot, fontsize=tfs)  # the new labels consistent with text
        ax.text(where_x, find_ana_ind(0, where_x)*3, s=r'\texttt{cDwS}', ha='right', va='center', rotation=rot, fontsize=tfs)
        ax.text(where_x, find_ana_ind(1, where_x)*30, s=r'\texttt{cSwS}', ha='right', va='center', rotation=rot, fontsize=tfs)
        ax.text(where_x, find_ana_ind(2, where_x)*10, s=r'\texttt{cSwD}', ha='right', va='center', rotation=rot, fontsize=tfs)


    """legend"""
    if legend_on_plot:
        where_x = 0.25
        rot = 10; sep = 1.2
        # ax.text(where_x, find_ana_ind(0, where_x) * sep, s=r'${t_{\rm cc,hw}}$', ha='left', va='bottom', rotation=rot, fontsize=tfs)
        # ax.text(where_x, find_ana_ind(1, where_x) * sep, s=r'${t_{\rm cc,wc}}$', ha='left', va='bottom', rotation=rot, fontsize=tfs)
        # ax.text(where_x, find_ana_ind(2, where_x) * sep, s=r'${t_{\rm cc,wc}}$', ha='left', va='bottom', rotation=rot, fontsize=tfs)
        # ax.text(where_x, find_ana_ind(0, where_x), s=r'$={t_{\rm cool,peak}}$', ha='left', va='top', rotation=rot, fontsize=tfs)
        # ax.text(where_x, find_ana_ind(1, where_x), s=r'$={t_{\rm cool,peak}}$', ha='left', va='top', rotation=rot, fontsize=tfs)
        # ax.text(where_x, find_ana_ind(2, where_x), s=r'$={t_{\rm grow,wc}}$', ha='left', va='top', rotation=rot, fontsize=tfs)
        ax.text(where_x, find_ana_ind(0, where_x) * sep / 3, s=r'$\frac {t_{\rm cc,hw}} {t_{\rm cool,peak}}$', ha='left', va='center', rotation=rot-1, fontsize=tfs+2)  # bottom to top
        ax.text(where_x, find_ana_ind(1, where_x) * sep * 3, s=r'$\frac {t_{\rm cc,wc}} {t_{\rm cool,peak}}$', ha='left', va='center', rotation=rot, fontsize=tfs+2)
        ax.text(where_x, find_ana_ind(2, where_x) * sep * 3, s=r'$\frac {t_{\rm cc,wc}} {t_{\rm grow,wc}}$', ha='left', va='center', rotation=rot+2, fontsize=tfs+2)

        ax.set_xlim(0.23, 0.92)
    else:  # standalone legend
        ax.legend(loc='lower left',
                bbox_to_anchor=[1, 0.75, 0.5, 0.1], fontsize=lfs,
                frameon=False, fancybox=False, edgecolor='darkgray')
        ax.set_xlim(0.28, 0.92)
    # axis
    ax.set_ylim(np.power(10., 0.5), np.power(10., log_ylimu))
    ax.set_yscale('log')
    ax.set_xlabel(r'$\mathcal{M}_{\rm hot,\ turb}$', fontsize=lfs)
    ax.set_ylabel(r'$\frac{R_{\rm cl}}{l_{\rm shatter}}$', rotation=0, labelpad=14, fontsize=lfs)

    """color bar"""
    cbar_width = 0.37
    # cold
    # cbar_ax1 = fig.add_axes([0.915, 0.1, 0.02, cbar_width])  # [left, bottom, width, height]
    cbar_ax1 = fig.add_axes([0.1, 0.92, cbar_width, 0.02])  # [left, bottom, width, height]
    cbar1 = plt.colorbar(scs[0], cax=cbar_ax1, orientation='horizontal', extend='both')
    # cbar1.set_ticklabels([])
    cbar_ax1.xaxis.set_ticks_position('top'); cbar_ax1.tick_params(axis='x', which='major', pad=1)  # make x axis ticks on top
    cbar1.ax.set_xlabel(r'$\log_{10} \frac{M_{\rm cold}}{M_{\rm cold, 0}}$', labelpad=-35, fontsize=lfs)
    # cbar1.ax.set_ylabel('c', labelpad=15, rotation=0, fontsize=tfs)
    
    # warm
    # cbar_ax2 = fig.add_axes([1.025, 0.1, 0.02, cbar_width])  # [left, bottom, width, height]
    cbar_ax2 = fig.add_axes([0.9-cbar_width, 0.92, cbar_width, 0.02])  # [left, bottom, width, height]
    cbar2 = plt.colorbar(scs[1], cax=cbar_ax2, orientation='horizontal', extend='max')
    # cbar2.ax.set_ylabel(r'$\log_{10} \frac{M_{\rm phase}}{M_{\rm cold, 0}}$ / slope', rotation=90, labelpad=3)
    cbar_ax2.xaxis.set_ticks_position('top'); cbar_ax2.tick_params(axis='x', which='major', pad=1)  # make x axis ticks on top
    cbar2.ax.set_xlabel(r'$\frac{t_{\rm grow, warm}} {t_{\rm cc}}$', labelpad=-35, fontsize=lfs)
    # cbar2.ax.set_ylabel('w', labelpad=5, rotation=0, fontsize=tfs)

    # ax.grid()
    plt.savefig(figpath, format="pdf", bbox_inches="tight")
    plt.show()




"""
Figure 2
----------------------------------------
Density evolution plot
For runs in different regions of the parameter space
"""

def density_evol_load(trials = ['240715_0.8_16', '240711_0.4_1600', '240715_0.8_160000', '240711_0.8_160000000'],
                      tccs = [0, 0.5, 1, 1.5]):
    # the row and columns to plot
    trials = trials[::-1]  # largest radii on top
    rcls = []
    
    num_runs = len(trials)
    num_epochs = len(tccs)

    # array contraining all image data
    rho_data = [[0]*num_epochs for i in range(num_runs)]

    for i, trial in tqdm(enumerate(trials)):
        # for each cloud size
        datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'

        """Grab t_cc"""
        rp = get_rp(trial=trial)
        # rcls.append(rp['cloud_radius'])
        rcls.append(trial.split('_')[-1])
        tcc = rp['t_cc']
        
        # convert to epochs
        time_code = tcc * np.array(tccs)
        epochs = np.floor_divide(time_code, rp['dt_hdf5']).astype(int)  # floor divide

        for j, epoch in enumerate(epochs):
            fname=f'{datapath}/cloud/Turb.out2.{epoch:05d}.athdf'
            # print(fname)
            rho = get_datamd(fname=fname, verbose=False, key='rho')
            
            # PROJECTED density, not averaged
            dens_proj = np.sum(rho, axis=0)
            rho_data[i][j] = dens_proj
    
    return num_runs, num_epochs, rho_data, rcls

def density_evol_plot(data, tccs,
                      figpath = '/ptmp/mpa/wuze/multiphase_turb/figures/density_evol.pdf',
                      cmap = 'viridis',
                      lfs = 16, tfs = 18):
    num_runs, num_epochs, rho_data, rcls = data
    # Create the figure and axes
    fig, axes = plt.subplots(num_runs, num_epochs, figsize=(2 * num_epochs, 2 * num_runs), dpi=300)
    
    # normalizations
    norm = mpl.colors.LogNorm(vmin=1e1, vmax=3e3)

    # Define titles for the columns and rows
    column_titles = [fr'{_}$\ t_{{\rm cc}}$' for _ in tccs]
    # R_cl values in pc
    # row_titles = rcls
    row_titles = [20, 2000, 200000, 200000000][::-1]

    # text for regime labels
    regimes = [r'\texttt{cDwD}', r'\texttt{cDwS}', r'\texttt{cSwS}', r'\texttt{cSwD}'][::-1]

    # Plot each panel
    for i in range(num_runs):
        # regime labels
        ax_temp = axes[i, 0]
        ax_temp.text(0.02, 0.98, regimes[i],
                     color='white', transform=ax_temp.transAxes, ha='left', va='top', fontsize=lfs)
        for j in range(num_epochs):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')

            # make the plot
            data = rho_data[i][j]
            img = ax.imshow(data, cmap=cmap, norm=norm)
            if i == 0:
                ax.set_title(column_titles[j], fontsize=lfs)
            if j == 0:
                coeff, expo = s_n(row_titles[i])
                ax.set_ylabel(fr'${coeff:.0f}\times 10^{{{expo:.0f}}}$', fontsize=lfs)

    fig.supylabel(r'$R_{\rm cl} / l_{\rm shatter}$', x=0.05, fontsize=lfs)

    cax = np.array(axes).flatten()[num_epochs * 2 - 1].inset_axes([1.1, -2, 0.08, 4])
    cbar = fig.colorbar(img, cax=cax, orientation='vertical', location='right', pad=0.1, shrink=0.8, aspect=30)
    cbar.set_label(r'projected density $\rho$ [amu]', fontsize=lfs) #mp/$\rm{cm}^3$

    # Add the main title
    fig.suptitle(r'$T_{\rm cloud} = 8\times 10^2\ {\rm K},\ L_{\rm box}/R_{\rm cl} = 50$', fontsize=tfs, x=0.53, y=0.95)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    
    # save and show the plot
    plt.savefig(figpath, format="pdf", bbox_inches="tight")
    plt.show()




"""
Figure 3
----------------------------------------
Temperature slices for a few runs
Taken at different snapshots

tccs corresponds to each trial
slice_layers gives n = 1 + 2 x slice
"""

def temp_rcl_load(trials = ['240715_0.8_16', '240711_0.4_1600', '240715_0.8_160000', '240711_0.8_160000000'],
                  tccs = [0.5, 1, 2.4, 1.1], slice_layers=1):
    # the row and columns to plot
    trials = trials  # largest radii on top
    rcls = []  # array for normalized cloud size
    
    if len(trials) != len(tccs):  # check that they have the same length
        return None
    num_runs = len(trials)  # number of runs

    # array contraining all image data for temperature, length num_runs
    temp_data = []

    for i, trial in tqdm(enumerate(trials)):
        # for each cloud size
        datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'

        """Grab t_cc"""
        rp = get_rp(trial=trial)
        # rcls.append(rp['cloud_radius'])
        rcls.append(trial.split('_')[-1])
        tcc = rp['t_cc']
        
        # convert to the snapshot
        tcc_trial = tccs[i]  # get the tcc corresponding to this trial
        time_code = tcc * tcc_trial
        # the specific snapshot
        epoch = np.floor_divide(time_code, rp['dt_hdf5']).astype(int)  # floor divide

        """Grab the temperature distribution"""
        fname=f'{datapath}/cloud/Turb.out2.{epoch:05d}.athdf'
        # print(fname)
        rho = get_datamd(fname=fname, verbose=False, key='rho')
        press = get_datamd(fname=fname, verbose=False, key='press')
        temperature = calc_T(press, rho)
        
        # append the flattened/sliced temperature
        resol_mid = int(len(temperature)/2)  # half of the resolution, take the center-most slice
        resol_com = np.argmax(np.sum(temperature < 800, axis=(1, 2))) # slice with the most cold gas, center-of-mass

        temp_slice = np.mean(temperature[resol_com-slice_layers:resol_com+slice_layers], axis=0)  #temperature[resol_com]
        temp_data.append(temp_slice)
    
    return num_runs, temp_data, rcls

def temp_rcl_plot(data, tccs,
                  figpath = '/ptmp/mpa/wuze/multiphase_turb/figures/temp_rcl.pdf',
                  vmin = 800, vmax = 1e5,
                  cmap = 'viridis', lfs = 7, tfs = 10):
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
    regimes = [r'\texttt{cDwD}', r'\texttt{cDwS}', r'\texttt{cSwS}', r'\texttt{cSwD}']

    # plot each panel
    for i in range(num_runs):
        ax = axes[i]
        ax.set_xticks([])
        ax.set_yticks([])

        # label the regimes
        ax.text(0.98, 0.02, regimes[i],
                color='white', transform=ax.transAxes, ha='right', va='bottom', fontsize=lfs)

        # make the plot
        data = temp_data[i]
        img = ax.imshow(data, cmap=cmap, norm=norm)
        
        # label the rcls
        coeff, expo = s_n(rcl_titles[i])
        ax.text(len(data)/2, len(data)+2, fr'${coeff:.0f}\times 10^{{{expo:.0f}}}\ l_{{\rm shatter}}$', ha='center', va='bottom', fontsize=tfs)

        # label the times
        ax.set_xlabel(tcc_titles[i], fontsize=lfs)

    # fig.supylabel(r'$R_{\rm cl} / l_{\rm shatter}$', x=0.08, fontsize=lfs)

    # colorbar
    cax = axes[0].inset_axes([0.05, -0.3, 4, 0.08])  #[1.1, -0.2, 0.08, 1.4]
    cbar = fig.colorbar(img, cax=cax, orientation='horizontal',
                        location='bottom', pad=0.1, shrink=0.8, aspect=30, extend='max')
    cax.tick_params(labelsize='small')
    cax.text(3e4, -1, 'T [K]', color='k', ha='center', va='center', alpha=1, rotation=0, fontsize=tfs)
    cax.axvline(1600, lw=0.5, ls='--', color='k', alpha=0.5)
    # cax.text(-1, 1700, r'$1600\ \rm{K}$', color='k', ha='center', va='center', alpha=0.5, rotation=90)
    cax.text(1300, -1, 'cold', color='blue', ha='center', va='center', alpha=0.5, rotation=0, fontsize=lfs)
    cax.text(2000, -1, 'warm', color='purple', ha='center', va='center', alpha=0.5, rotation=0, fontsize=lfs)

    # add the main title
    fig.suptitle(r'$T_{\rm cloud} = 8\times 10^2\ {\rm K},\ L_{\rm box}/R_{\rm cl} = 50$', va='top', fontsize=tfs, x=0.53, y=0.96)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    
    # save and show the plot
    plt.savefig(figpath, format="pdf", bbox_inches="tight")
    plt.show()






"""
Figure 4
----------------------------------------
Cold & warm gas mass evolution plot for the Tfloor = 8e2 runs
With a particular Mach number
"""

def mass_evol(figpath = '/ptmp/mpa/wuze/multiphase_turb/figures/mass_evol.pdf',
              csvpath = '/freya/ptmp/mpa/wuze/multiphase_turb/saves/cloud_8e3.csv',
              mach = 0.4,
              cm = None, alpha = 0.5, verbose = False, plot_growth = 0, plot_legend = False):
    """
    Plots evolution of only COLD
    Plots ALL runs for a single Mach number

    Corresponds to Figure 6 in the paper
    -----
    mach: only plot the points for a certain mach number
    """
    from matplotlib.collections import LineCollection
    import pandas as pd
    df = pd.read_csv(csvpath, comment='#')
    df['yval'] = df["r_cl"] / df["l_shat"]
    df.sort_values(by='yval', inplace=True)
    
    fig, ax = plt.subplots(dpi=200, figsize=(5,3))
    
    for _, row in tqdm(df.iterrows()):
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
    
        """Load run parameters"""
        rp = get_rp(trial=row['trial'])
    
        x = dataf['time'] / rp['t_eddy']
        y = dataf['cold_gas'] / dataf['cold_gas'][cg_st_epoch]
        
        from scipy.ndimage.filters import gaussian_filter1d
        ysmoothed = gaussian_filter1d(y, sigma=10)

        color = cm(plt.Normalize(0, 8)(np.log10(row["yval"])))

        # get the normalized cloud size
        coeff, expo = s_n(row["yval"])
        ax.plot(x, ysmoothed, lw=1, ls='-', color=color, alpha=0.5, zorder=1, label=fr'${coeff:.0f}\times 10^{{{expo:.0f}}}$')  #R/l_{{\rm shatter}} = 


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


    # add a horizontal dashed line at y=1
    ax.axhline(y=1, lw=0.8, ls='--', alpha=0.5, color='k', zorder=-1)

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
    
    if plot_legend:
        cbar_dims = [0.915, 0.10, 0.02, 0.4]
        ax.legend(loc='lower right', bbox_to_anchor=(1.17, 0.5), fontsize=5, alignment='left')
    else:
        cbar_dims = [0.915, 0.12, 0.02, 0.75]
    
    # add colorbar
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 8))
    sm.set_array([])
    cbar_ax = fig.add_axes(cbar_dims)  # [left, bottom, width, height]
    cbar = plt.colorbar(sm, cax=cbar_ax, ax=ax)#, extend='both')
    cbar.set_ticks([0, 2, 4, 6, 8])
    cbar.set_ticklabels([0, 2, 4, 6, 8])
    cbar.ax.set_ylabel(r'$\log_{10} \frac{R_{\rm cl}}{l_{\rm shatter}}$', rotation=90, labelpad=10)
    ax.set_title(fr'${{\mathcal{{M}} = {mach}}}$, resolution: ${rp['grid_dim']}^3$')

    plt.savefig(figpath, format="pdf", bbox_inches="tight")
    plt.show()

def mass_evol_both(figpath = '/ptmp/mpa/wuze/multiphase_turb/figures/mass_evol_both.pdf',
                    csvpath = '/freya/ptmp/mpa/wuze/multiphase_turb/saves/cloud_8e3.csv',
                    mach = 0.4, plot_growth = 0,
                    cm = None, alpha = 0.5, verbose = False, plot_legend = False,
                    lfs = 12, tfs = 14):
    """
    Plots evolution of both COLD AND WARM
    Plots ALL runs for a single Mach number

    The mass evolutions are cropped for cold gas based on the following criteria,
    dictated by the 'stop_time' column in the .csv,
    which is decided by get_stop_time() in timescales.py

    Warm gas is plotted the whole way
        ○ Hot gas mass lower than
			§ 0.5
			§ 0.2 if necessary
		○ Cold gas mass lower than
			§ 5%
		○ Pressure floor
			§ Old runs


    Corresponds to Figure 6 in the paper
    -----
    mach: only plot the points for a certain mach number
    plot_growth: plot the expected growth rates from two-phase simulations (-1 = highest, 0 = no plot)
    """
    from matplotlib.collections import LineCollection
    import pandas as pd
    df = pd.read_csv(csvpath, comment='#')
    df['yval'] = df["r_cl"] / df["l_shat"]
    df.sort_values(by='yval', inplace=True)
    
    # plots
    fig = plt.figure(figsize=(4, 5), dpi=200)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])  # ratio
    ax1 = fig.add_subplot(gs[0])  # cold
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # warm
    
    xstop_list = []
    for _, row in tqdm(df.iterrows()):
        if (row['x_mach'] < mach - 0.05) or (row['x_mach'] > mach + 0.05):  # if not within range
            continue
        trial = row['trial']
        datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'
        
        # get mass evol history
        rp = get_rp(trial=trial)
        dataf = get_hst(trial=trial)
        x_stop, y_cg, y_wg, x_nostop = get_stopped_mass_evol(trial=trial, stop_time=row['stop_time'])
        
        # smoothen the lines
        from scipy.ndimage.filters import gaussian_filter1d
        y_cg_smoothed = gaussian_filter1d(y_cg, sigma=10)
        y_wg_smoothed = gaussian_filter1d(y_wg, sigma=10)

        # get color
        color = cm(plt.Normalize(0, 8)(np.log10(row["yval"])))

        # get the normalized cloud size
        coeff, expo = s_n(row["yval"])
        """cold, stopped"""
        xstop_list.append(x_stop[-1])
        ax1.plot(x_stop, y_cg_smoothed, lw=1, ls='-', color=color, alpha=0.5, zorder=1, label=fr'${coeff:.0f}\times 10^{{{expo:.0f}}}$')  #R/l_{{\rm shatter}} = 
        ax1.scatter(x_stop[-1], y_cg_smoothed[-1], marker='x', s=8, linewidths=1, color=color, alpha=0.5)  # add a cross at the end
        
        """warm, stopped"""
        ax2.plot(x_stop, y_wg_smoothed, lw=1, ls='-', color=color, alpha=0.5, zorder=1, label=fr'${coeff:.0f}\times 10^{{{expo:.0f}}}$')  #R/l_{{\rm shatter}} = 
        ax2.scatter(x_stop[-1], y_wg_smoothed[-1], marker='x', s=8, linewidths=1, color=color, alpha=0.5)  # add a cross at the end


        """Plot all expected growths"""
        if plot_growth:
            if (plot_growth == -1) and (row["yval"] < 1e8):
                continue
            # load t_cool_min
            def calc_tgrow(rcl_lshat, mach,
                        alpha=0.5, T=800, T_hot=4e6,
                        t_cool_min = 0.0001217175017901, l_shat = 2.4989679118099205e-6, box_rad=50):
                # calculate t_cc
                from codes.funcs import calc_cs
                vturb = mach * calc_cs(T_hot)
                chi = T_hot / T
                t_cc = (chi ** (1/2) * rcl_lshat * l_shat / vturb)
                # calculate tgrow
                tgrow = alpha * chi
                tgrow *= (t_cool_min * t_cc) ** (1/2)
                tgrow *= (box_rad) ** (1/6)
                return tgrow / t_cc # calculated
    
            # use actual time, plot eddie time
            tgrow_pred = calc_tgrow(float(row["yval"]), mach=mach, T=800)  # cold gas growth rate
            cold_frac = np.exp(x_stop / tgrow_pred)
            ax1.plot(x_stop, cold_frac, lw=1, ls=':', alpha=0.5, color=color)

    # add a horizontal dashed line at y=1
    ax1.axhline(y=1, lw=0.8, ls='--', alpha=0.5, color='k', zorder=-1)
    ax2.axhline(y=1, lw=0.8, ls='--', alpha=0.5, color='k', zorder=-1)
    
    """Cold"""
    # y axis
    ax1.set_yscale('log')
    ax1.set_ylim(1/7, 7)
    yticks = [1/5, 1, 5]
    ax1.set_yticks(yticks, labels=['1/5', 1, 5])
    ax1.set_ylabel(r'$M_{\rm cold} / M_{\rm cold, ini}$', fontsize=lfs)
    ax1.tick_params(axis='x',
                    which='both',
                    labelbottom=False)
    # x axis
    largest_xstop = np.max(xstop_list)
    ax1.set_xlim(0, largest_xstop+0.05)
    # ax1.set_xlabel(r"Time $t / t_{\rm cc}$")
    
    if plot_legend:
        cbar_dims = [0.915, 0.10, 0.02, 0.4]
        ax1.legend(loc='lower right', bbox_to_anchor=(1.17, 0.5), fontsize=5, alignment='left')
    else:
        cbar_dims = [0.915, 0.12, 0.02, 0.75]


    """Warm"""
    ax2.set_ylim(1e-3, 9e0)
    ax2.set_yscale('log')
    ax2.set_ylabel(r'$M_{\rm warm} / M_{\rm cold, ini}$', fontsize=lfs)
    ax2.set_xlabel(r"time $[t_{\rm cc}]$", fontsize=lfs)
    # ax2.set_xticks(np.arange(0, 2, 0.5), np.arange(0, 2, 0.5))
    
    # add colorbar
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 8))
    sm.set_array([])
    cbar_ax = fig.add_axes(cbar_dims)  # [left, bottom, width, height]
    cbar = plt.colorbar(sm, cax=cbar_ax, ax=ax1)#, extend='both')
    cbar.set_ticks([0, 2, 4, 6, 8])
    cbar.set_ticklabels([0, 2, 4, 6, 8])
    cbar.ax.set_ylabel(r'$\log_{10} \frac{R_{\rm cl}}{l_{\rm shatter}}$', rotation=90, labelpad=10, fontsize=lfs)
    ax1.set_title(fr'${{\mathcal{{M}} = {mach}}}$, resolution: ${rp['grid_dim']}^3$', fontsize=tfs)

    plt.subplots_adjust(hspace=0)
    plt.savefig(figpath, format="pdf", bbox_inches="tight")
    plt.show()

"""
Figure 6
----------------------------------------
Mass distributions by scalar value
< and > 0.1 scalars temperature
Shown as a 2D histogram
"""

def tracer_temp_evol_load(figpath = '/ptmp/mpa/wuze/multiphase_turb/figures/tracer_temp_evol.pdf',
                          trial='240711_0.4_16000', scalar_thres = 0.1, nbins_temp = 200):
    
    """read parameters"""
    datapath = f'/freya/ptmp/mpa/wuze/data/{trial}'

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
    bins_temp = np.power(10., np.linspace(np.log10(8e2), np.log10(1e6), nbins_temp))
    temps = [temp_large, temp_small]  # large and small

    imgdata = [[], []]
    for i, temp_switch in enumerate(temps):  # for both larger and smaller
        for temp in temp_switch:  # for each epoch
            hist, _ = np.histogram(temp, bins=bins_temp)
            imgdata[i].append(hist)
        
    return time_athdf, temp_athdf, scalar_athdf, temp_large, temp_small, imgdata


def tracer_temp_evol_plot_onepanel(figpath = '/ptmp/mpa/wuze/multiphase_turb/figures/tracer_temp_evol.pdf',
                                    data = None, trial = '240711_0.4_16000', tcc_lim = 0, vmin = 1e-8,
                                    cmap = 'viridis', lfs = 12, second_axis_snapshots = False, shading = 'auto'):
    # load params
    time_athdf, imgdata, bin_temp_edges = data
    # normalize and conserve
    imgdata = np.array(imgdata) / imgdata[0][0]
    imgdata[imgdata == 0.] = 1e-10

    print(f'trial = {trial}')
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
    tcc_lim_snapshots = int(rp['t_cc'] * tcc_lim / rp['dt_hdf5']) if tcc_lim else len(bins_time)

    img = \
    ax1.pcolormesh(bins_time[:tcc_lim_snapshots], bin_temp_edges[:-1], imgdata[:tcc_lim_snapshots].T[:-1],  # remove the upper temperature bin
                   cmap=cmap, alpha=0.8,
                   norm=LogNorm(vmin=vmin, vmax=1),  #Normalize
                   shading=shading, rasterized=True)


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
        t_cc_labels = np.linspace(0, ax2.get_xlim()[1] * rp['t_cc'] / rp['dt_hdf5'] // 10 * 10, 1)
        t_cc_ticks = t_cc_labels / rp['t_cc'] * rp['dt_hdf5']
        ax2.set_xticks(t_cc_ticks)
        ax2.set_xticklabels([f'{x:.0f}' for x in t_cc_labels])
        ax2.set_xlabel(r"time [snapshots]", fontsize=lfs)

    ax1.legend()

    # plt.savefig(figpath, format="pdf", bbox_inches="tight")
    plt.show()


# the actual function used in the paper
def tracer_temp_evol_plot(figpath = '/ptmp/mpa/wuze/multiphase_turb/figures/tracer_temp_evol.pdf',
                          trials = ['240716_0.8_4000', '240709_0.6_1600000', '240709_0.6_160000000'],
                          tcc_lim = 0, vmins = [5e-4, 3e-7, 1e-7],
                          cmap = 'viridis', lfs = 12, tfs = 14, second_axis_snapshots = False, shading = 'auto'):
    from matplotlib.ticker import FixedLocator
    from matplotlib.colors import LogNorm, Normalize
    import json

    # make the plot
    fig = plt.figure(figsize=(10, 3), dpi=200)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])  # ratio
    
    ax1 = fig.add_subplot(gs[0])  # warm survive
    ax2 = fig.add_subplot(gs[1], sharey=ax1)  # both survive
    ax3 = fig.add_subplot(gs[2], sharey=ax1)  # cold survive

    regimes = [r'\texttt{cDwS}', r'\texttt{cSwS}', r'\texttt{cSwD}']  # text labels for each regime

    for trial, vmin, ax, regime in zip(trials, vmins, [ax1, ax2, ax3], regimes):
        # load params
        with open(f'/freya/ptmp/mpa/wuze/multiphase_turb/saves/tracer_avg_evol_{trial}.json') as f:
            data = json.load(f)

        time_athdf, imgdata, bin_temp_edges = data
        # normalize and conserve
        imgdata = np.array(imgdata) / imgdata[0][0]
        imgdata[imgdata == 0.] = 1e-10

        print(f'trial = {trial}')
        rp = get_rp(trial=trial)
        # time in units of tcc
        bins_time = np.array(time_athdf)/float(rp['t_cc'])
        # temperature flipped
        bin_temp_edges = bin_temp_edges[:-1]
        bin_temp_edges = bin_temp_edges[:len(imgdata.T)]
        
        # take the first however many
        tcc_lim_snapshots = int(rp['t_cc'] * tcc_lim / rp['dt_hdf5']) if tcc_lim else len(bins_time)

        """make the plot"""
        imgdata_trimmed = imgdata[:tcc_lim_snapshots].T[:-1]
        img = \
        ax.pcolormesh(bins_time[:tcc_lim_snapshots], bin_temp_edges[:-1], imgdata_trimmed,  # remove the upper temperature bin
                    cmap=cmap, alpha=0.8,
                    norm=LogNorm(vmin=vmin, vmax=1),  #Normalize
                    shading=shading, rasterized=True)

        # label the rcls
        rcl_title = int(trial.split('_')[-1])
        coeff, expo = s_n(rcl_title)
        ax.text(0.5, 0.98, fr'${coeff:.0f}\times 10^{{{expo:.0f}}}\ l_{{\rm shatter}}$ ({regime})',
                transform=ax.transAxes, ha='center', va='top', fontsize=lfs)

        """colorbars"""
        cbar_ax = ax.inset_axes([0., 1.005, 1, 0.03])
        cbar = plt.colorbar(img, cax=cbar_ax, orientation='horizontal')
        cbar_ax.xaxis.set_ticks_position('top'); cbar_ax.tick_params(axis='x', which='major', pad=1)

        if ax == ax2:
            cbar.set_label('normalized scalar sum', fontsize=lfs, labelpad=-30)

        """axis ticks"""
        # ax.set_ylabel(r'$\log_{10}(T [\rm{K}])$')
        ax.set_yscale('log')
        if ax in [ax2, ax3]:
            ax.tick_params(axis='y',
                            which='both',
                            labelleft=False)

        # set major ticks for tcc
        t_cc_labels = np.arange(bins_time.min(), bins_time.max(), 1)  # every 1 tccs
        t_cc_ticks = t_cc_labels
        ax.set_xticks(t_cc_ticks)
        ax.set_xticklabels([f'{x:.0f}' for x in t_cc_labels])
        # add minor ticks without labels between 0 and 1
        minor_locator = FixedLocator(np.arange(0, 2, 0.2))
        ax.xaxis.set_minor_locator(minor_locator)

        """Make a secondary axis on snapshots"""
        if second_axis_snapshots:
            ax_alt = ax.twiny()
            ax_alt.set_xlim(ax.get_xlim())
            # set major ticks for snapshots
            t_cc_labels = np.linspace(0, ax_alt.get_xlim()[1] * rp['t_cc'] / rp['dt_hdf5'] // 10 * 10, 1)
            t_cc_ticks = t_cc_labels / rp['t_cc'] * rp['dt_hdf5']
            ax_alt.set_xticks(t_cc_ticks)
            ax_alt.set_xticklabels([f'{x:.0f}' for x in t_cc_labels])
            ax_alt.set_xlabel(r"time [snapshots]", fontsize=lfs)
    
    # all axis
    ax2.set_xlabel(r"time [$t_{\rm cc}$]", fontsize=tfs)
    ax1.set_ylabel(r'$T$ [K]', fontsize=tfs)

    plt.subplots_adjust(wspace=0.02)
    plt.savefig(figpath, format="pdf", bbox_inches="tight")
    plt.show()


"""
Figure 8
----------------------------------------
Cold & warm gas t_grow plot
"""
def tgrow_exp_both(figpath = '/ptmp/mpa/wuze/multiphase_turb/figures/tgrow_exp_both.pdf',
                    cm = 'viridis', pred = True,
                    lfs = 12, tfs = 10):
    """
    Plots the actual t_grow for both cold and warm gas in two panels


    Corresponds to Figure 8 in the paper
    -----
    mach: only plot the points for a certain mach number
    plot_growth: plot the expected growth rates from two-phase simulations (-1 = highest, 0 = no plot)
    """
    from codes.plotting.growth import growth_rate_comp
    df, trial_list, cg_run_list, wg_run_list, cg_err_list, wg_err_list, cg_calc_list, wg_calc_list =\
    growth_rate_comp(csvpath='/freya/ptmp/mpa/wuze/multiphase_turb/saves/cloud_8e2_new.csv')
    # mach
    mach_list = np.array([df.loc[df['trial'] == trial, 'x_mach'].to_numpy() for trial in trial_list]).flatten()
    color = cm(plt.Normalize(0, 1)(mach_list))
    # x-axis, normalized cloud size
    rcl_lshat_list = np.array([df.loc[df['trial'] == trial, 'y'].to_numpy() for trial in trial_list]).flatten()
    
    # make subplots
    fig = plt.figure(figsize=(4, 5), dpi=200)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])  # ratio
    ax1 = fig.add_subplot(gs[0])  # cold
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # warm
    
    for ax, run_list in zip([ax1, ax2], [cg_run_list, wg_run_list]):
        # early and late points
        ax.scatter(rcl_lshat_list, run_list[:, 0],
                    marker='^', s=20, linewidths=1, fc='none', ec=color, alpha=0.5, label='early')
        ax.scatter(rcl_lshat_list, run_list[:, 1],
                    marker='o', s=30, linewidths=1, fc=color, ec='k', alpha=0.8, label='late')
        
        # axis properties
        ax.set_xscale('log')
        ax.set_yscale('log')

    # predicted growths
    if pred:
        def calc_tgrow(rcl_lshat, mach,
                        alpha=0.5, T=800, T_hot=4e6,
                        t_cool_min = 0.0001217175017901, l_shat = 2.4989679118099205e-6, box_rad=50):
            # calculate t_cc
            from codes.funcs import calc_cs
            vturb = mach * calc_cs(T_hot)
            chi = T_hot / T
            t_cc = (chi ** (1/2) * rcl_lshat * l_shat / vturb)
            # calculate tgrow
            tgrow = alpha * chi
            tgrow *= (t_cool_min * t_cc) ** (1/2)
            tgrow *= (box_rad) ** (1/6)
            return tgrow / t_cc # calculated
        
        l_lin = np.logspace(1, 9)
        mach_lin = [0.4, 0.6, 0.8]
        predline_colors = cm(plt.Normalize(0, 1)(mach_lin))
        # plot the predicted lines
        for i, mach in enumerate(mach_lin):
            tgrow_pred_cg = calc_tgrow(l_lin, mach, T=800)
            tgrow_pred_wg = calc_tgrow(l_lin, mach, T=8000)
            ax1.plot(l_lin, tgrow_pred_cg, lw=1, ls=':', color=predline_colors[i], alpha=0.8, label='' if i else 'analytic')
            ax2.plot(l_lin, tgrow_pred_wg, lw=1, ls=':', color=predline_colors[i], alpha=0.8)

    # cold
    ax1.set_ylim(3e-1, 1e2)
    ax1.set_ylabel(r'$\frac{t_{\rm grow, exp}}{t_{\rm cc}}$ (cold)', fontsize=lfs)
    ax1.legend(loc='upper right', fontsize=tfs)
    # warm
    ax2.set_ylim(1e-1, 2e1)
    ax2.set_ylabel(r'$\frac{t_{\rm grow, exp}}{t_{\rm cc}}$ (warm)', fontsize=lfs)

    ax2.set_xlabel(r'$\frac{R_{\rm cl}}{l_{\rm shatter}}$', fontsize=lfs)

    # add colorbar
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar_dims = [0.915, 0.12, 0.02, 0.75]  # for being twice as tall as usual
    cbar_ax = fig.add_axes(cbar_dims)  # [left, bottom, width, height]
    cbar = plt.colorbar(sm, cax=cbar_ax, ax=ax1)
    cbar_ticks = np.arange(0, 1.2, 0.2)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f'{cbar_tick:.1f}' for cbar_tick in cbar_ticks])
    cbar.ax.set_ylabel(r'$\mathcal{M}$', rotation=0, labelpad=5, fontsize=lfs)

    plt.subplots_adjust(hspace=0)
    plt.savefig(figpath, format="pdf", bbox_inches="tight")
    plt.show()


"""
Figure 9

Four quadrants plot for the tgrow in warm / cold gas
"""

def tgrow_four_quadrants(figpath = '/ptmp/mpa/wuze/multiphase_turb/figures/tgrow_four_panels.pdf',
                        yraise = 1.2, plot_err = False,
                        cmap = 'viridis', lfs = 12, tfs = 10):
    from codes.plotting.growth import growth_rate_comp
    df, trial_list, cg_run_list, wg_run_list, cg_err_list, wg_err_list, cg_calc_list, wg_calc_list =\
    growth_rate_comp(csvpath='/freya/ptmp/mpa/wuze/multiphase_turb/saves/cloud_8e2_new.csv')

    # colormap
    r_cl_cm = cmap
    rcl_lshat_list = np.array([df.loc[df['trial'] == trial, 'y'].to_numpy() for trial in trial_list]).flatten()
    color = r_cl_cm(plt.Normalize(1, 9)(np.log10(rcl_lshat_list)))

    fig, ax = plt.subplots(dpi=200, figsize=(4,4))
    ax.scatter(1/cg_run_list[:, 0], 1/wg_run_list[:, 0], marker='^', s=20, linewidths=1, fc='none', ec=color, alpha=0.5, label='early')
    ax.scatter(1/cg_run_list[:, 1], 1/wg_run_list[:, 1], marker='o', s=40, linewidths=1, fc=color, ec='k', alpha=1, label='late')

    if plot_err:
        # early error
        ax.errorbar(1/cg_run_list[:, 0], 1/wg_run_list[:, 0],
                    xerr=.1/cg_err_list[:, 0], yerr=.1/wg_err_list[:, 0],
                    markersize=10, ecolor='k', elinewidth=0.5, capsize=0, linewidth=0, color='k', alpha=0.5, zorder=-1)
        # late error
        ax.errorbar(1/cg_run_list[:, 1], 1/wg_run_list[:, 1],  # in units of 0.1 * tcc/MEH
                    xerr=.1/cg_err_list[:, 1], yerr=.1/wg_err_list[:, 1],
                    markersize=10, ecolor='k', elinewidth=0.5, capsize=0, linewidth=0, color='k', alpha=1, zorder=-1)

    # ax.axhline(0, lw=1, ls='--', color='k', alpha=0.5)  # at origin
    # ax.axvline(0, lw=1, ls='--', color='k', alpha=0.5)
    ax.set_xlabel(r'$\frac {t_{\rm cc}} {t_{\rm grow, exp}}$ (cold)', fontsize=lfs)
    ax.set_ylabel(r'$\frac {t_{\rm cc}} {t_{\rm grow, exp}}$ (warm)', fontsize=lfs)

    # limits
    xlim = [-2.6, 2.6]
    ylim = [-3e-1, 6]
    # ax.set_xscale('symlog')
    # ax.set_yscale('symlog')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # shade fill
    x = np.sort(np.concatenate([-np.logspace(-10, np.log10(-xlim[0]), 400), np.logspace(-10, np.log10(xlim[1]), 400)]))
    ax.fill_between(x, yraise, ylim[0], where=(x < 0), ls='-.', lw=1, ec='None', color='red', alpha=0.2, zorder=-1)
    ax.fill_between(x, yraise, ylim[1], where=(x < 0), ls='-.', lw=1, ec='None', color='orange', alpha=0.2, zorder=-1)
    ax.fill_between(x, yraise, ylim[1], where=(x > 0), ls='-.', lw=1, ec='None', color='green', alpha=0.2, zorder=-1)
    ax.fill_between(x, yraise, ylim[0], where=(x > 0), ls='-.', lw=1, ec='None', color='teal', alpha=0.2, zorder=-1)

    # text
    ax.text(xlim[0]/2, (yraise+ylim[0])/2, s=r'\texttt{cDwD}', ha='center', va='center', fontsize=tfs)  # the new labels consistent with text
    ax.text(xlim[0]/2, (yraise+ylim[1])/2, s=r'\texttt{cDwS}', ha='center', va='center', fontsize=tfs)
    ax.text(xlim[1]/2, (yraise+ylim[1])/2, s=r'\texttt{cSwS}', ha='center', va='center', fontsize=tfs)
    ax.text(xlim[1]/2, (yraise+ylim[0])/2, s=r'\texttt{cSwD}', ha='center', va='center', fontsize=tfs)

    # colorbar
    sm = plt.cm.ScalarMappable(cmap=r_cl_cm, norm=plt.Normalize(0, 8))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.915, 0.12, 0.02, 0.75])  # [left, bottom, width, height]

    cbar = plt.colorbar(sm, cax=cbar_ax, ax=ax)
    cbar.set_ticks([0, 2, 4, 6, 8])
    cbar.set_ticklabels([0, 2, 4, 6, 8])
    cbar.ax.set_ylabel(r'$\log_{10} \frac{R_{\rm cl}}{l_{\rm shatter}}$', rotation=90, labelpad=5, fontsize=lfs)

    ax.legend()
    plt.savefig(figpath, format="pdf", bbox_inches="tight")
    plt.show()


"""
Figure 10
----------------------------------------
8e2 runs parameter space: t_cc / tchar & Mach number
Shown in 3 panels
With analytical linear fits

Pickle file generated from plot_params_coolmix_cc()
"""

def params_tcc_tchar_fit_hori(figpath = '/ptmp/mpa/wuze/multiphase_turb/figures/params_tcc_tchar_fit_hori.pdf',
                            pickle_path = f'/freya/ptmp/mpa/wuze/multiphase_turb/saves/cloud_8e2_new_timescales.pkl',
                            log_ylimu = 9, ms = 30, plot_ana = False, show_text = True, fs = 14,
                            cm_cold = None, cm_warm = None):
    # load the trials
    from matplotlib.gridspec import GridSpec
    # initialize the plot
    fig = plt.figure(figsize=(4, 14))
    gs = GridSpec(1, 31, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0:9])
    ax2 = fig.add_subplot(gs[0, 11:20])
    ax3 = fig.add_subplot(gs[0, 22:31])
    # make the markers
    lmarker = MarkerStyle("o")
    rmarker = MarkerStyle("H")

    # laod the pickle
    with open(pickle_path, 'rb') as handle:
        plt_elements = pickle.load(handle)

    # plt_elements has dimension [trial, cww, [x, y, pt_size, log_frac_mean]]
    
    for i, trial_elements in enumerate(plt_elements):
        # per trial
        
        """Plot both cold & hot gas plots"""
        scs = [None, None, None]  # for plotting colorbar
        
        for j, (pt_elements, ax, marker, cm) in \
        enumerate(zip(trial_elements, [ax1, ax2, ax3], [lmarker, rmarker, rmarker], [cm_cold, cm_warm, cm_warm])):
            # load the parameters
            [x, y, pt_size, log_frac_mean] = pt_elements
            
            if j == 1:
                """Warm 1"""
                scs[j] = ax.scatter(x, y, marker=marker, s=pt_size,
                                    c=log_frac_mean, vmin=0, vmax=3, ec='k', cmap=cm)#, label='warm 1')
            elif j == 2:
                """Warm 2"""
                scs[j] = ax.scatter(x, y, marker=marker, s=pt_size,
                                    c=log_frac_mean, vmin=0, vmax=3, ec='k', cmap=cm)#, label='warm 2')
                y_ticks = np.arange(-3, 2, 0.75)
                # ax.set_ylim([-3, 1])
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(f'{x:.2f}'.replace('-', '\N{MINUS SIGN}') for x in y_ticks)
            else:
                """Cold"""
                scs[j] = ax.scatter(x, y, marker=marker, s=pt_size,
                                    c=log_frac_mean, vmin=-0.3, vmax=0.3, ec='k', cmap=cm)#, label='cold')
            
            # color label with text
            col = log_frac_mean
            if np.isclose(col, -.5): col = -np.inf  # catchs only the cold values
            col = f'{col:.1f}'.replace('-', '\N{MINUS SIGN}')

            if show_text:
                if j:  # for warm
                    ax.text(x + 0.3 * np.sqrt(pt_size)/ms, y, col, fontsize=6, ha='center', va='center', path_effects=[pe.withStroke(linewidth=0.5, foreground="white")])
                else:  # for cold
                    ax.text(x - 0.3 * np.sqrt(pt_size)/ms, y, col, fontsize=6, ha='center', va='center', path_effects=[pe.withStroke(linewidth=0.5, foreground="white")])

    
    # y axis label for t_cc / t_cool
    ax1.set_ylabel(r'$\log_{10}(t_{\rm cc, wc} / t_{\rm cool, peak})$', fontsize=fs, rotation=90)
    ax2.set_ylabel(r'$\log_{10}(t_{\rm cc, hw} / t_{\rm cool, peak})$', fontsize=fs, rotation=90)
    ax3.set_ylabel(r'$\log_{10}(t_{\rm cc, wc} / t_{\rm grow, wc})$', fontsize=fs, rotation=90)
    
    # axis
    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.set_xlim(0.2, 1)
        # ax.set_ylim(0, 8)
        ax.set_xlabel(r'$\mathcal{M}_{\rm hot,\ turb}$', fontsize=fs)
        # ax.legend(loc='best', frameon=True, fancybox=False, shadow=True)

        if plot_ana:
            ana_x = np.linspace(0, 1.2, 100)
            if i:  # warm
                ax.plot(ana_x, 0.6 * ana_x, color='k', lw=1, ls='-.', alpha=0.5, label='og')
            else:  # cold
                ax.plot(ana_x, 0.6 * ana_x, color='k', lw=1, ls='-.', alpha=0.5, label='og')

    """color bar"""
    # cold
    cbar_ax1 = fig.add_axes([0.905, 0.15, 0.008, 0.7])  # [left, bottom, width, height]
    cbar1 = plt.colorbar(scs[0], cax=cbar_ax1, extend='both')
    # cbar1.set_ticklabels([])
    cbar1.ax.set_xlabel('c', labelpad=10)
    # warm
    cbar_ax2 = fig.add_axes([0.940, 0.15, 0.008, 0.7])  # [left, bottom, width, height]
    cbar2 = plt.colorbar(scs[1], cax=cbar_ax2, extend='max')
    cbar2.ax.set_ylabel(r'$\log_{10} \frac{M_{\rm cold}}{M_{\rm cold, 0}}$ \& warm slope', fontsize=fs, rotation=90, labelpad=5)
    cbar2.ax.set_xlabel('w', labelpad=10)
    
    plt.savefig(figpath, format="pdf", bbox_inches="tight")
    plt.show()


"""
This version is vertical instead of horizontal
"""
def params_tcc_tchar_fit(figpath = '/ptmp/mpa/wuze/multiphase_turb/figures/params_tcc_tchar_fit.pdf',
                         pickle_path = f'/freya/ptmp/mpa/wuze/multiphase_turb/saves/cloud_8e2_new_timescales.pkl',
                         log_ylimu = 9, ms = 30, plot_ana = False, show_text = True, lfs = 14, tfs = 6,
                         cm_cold = None, cm_warm = None):
    # load the trials
    from matplotlib.gridspec import GridSpec
    # initialize the plot
    fig = plt.figure(figsize=(4.2, 12), dpi=200)
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1])  # ratio
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    # make the markers
    lmarker = MarkerStyle("o")
    rmarker = MarkerStyle("H")

    # laod the pickle
    with open(pickle_path, 'rb') as handle:
        plt_elements = pickle.load(handle)

    # plt_elements has dimension [trial, cww, [x, y, pt_size, log_frac_mean]]
    
    for i, trial_elements in enumerate(plt_elements):
        # per trial
        
        """Plot both cold & hot gas plots"""
        scs = [None, None, None]  # for plotting colorbar
        
        for j, (pt_elements, ax, marker, cm) in \
        enumerate(zip(trial_elements, [ax1, ax2, ax3], [lmarker, rmarker, rmarker], [cm_cold, cm_warm, cm_warm])):
            # load the parameters
            [x, y, pt_size, log_frac_mean] = pt_elements
            
            if j == 1:
                """Warm 1"""
                scs[j] = ax.scatter(x, y, marker=marker, s=pt_size,
                                    c=log_frac_mean, vmin=0, vmax=3, ec='k', cmap=cm)#, label='warm 1')
            elif j == 2:
                """Warm 2"""
                # print(log_frac_mean)  # for warm gas, it is the slope of log
                scs[j] = ax.scatter(x, y, marker=marker, s=pt_size,
                                    c=log_frac_mean, vmin=0, vmax=3, ec='k', cmap=cm)#, label='warm 2')
                y_ticks = np.arange(-3, 2, 0.75)
                # ax.set_ylim([-3, 1])
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(f'{x:.2f}'.replace('-', '\N{MINUS SIGN}') for x in y_ticks)
            else:
                """Cold"""
                scs[j] = ax.scatter(x, y, marker=marker, s=pt_size,
                                    c=log_frac_mean, vmin=-0.3, vmax=0.3, ec='k', cmap=cm)#, label='cold')
            
            # color label with text
            col = log_frac_mean
            if np.isclose(col, -.5): col = -np.inf  # catchs only the cold values
            col = f'{col:.1f}'.replace('-', '\N{MINUS SIGN}')

            if show_text:
                if j:  # for warm
                    ax.text(x + 0.3 * np.sqrt(pt_size)/ms, y, col, fontsize=tfs, ha='center', va='center', path_effects=[pe.withStroke(linewidth=0.5, foreground="white")])
                else:  # for cold
                    ax.text(x - 0.3 * np.sqrt(pt_size)/ms, y, col, fontsize=tfs, ha='center', va='center', path_effects=[pe.withStroke(linewidth=0.5, foreground="white")])

    
    # y axis label for t_cc / t_cool
    ax1.set_ylabel(r'$\log_{10}(t_{\rm cc, wc} / t_{\rm cool, peak})$', fontsize=lfs, rotation=90)
    ax2.set_ylabel(r'$\log_{10}(t_{\rm cc, hw} / t_{\rm cool, peak})$', fontsize=lfs, rotation=90)
    ax3.set_ylabel(r'$\log_{10}(t_{\rm cc, wc} / t_{\rm grow, wc})$', fontsize=lfs, rotation=90)
    ax3.set_xlabel(r'$\mathcal{M}_{\rm hot,\ turb}$', fontsize=lfs)
    
    # axis
    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.set_xlim(0.2, 1)
        # ax.set_ylim(0, 8)
        # ax.legend(loc='best', frameon=True, fancybox=False, shadow=True)

        if plot_ana:
            ana_x = np.linspace(0, 1.2, 100)
            if i:  # warm
                ax.plot(ana_x, 0.6 * ana_x, color='k', lw=1, ls='-.', alpha=0.5, label='og')
            else:  # cold
                ax.plot(ana_x, 0.6 * ana_x, color='k', lw=1, ls='-.', alpha=0.5, label='og')

        if i != 2:
            ax.tick_params(axis='x',
                            which='both',
                            labelbottom=False)

    """color bar"""
    # cold
    # cbar_ax1 = fig.add_axes([0.940, 0.55, 0.02, 0.3])  # [left, bottom, width, height]
    cbar_ax1 = ax1.inset_axes([1.05, 0, 0.02, 1])
    cbar1 = plt.colorbar(scs[0], cax=cbar_ax1, extend='both')
    cbar1.ax.set_ylabel(r'$\log_{10} \frac{M_{\rm cold}}{M_{\rm cold, 0}}$', fontsize=lfs, rotation=90, labelpad=5)
    cbar1.ax.set_xlabel('cold', labelpad=20)
    # warm
    # cbar_ax2 = fig.add_axes([0.940, 0.15, 0.02, 0.3])  # [left, bottom, width, height]
    cbar_ax2 = ax3.inset_axes([1.05, 0.5, 0.02, 1])
    cbar2 = plt.colorbar(scs[1], cax=cbar_ax2, extend='max')
    cbar2.ax.set_ylabel(r'$\frac{t_{\rm grow, warm}} {t_{\rm cc}}$', fontsize=lfs, rotation=90, labelpad=5)
    cbar2.ax.set_xlabel('warm', labelpad=10)
    
    plt.subplots_adjust(hspace=0)
    plt.savefig(figpath, format="pdf", bbox_inches="tight")
    plt.show()



"""
Appendix Figure 1
----------------------------------------

Mass percentage evolution plot of the three phases
Shown in 4 panels, each corresponding to the phase parameter space
"""

def app_mass_evol(figpath = '/ptmp/mpa/wuze/multiphase_turb/figures/app_mass_evol.pdf',
                  trials = [],
                  phase_colors = ['royalblue', 'darkorange', 'crimson'],
                  lfs = 10, tfs = 12, second_axis_snapshots = False):
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import FixedLocator
    
    # initialize the plot
    fig = plt.figure(figsize=(12.5, 3), dpi=200)
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 1])  # split into 4 equal parts
    axs = []
    regimes = [r'\texttt{cDwD}', r'\texttt{cDwS}', r'\texttt{cSwS}', r'\texttt{cSwD}']  # text labels for each regime

    # make each of the four plots    
    for i, trial in enumerate(trials):
        # load the data
        print(f'trial = {trial}')
        datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'
        rp = get_rp(trial=trial)

        """get time and masses"""
        # load hst data
        fname = f'{datapath}/cloud/Turb.hst'
        with open(fname, 'r') as file: keys_raw = file.readlines()[1]
        keys = [a.split('=')[1] for a in keys_raw.split()[1:]]
        data = np.loadtxt(fname).T
        dataf = {keys[i]: data[i] for i in range(len(keys))}
        # get time and masses
        time = dataf['time']
        tot_mass = np.sum(get_datamd(fname=f'{datapath}/cloud/Turb.out2.00001.athdf', key='rho', verbose=False)) * ((rp['box_size'] / rp['grid_dim']) ** 3)
        cg_mass = dataf['cold_gas']
        wg_mass = dataf['warm_gas']
        hg_mass = tot_mass - cg_mass - wg_mass

        """process masses"""
        # stop when the hot gas mass reach 10%
        if np.any((hg_mass / tot_mass) < 0.2):  # if the hot gas disappeears
            print('Cropped for hot mass')
            hot_fill_ind = np.argmin(np.abs(hg_mass / tot_mass - 0.05)) - 50
        else:
            hot_fill_ind = -1

        """make the plot"""
        # current axis
        ax = fig.add_subplot(gs[i])
        time_cropped_tcc = time[:hot_fill_ind] / rp['t_cc']
        ax.stackplot(time_cropped_tcc,  # time in tcc
                    cg_mass[:hot_fill_ind] / tot_mass,  # c, w, and h
                    wg_mass[:hot_fill_ind] / tot_mass,
                    hg_mass[:hot_fill_ind] / tot_mass,
                    labels=['cold', 'warm', 'hot'],
                    colors=phase_colors,
                    alpha=0.5)
        ax.set_ylim(0, 1)        

        # set major ticks for tcc
        time_min, time_max = time_cropped_tcc.min(), time_cropped_tcc.max()
        t_cc_labels = np.arange(time_min, time_max, 1)  # every 1 tccs
        t_cc_ticks = t_cc_labels
        ax.set_xlim(time_min, time_max)
        ax.set_xticks(t_cc_ticks)
        ax.set_xticklabels([f'{x:.0f}' for x in t_cc_labels], fontsize=lfs)
        # add minor ticks without labels between 0 and 1
        minor_locator = FixedLocator(np.arange(0, 2, 0.2))
        ax.xaxis.set_minor_locator(minor_locator)

        ax.set_yticks(np.linspace(0, 1, 5) if i == 0 else [], labels=[0, 0.25, 0.5, 0.75, 1] if i == 0 else [])


        # make a secondary axis on snapshots
        if second_axis_snapshots:
            ax_alt = ax.twiny()
            ax_alt.set_xlim(ax.get_xlim())
            t_cc_labels = np.linspace(0, ax_alt.get_xlim()[1] * rp['t_cc'] / rp['dt_hdf5'] // 10 * 10, 1)
            t_cc_ticks = t_cc_labels / rp['t_cc'] * rp['dt_hdf5']
            ax_alt.set_xticks(t_cc_ticks)
            ax_alt.set_xticklabels([f'{x:.0f}' for x in t_cc_labels])
            ax_alt.set_xlabel(r"time [snapshots]", fontsize=lfs)

        # label the rcls
        rcl_title = int(trial.split('_')[-1])
        coeff, expo = s_n(rcl_title)
        ax.text(0.5, 1.01, fr'${coeff:.0f}\times 10^{{{expo:.0f}}}\ l_{{\rm shatter}}$',
                transform=ax.transAxes, ha='center', va='bottom', fontsize=lfs)
        # label with regimes
        ax.text(0.5, 0.98, f'{regimes[i]}',
                transform=ax.transAxes, ha='center', va='top', fontsize=lfs)

        axs.append(ax)
    
    axs[0].set_ylabel('gas mass \%', fontsize=tfs, rotation=90)
    axs[-1].legend(loc='upper left', bbox_to_anchor=[-0.8, -0.08], fontsize=tfs, ncols=3)
    fig.supxlabel(r"time [$t_{\rm cc}$]", fontsize=tfs, y=-0.07)
    
    plt.subplots_adjust(wspace=0)  # share y axis
    plt.savefig(figpath, format="pdf", bbox_inches="tight")
    plt.show()