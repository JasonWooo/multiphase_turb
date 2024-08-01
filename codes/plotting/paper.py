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
Figure 2
----------------------------------------
8e3 runs parameter space: R_cl / l_shatter & Mach number
Shown in 1 panel with the analytical fit
"""

def params_8e3(csvpath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/saves/cloud_8e3.csv',
               cm = None, cg_st_epoch = 0,
               verbose = False):
    # load the trials
    df = pd.read_csv(csvpath, comment='#')
    trials = df['trial'].to_numpy()

    # initialize the plot
    fig, ax = plt.subplots(figsize=(4,4))
    
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

        col = log_frac_mean
        if np.isclose(col, -.5): col = -np.inf  # catchs only the cold values
        col = f'{col:.2f}'.replace('-', '\N{MINUS SIGN}')
        plt.text(x + 0.05, y, col, fontsize=8, ha='left', va='center')
        
    # analytical line
    ana_x = np.linspace(0, 1.2, 100)
    ana_y = ana_x * (t_cool_cold) / t_cool_min * 10. ** (0.6 * ana_x) / 1.5
    ax.plot(ana_x, ana_y, ls='-.', lw=1, color='k', alpha=0.5)
    
    # axis
    ax.set_xlim(0.18, 1.02)
    ax.set_ylim(np.power(10., 0), np.power(10., 5))
    ax.set_yscale('log')
    ax.set_xlabel(r'$\mathcal{M}_{\rm hot,\ turb}$')
    ax.set_ylabel(r'$\frac{R_{\rm cl}}{l_{\rm shatter}}$', rotation=0, labelpad=14)
    ax.legend()
    # ax.grid()

    # add colorbar
    cbar_ax = fig.add_axes([0.915, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = plt.colorbar(sc, cax=cbar_ax, extend='both')
    # cbar1.set_ticklabels([])
    cbar.ax.set_xlabel('cold', ha='left', labelpad=10)
    cbar.ax.set_ylabel(r'$\log_{10} \frac{M_{\rm cold}}{M_{\rm cold, 0}}$', rotation=90, labelpad=3)
    plt.show()
    return trial, x, y, rp, rp['cloud_radius'] / y, log_cold_frac, t_cool_mix, t_cool_min



"""
Figure 3
----------------------------------------
8e2 runs parameter space: R_cl / l_shatter & Mach number
Shown in 1 panel
WITH analytical fits
"""

def params_8e2(pickle_path = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/saves/cloud_8e2_new.pkl',
               T_cold = 800, T_hot = 4e6,
               log_ylimu = 9, ms = 30, plot_ana = False, show_text = True,
               cm_cold = None, cm_warm = None):
    from matplotlib.gridspec import GridSpec

    with open(pickle_path, 'rb') as handle:
        plt_elements = pickle.load(handle)

    # initialize the plot
    fig, ax = plt.subplots(figsize=(4.2, 4))
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
    if plot_ana:
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
        if plot_ana[0]:
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
            ax.plot(ana_x, ana_y, lw=1, ls='-.', color='orangered', alpha=0.5, label='warm 1')
            plt_elements.append([ana_x, ana_y])

        """cold analytical line"""
        if plot_ana[1]:
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
            ax.plot(ana_x, ana_y, lw=1, ls='-.', color='blue', alpha=0.5, label='cold')
        
        """warm analytical line 2"""
        if plot_ana[2]:
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

            ax.plot(ana_x, ana_y, lw=1, ls='-.', color='brown', alpha=0.5, label='warm 2')

    ax.legend(loc='lower left',
              bbox_to_anchor=[1, 0.75, 0.5, 0.1], fontsize=10,
              frameon=False, fancybox=False, edgecolor='darkgray')
    # axis
    # ax.set_xlim(0.2, 1)
    ax.set_xlim(0.28, 0.92)
    ax.set_ylim(np.power(10., 0.5), np.power(10., log_ylimu))
    ax.set_yscale('log')
    ax.set_xlabel(r'$\mathcal{M}_{\rm hot,\ turb}$')
    ax.set_ylabel(r'$\frac{R_{\rm cl}}{l_{\rm shatter}}$', rotation=0, labelpad=14)

    """color bar"""
    # cold
    cbar_ax1 = fig.add_axes([0.915, 0.1, 0.02, 0.6])  # [left, bottom, width, height]
    cbar1 = plt.colorbar(scs[0], cax=cbar_ax1, extend='both')
    # cbar1.set_ticklabels([])
    cbar1.ax.set_xlabel('c', labelpad=10)
    # warm
    cbar_ax2 = fig.add_axes([1.025, 0.1, 0.02, 0.6])  # [left, bottom, width, height]
    cbar2 = plt.colorbar(scs[1], cax=cbar_ax2, extend='max')
    # cbar2.ax.set_ylabel(r'$\log_{10} \frac{M_{\rm phase}}{M_{\rm cold, 0}}$ / slope', rotation=90, labelpad=3)
    cbar2.ax.set_ylabel(r'$\log_{10} \frac{M_{\rm cold}}{M_{\rm cold, 0}}$ \& warm slope', rotation=90, labelpad=3)
    cbar2.ax.set_xlabel('w', labelpad=1)

    # ax.grid()
    plt.show()




"""
Figure 4
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
    temp_data = [[0]*num_epochs for i in range(num_runs)]
    rho_data = [[0]*num_epochs for i in range(num_runs)]

    for i, trial in tqdm(enumerate(trials)):
        # for each cloud size
        datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'

        """Grab t_cc"""
        with open(f'{datapath}/params.pickle', 'rb') as handle:
            rp = pickle.load(handle)
        rcls.append(rp['cloud_radius'])
        tcc = rp['t_cc']
        
        # convert to epochs
        time_code = tcc * np.array(tccs)
        epochs = np.floor_divide(time_code, rp['dt_hdf5']).astype(int)  # floor divide

        for j, epoch in enumerate(epochs):
            fname=f'{datapath}/cloud/Turb.out2.{epoch:05d}.athdf'
            # print(fname)
            rho = get_datamd(fname=fname, verbose=False, key='rho')
            press = get_datamd(fname=fname, verbose=False, key='press')
            temperature = calc_T(press, rho)
            # append the flattened/sliceed temperature
            temp_slice = temperature[int(len(temperature)/2)] #np.sum(temperature, axis=0)
            dens_proj = np.sum(rho, axis=0)
            temp_data[i][j] = temp_slice
            rho_data[i][j] = dens_proj
    
    return num_runs, num_epochs, temp_data, rho_data, rcls

def density_evol_plot(data, tccs, plot_temp = True, cmap = 'viridis',
                      lfs = 16, tfs = 18):
    num_runs, num_epochs, temp_data, rho_data, rcls = data
    # Create the figure and axes
    fig, axes = plt.subplots(num_runs, num_epochs, figsize=(2 * num_epochs, 2 * num_runs))
    
    # normalizations
    norm = mpl.colors.LogNorm(vmin=1e2, vmax=1e7) if plot_temp else mpl.colors.LogNorm(vmin=1e1, vmax=3e3)

    # Define titles for the columns and rows
    column_titles = [fr'{_}$\ t_{{\rm cc}}$' for _ in tccs]
    # R_cl values in pc
    row_titles = rcls #[20, 2000, 200000, 200000000][::-1] #rcls

    # Plot each panel
    for i in range(num_runs):
        for j in range(num_epochs):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')

            # make the plot
            data = temp_data[i][j] if plot_temp else rho_data[i][j]
            img = ax.imshow(data, cmap=cmap, norm=norm)
            if i == 0:
                ax.set_title(column_titles[j], fontsize=lfs)
            if j == 0:
                coeff, expo = s_n(row_titles[i])
                ax.set_ylabel(fr'${coeff:.0f}\times 10^{{{expo:.0f}}}$', fontsize=lfs)

    fig.supylabel(r'$R_{\rm cl} / l_{\rm shatter}$', x=0.05, fontsize=lfs)

    cax = np.array(axes).flatten()[num_epochs * 2 - 1].inset_axes([1.1, -2, 0.08, 4])
    cbar = fig.colorbar(img, cax=cax, orientation='vertical', location='right', pad=0.1, shrink=0.8, aspect=30)
    cbar.set_label('Temperature [K]' if plot_temp else 'Density [u]', fontsize=lfs)

    # Add the main title
    fig.suptitle(r'$T_{\rm cl} = 8\times 10^2\ {\rm K},\ L_{\rm box}/R_{\rm cl} = 50$', fontsize=tfs, x=0.53, y=0.95)
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    # Show the plot
    plt.show()




"""
Figure 5
----------------------------------------
Cold gas mass evolution plot for the Tfloor = 8e2 runs
With a particular Mach number
"""

def mass_evol(csvpath = '/freya/ptmp/mpa/wuze/multiphase_turb/data/saves/cloud_8e3.csv', mach = 0.3, cm = None, alpha = 0.5, verbose = False, plot_growth = False):
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



"""
Figure 6
----------------------------------------
Mass distributions by scalar value
< and > 0.1 scalars temperature
Shown as a 2D histogram
"""

def tracer_temp_evol():
    pass



"""
Figure 7
----------------------------------------
8e2 runs parameter space: t_cc / tchar & Mach number
Shown in 3 panels
With analytical linear fits
"""

def params_tcc_tchar_fit(pickle_path = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/saves/cloud_8e2_new_timescales.pkl',
                         log_ylimu = 9, ms = 30, plot_ana = False, show_text = True, fs = 14,
                         cm_cold = None, cm_warm = None):
    # load the trials
    from matplotlib.gridspec import GridSpec
    # initialize the plot
    fig = plt.figure(figsize=(14, 4))
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
    
    plt.show()