"""

This corresponds to the parameter_func_8e2.ipynb notebook

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
Plot all params in the R_cl / l_shat & Mach space
From the .csv file
"""

def plot_params_all(csvpath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/saves/cloud_8e3.csv', switch_phase='cold'):
    """
    Plot all parameters
    """
    import pandas as pd
    df = pd.read_csv(csvpath, comment='#')

    all_x = df['x_mach'].to_numpy()
    all_y = df['y'].to_numpy()
    if switch_phase == 'cold':
        all_color = df['log_cold_mass'].to_numpy()
    elif switch_phase == 'warm':
        all_color = df['log_warm_mass'].to_numpy()
    else:
        print('switch_phase is not in [cold, warm]')
        return

    
    # set cloud and floor temperature text
    if df['T_cloud'][0] == 8e3:
        text_T_cloud = r'$T_{\rm cloud} = 8\times 10^3$'
    elif df['T_cloud'][0] == 4e4:
        text_T_cloud = r'$T_{\rm cloud} = 4\times 10^4$'
    elif df['T_cloud'][0] == 8e2:
        text_T_cloud = r'$T_{\rm cloud} = 8\times 10^2$'
    
    if df['T_floor'][0] == 8e3:
        text_T_floor = r'$T_{\rm floor} = 8\times 10^3$'
    elif df['T_floor'][0] == 4e4:
        text_T_floor = r'$T_{\rm cloud} = 4\times 10^4$'
    elif df['T_floor'][0] == 8e2:
        text_T_floor = r'$T_{\rm cloud} = 8\times 10^2$'
    text_T_cold = r'$T_{\rm cold} = 1.6\times 10^4$'

    plt.subplots(figsize=(5,4))
    cm = plt.colormaps['bwr_r']
    
    # scale the fraction
    all_color = np.clip(all_color, -1, 1)
    
    # scatter the points
    sc = plt.scatter(all_x, all_y, marker='o',
                     c=all_color, vmin=-0.3, vmax=0.3, ec='k', cmap=cm)
    
    # annotate with its color value
    for i in range(len(all_x)):
        col = all_color[i]
        if col == 1: col = np.inf
        elif col == -1: col = -np.inf
        col = f'{col:.2f}'.replace('-', '\N{MINUS SIGN}')
        plt.text(all_x[i] + 0.05, all_y[i], col, fontsize=8, ha='left', va='center')
    
    
    # label the temperatures
    plt.text(1., 1e0, text_T_cloud, fontsize=10, ha='center', va='center')
    plt.text(1., 5e-1, text_T_floor, fontsize=10, ha='center', va='center')
    plt.text(1., 2e0, text_T_cold, fontsize=10, ha='center', va='center')
    
    # analytical line
    ana_x = np.linspace(0, 1.2, 100)
    ana_y = ana_x * df['t_cool_mix'][0] / df['t_cool_min'][0] * 10. ** (0.6 * ana_x + 0.02)
    plt.plot(ana_x, ana_y, ls='-.', color='k')
    
    # axis
    plt.xlim(0, 1.2)
    # plt.ylim(np.power(10., -0.5), np.power(10., 6.5))
    plt.yscale('log')
    plt.xlabel(r'$\mathcal{M}_{\rm hot,\ turb}$')
    plt.ylabel(r'$\frac{R_{\rm cl}}{l_{\rm shatter}}$', rotation=0)
    
    # color bar
    cbar = plt.colorbar(sc, extend='both')
    if switch_phase == 'cold': cbar.ax.set_ylabel(r'$\log_{10} \frac{M_{\rm cold}}{M_{\rm cold, 0}}$', rotation=90, labelpad=10)
    else: cbar.ax.set_ylabel(r'$\log_{10} \frac{M_{\rm warm}}{M_{\rm cold, 0}}$', rotation=90, labelpad=10)
    plt.legend()
    plt.grid()
    plt.show()

"""
Plot all params in the R_cl / l_shat & Mach space
From /data directory
Rather than from the .csv file
"""

def plot_params_all_file(csvpath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/saves/cloud_8e2_old.csv',
                         cg_st_epoch = 0, log_ylimu = 9, ms = 30,
                         verbose = False):
    # load the trials
    from matplotlib.gridspec import GridSpec
    import pandas as pd
    df = pd.read_csv(csvpath, comment='#')
    trials = df['trial'].to_numpy()
    
    # extra parameters
    stop_time = df['stop_time'].to_numpy()
    grid_dim = df['grid_dim'].to_numpy()

    # set cloud and floor temperature text
    if df['T_cloud'][0] == 8e3:
        text_T_cloud = r'$T_{\rm cloud} = 8\times 10^3$'
    elif df['T_cloud'][0] == 4e4:
        text_T_cloud = r'$T_{\rm cloud} = 4\times 10^4$'
    elif df['T_cloud'][0] == 8e2:
        text_T_cloud = r'$T_{\rm cloud} = 8\times 10^2$'
    
    if df['T_floor'][0] == 8e3:
        text_T_floor = r'$T_{\rm floor} = 8\times 10^3$'
    elif df['T_floor'][0] == 4e4:
        text_T_floor = r'$T_{\rm floor} = 4\times 10^4$'
    elif df['T_floor'][0] == 8e2:
        text_T_floor = r'$T_{\rm floor} = 8\times 10^2$'
    text_T_cold = r'$T_{\rm cold} = 1.6\times 10^4$'
    
    # initialize the plot
    fig = plt.figure(figsize=(10, 4))
    gs = GridSpec(1, 10, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0:4])
    ax2 = fig.add_subplot(gs[0, 5:10])

    
    # make the plot
    cm = plt.colormaps['bwr_r']
    
    for i, trial in enumerate(trials):
        # load the hst file
        datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'
        fname = f'{datapath}/cloud/Turb.hst'
        with open(fname, 'r') as file: keys_raw = file.readlines()[1]
        keys = [a.split('=')[1] for a in keys_raw.split()[1:]]
        fname = f'{datapath}/cloud/Turb.hst'
        data = np.loadtxt(fname).T
        hst_data = {keys[i]: data[i] for i in range(len(keys))}
        
        rp, x, y, t_cool_mix, t_cool_min, _ = add_point(trial=trial, verbose=verbose)
        log_cold_frac, log_warm_frac, _ = add_color(rp=rp, trial=trial, verbose=verbose, cg_st_epoch=cg_st_epoch, return_cropped=False)  # do not crop, but use the 
        # normalize for all fractions
        log_cold_frac = np.nan_to_num(log_cold_frac, posinf=1, neginf=-1)
        log_warm_frac = np.nan_to_num(log_warm_frac, posinf=1, neginf=-1)

        # plotting params from dataframe
        pt_size = grid_dim[i] / 128 * ms
        stop_ind = int(np.ceil(stop_time[i] * rp['t_cc'] / (rp['dt_hdf5'] / 100)))  # find the index to stop
        
        """Plot both cold & hot gas plots"""
        for i, (ax, log_frac) in enumerate(zip([ax1, ax2], [log_cold_frac, log_warm_frac])):
            # find the stable points
            t_cc_lim = 1  # when is the point stable
            stable_ind = int(np.ceil(t_cc_lim * rp['t_cc'] / (rp['dt_hdf5'] / 100)))  # normalize to hst time
            log_frac_stable = log_frac[stable_ind:stop_ind]  # select all points onward, up to stable point
            
            # take the mean and the std
            log_frac_mean = np.clip(np.mean(log_frac_stable), -0.5, 1)
            
            # scatter the points
            sc = ax.scatter(x, y, marker='o', s=pt_size,
                            c=log_frac_mean, vmin=-0.3, vmax=0.3, ec='k', cmap=cm)
            # color label with text
            col = log_frac_mean
            if np.isclose(col, -.5): col = -np.inf  # catchs only the cold values
            col = f'{col:.2f}'.replace('-', '\N{MINUS SIGN}')
            ax.text(x + 0.05, y, col, fontsize=8, ha='left', va='center')
                
            # analytical line
            ana_x = np.linspace(0, 1.2, 100)
            ana_y = ana_x * t_cool_mix / t_cool_min * 10. ** (0.6 * ana_x + 0.02)

    for i, ax in enumerate([ax1, ax2]):
        ax.plot(ana_x, ana_y, ls='-.', color='k')
        ax.text(1, 1e8, 'Warm phase' if i else 'Cold phase', ha='center')
        
        # axis
        ax.set_xlim(0, 1.2)
        ax.set_ylim(np.power(10., -0.5), np.power(10., log_ylimu))
        ax.set_yscale('log')
        ax.set_xlabel(r'$\mathcal{M}_{\rm hot,\ turb}$')
        ax.set_ylabel(r'$\frac{R_{\rm cl}}{l_{\rm shatter}}$', rotation=0)


        start_text_y = 5e-1
        ax.text(1., start_text_y, text_T_floor, fontsize=10, ha='center', va='center')
        ax.text(1., start_text_y * log_ylimu/3, text_T_cloud, fontsize=10, ha='center', va='center')
        ax.text(1., start_text_y * log_ylimu, text_T_cold, fontsize=10, ha='center', va='center')
    
    
        ax.legend()
        ax.grid()

    """color bar"""
    cbar = plt.colorbar(sc, extend='both')
    cbar.ax.set_ylabel(r'$\log_{10} \frac{M_{\rm phase}}{M_{\rm cold, 0}}$', rotation=90, labelpad=10)
    plt.show()

"""
Plot all params in the R_cl / l_shat & Mach space
From /data directory

Everything is in a single plane
Corresponds to Figure 3 in the paper
"""

def plot_params_all_file_s(csvpath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/saves/cloud_8e2_old.csv',
                           cg_st_epoch = 0, log_ylimu = 9, ms = 30, plot_ana = False, real_mach = False, show_text = True,
                           cm_cold = None, cm_warm = None, verbose = False):
    # load the trials
    from matplotlib.gridspec import GridSpec
    import pandas as pd
    df = pd.read_csv(csvpath, comment='#')
    trials = df['trial'].to_numpy()
    
    # extra parameters
    actual_mach = df['x_mach'].to_numpy()
    stop_time = df['stop_time'].to_numpy()
    grid_dim = df['grid_dim'].to_numpy()

    # set cloud and floor temperature text
    if df['T_cloud'][0] == 8e3:
        text_T_cloud = r'$T_{\rm cloud} = 8\times 10^3$'
    elif df['T_cloud'][0] == 4e4:
        text_T_cloud = r'$T_{\rm cloud} = 4\times 10^4$'
    elif df['T_cloud'][0] == 8e2:
        text_T_cloud = r'$T_{\rm cloud} = 8\times 10^2$'
    
    if df['T_floor'][0] == 8e3:
        text_T_floor = r'$T_{\rm floor} = 8\times 10^3$'
    elif df['T_floor'][0] == 4e4:
        text_T_floor = r'$T_{\rm floor} = 4\times 10^4$'
    elif df['T_floor'][0] == 8e2:
        text_T_floor = r'$T_{\rm floor} = 8\times 10^2$'
    text_T_cold = r'$T_{\rm cold} = 1.6\times 10^4$'
    
    # initialize the plot
    fig, ax = plt.subplots(figsize=(4.5, 4))
    plt_elements = [] # empty list for elements in the plot
    xs = [[], []]; ys = [[], []]; pt_sizes = [[], []]; log_frac_means = [[], []]
    # make the markers
    lmarker = MarkerStyle("o", fillstyle="left")
    rmarker = MarkerStyle("o", fillstyle="right")
    
    for i, trial in enumerate(trials):
        # load the hst file
        datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'
        fname = f'{datapath}/cloud/Turb.hst'
        with open(fname, 'r') as file: keys_raw = file.readlines()[1]
        keys = [a.split('=')[1] for a in keys_raw.split()[1:]]
        fname = f'{datapath}/cloud/Turb.hst'
        data = np.loadtxt(fname).T
        hst_data = {keys[i]: data[i] for i in range(len(keys))}
        
        rp, x, y, t_cool_mix, t_cool_min, [t_cool_func, T_tcoolmin, T_mix] = add_point(trial=trial, verbose=verbose)
        # replace x with actual mach
        if real_mach: x = actual_mach[i]
        # calculate turbulent velocity for later t_cc calcs
        vturb = x * calc_cs(rp['T_hot'])
        log_cold_frac, log_warm_frac, hst_time = add_color_time(rp=rp, trial=trial, verbose=verbose, cg_st_epoch=cg_st_epoch, return_cropped=False)  # do not crop, but use the 
        # normalize for all fractions
        log_cold_frac = np.nan_to_num(log_cold_frac, posinf=1, neginf=-1)
        log_warm_frac = np.nan_to_num(log_warm_frac, posinf=1, neginf=-1)

        # plotting params from dataframe
        pt_size = grid_dim[i] / 128 * ms
        stop_ind = int(np.ceil(stop_time[i] * rp['t_cc'] / (rp['dt_hdf5'] / 100)))  # find the index to stop
        
        """Plot both cold & hot gas plots"""
        scs = [None, None]
        for j, (log_frac, marker, cm) in \
        enumerate(zip([log_cold_frac, log_warm_frac], [lmarker, rmarker], [cm_cold, cm_warm])):
            # find the stable points
            t_cc_lim = 1  # when is the point stable
            stable_ind = int(np.ceil(t_cc_lim * rp['t_cc'] / (rp['dt_hdf5'] / 100)))  # normalize to hst time
            log_frac_stable = log_frac[stable_ind:stop_ind]  # select all points onward, up to stable point
            
            # take the mean and the std
            if j:  # warm, make linear
                # log_frac_mean = np.nansum(np.diff(log_frac_stable) ** 3) ** (1/3)  # make it the slope # log_frac_mean = np.power(10., np.mean(log_frac_stable))
                log_frac_mean, _ = np.polyfit(hst_time[stable_ind:stop_ind], log_frac_stable, 1)
                # scatter the points
                scs[j] = ax.scatter(x, y, marker=marker, s=pt_size,
                                    c=log_frac_mean, vmin=-1, vmax=3, ec='k', cmap=cm)
            else:  # cold
                log_frac_mean = np.clip(np.mean(log_frac_stable), -0.5, 1)
                # scatter the points
                scs[j] = ax.scatter(x, y, marker=marker, s=pt_size,
                                    c=log_frac_mean, vmin=-0.3, vmax=0.3, ec='k', cmap=cm)

            xs[j].append(x)
            ys[j].append(y)
            pt_sizes[j].append(pt_size)
            log_frac_means[j].append(log_frac_mean)
            
            # color label with text
            col = log_frac_mean
            if np.isclose(col, -.5): col = -np.inf  # catchs only the cold values
            col = f'{col:.1f}'.replace('-', '\N{MINUS SIGN}')

            if show_text:
                if j:  # for warm
                    ax.text(x + 0.3 * np.sqrt(pt_size)/ms, y, col, fontsize=6, ha='center', va='center', path_effects=[pe.withStroke(linewidth=1, foreground="white")])
                else:  # for cold
                    ax.text(x - 0.3 * np.sqrt(pt_size)/ms, y, col, fontsize=6, ha='center', va='center', path_effects=[pe.withStroke(linewidth=1, foreground="white")])

    # append for all trials
    for i in range(2):
        plt_elements.append([xs[i], ys[i], pt_sizes[i], log_frac_means[i]])

    """
    –––––––––––
    Plot the analytical lines according to theoretical predictions
    
    """
    if plot_ana:
        ana_x = np.linspace(0, 1.2, 100)
        # original analytical line
        # ana_y = ana_x * t_cool_mix / t_cool_min * 10. ** (0.6 * ana_x + 0.02) / 5
        # ax.plot(ana_x, ana_y, ls='-.', color='k', alpha=0.5, label='OG ana')
    
        """warm analytical line 1"""  #  t_cc_hw / t_cool_peak
        if plot_ana[0]:
            # temperatures
            T_cold = rp['T_cloud']
            T_peak = 8.57e+03
            T_warm = 8e3
            T_hot = rp['T_hot']
            omega = T_cold / T_tcoolmin
            
            # timescale @ temperature of interest: where t_cool peaks, aka T_cold for 8000K runs
            t_cool_peak = t_cool_func(T_peak)

            # recalculate cooling function from hot to warm gas
            # ratio of t_cc(hot/cold) to t_cc(hot/warm), to get back to the original
            ratio_tcc = np.sqrt(T_warm / T_cold)  # warm / cold
            
            ana_y = ana_x * t_cool_peak / t_cool_min * ratio_tcc * (10. ** (0.6 * ana_x)) * np.sqrt(omega)
            ax.plot(ana_x, ana_y, lw=1, ls='-.', color='orange', alpha=0.5, label='warm ana 1')
            plt_elements.append([ana_x, ana_y])

        """cold analytical line"""  #  t_cc_wc / t_cool_peak
        if plot_ana[1]:
            # temperatures
            T_cold = rp['T_cloud']
            T_peak = 8.57e+03
            T_warm = 8e3
            T_hot = rp['T_hot']
            omega = T_cold / T_tcoolmin
            
            # timescale @ temperature of interest: where t_cool peaks, aka T_cold for 8000K runs
            t_cool_peak = t_cool_func(T_peak)

            # recalculate cooling function from warm to cold gas
            # ratio of t_cc(hot/cold) to t_cc(warm/cold), to get back to the original            
            ratio_tcc = np.sqrt(T_hot / T_warm)  # hot / warm
            
            ana_y = ana_x * t_cool_peak / t_cool_min * ratio_tcc * (10. ** (0.6 * ana_x)) * np.sqrt(omega)
            ax.plot(ana_x, ana_y, lw=1, ls='-.', color='blue', alpha=0.5, label='cold ana')
            plt_elements.append([ana_x, ana_y])
        
        """warm analytical line 2"""  #  t_cc_wc / t_grow_wc
        if plot_ana[2]:
            # temperatures
            T_cold = rp['T_cloud']
            T_peak = 8.57e+03
            T_warm = 8e3
            T_hot = rp['T_hot']
            omega = T_cold / T_tcoolmin

            # timescale of interest: growth time for the warm & cold gas
            chi_wc = T_warm / T_cold
            l_shat = 2.4989679118099205e-06
            cloud_box_ratio = 1/50
            cs_hot = calc_cs(T_hot)
            # t_cool of minimum temperature (800K)
            t_cool_low = t_cool_func(T_cold)
            
            ana_y = ana_x *\
            (cs_hot * l_shat / t_cool_low) ** (2/3)
            (10. ** (0.6 * ana_x))**(2/3) *\
            (cloud_box_ratio**(1/9)) *\
            (chi_wc**(1/3)) *\
            np.sqrt(omega)
            ax.plot(ana_x, ana_y, lw=1, ls='-.', color='brown', alpha=0.5, label='warm ana 2')
            plt_elements.append([ana_x, ana_y])

    # save everything as a pickle
    import pickle
    with open(f'{csvpath.split('.')[0]}.pkl', 'wb') as handle:
        pickle.dump(plt_elements, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # axis
    ax.set_xlim(0.2, 1)
    ax.set_ylim(np.power(10., -0.5), np.power(10., log_ylimu))
    ax.set_yscale('log')
    ax.set_xlabel(r'$\mathcal{M}_{\rm hot,\ turb}$')
    ax.set_ylabel(r'$\frac{R_{\rm cl}}{l_{\rm shatter}}$', rotation=0)


    # start_text_y = 5e-1
    # ax.text(1., start_text_y, text_T_floor, fontsize=10, ha='center', va='center')
    # ax.text(1., start_text_y * log_ylimu/3, text_T_cloud, fontsize=10, ha='center', va='center')
    # ax.text(1., start_text_y * log_ylimu, text_T_cold, fontsize=10, ha='center', va='center')


    ax.legend(loc='best')

    """color bar"""
    # cold
    cbar_ax1 = fig.add_axes([0.925, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar1 = plt.colorbar(scs[0], cax=cbar_ax1, extend='both')
    # cbar1.set_ticklabels([])
    cbar1.ax.set_xlabel('c', labelpad=10)
    # warm
    cbar_ax2 = fig.add_axes([1.025, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar2 = plt.colorbar(scs[1], cax=cbar_ax2, extend='max')
    cbar2.ax.set_ylabel(r'$\log_{10} \frac{M_{\rm phase}}{M_{\rm cold, 0}}$', rotation=90, labelpad=10)
    cbar2.ax.set_xlabel('w', labelpad=1)
    
    plt.show()

"""
^^^ Plots the above from the saved .pkl file
"""

def plot_params_all_file_save(pickle_path = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/saves/cloud_8e2_new.pkl',
                              log_ylimu = 9, ms = 30, plot_ana = False, show_text = True,
                              cm_cold = None, cm_warm = None, verbose = False):
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
        T_hot = 4e6
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
            T_cold = rp['T_cloud']
            T_peak = 8.57e+03
            T_warm = 8e3
            T_hot = rp['T_hot']
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
            T_cold = rp['T_cloud']
            T_peak = 8.57e+03
            T_warm = 8e3
            T_hot = rp['T_hot']
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
            T_cold = rp['T_cloud']
            T_peak = 8.57e+03
            T_warm = 8e3
            T_hot = rp['T_hot']
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

            print(cs_hot)
            ax.plot(ana_x, ana_y, lw=1, ls='-.', color='brown', alpha=0.5, label='warm 2')
            print(ana_y)

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
Plots the relative timescales
"""

def plot_timescale_mach(pickle_path = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/saves/cloud_8e2_new.pkl',
                        ms = 30, plot_ana = False, mach = 0.8, pt_y = 1e4,
                        cm_cold = None, cm_warm = None, verbose = False):
    from matplotlib.gridspec import GridSpec
    import pandas as pd

    with open(pickle_path, 'rb') as handle:
        plt_elements = pickle.load(handle)

    # initialize the plot
    fig, ax = plt.subplots(figsize=(4.5, 4))
    # make the markers
    lmarker = MarkerStyle("o", fillstyle="left")
    rmarker = MarkerStyle("o", fillstyle="right")
    
    """Plot both cold & hot gas plots from the pickle file"""
    i = 0
    ylist = []
    scs = [None, None]
    for marker, cm in \
    zip([lmarker, rmarker], [cm_cold, cm_warm]):
        [xs, ys, pt_sizes, log_frac_means] = plt_elements[i]
        for x, y, pt_size, log_frac_mean in zip(xs, ys, pt_sizes, log_frac_means):
            # plot only if within mach range
            if (x < mach+0.1) and (x > mach-0.1):
                # plot the point
                if i:  # warm, make linear
                    scs[i] = ax.scatter(y, pt_y, marker=marker, s=pt_size,
                                        c=log_frac_mean, vmin=0, vmax=1, ec='k', cmap=cm)
                else:  # cold, colorbar in log
                    scs[i] = ax.scatter(y, pt_y, marker=marker, s=pt_size,
                                        c=log_frac_mean, vmin=-0.3, vmax=0.3, ec='k', cmap=cm)

                ylist.append(y)
        i += 1  # update the savescounter

    """Plot the analytic line"""
    if plot_ana:
        """warm analytical line 1"""
        if plot_ana[0]:
            ax.axvline(1e3, lw=2, ls='-', color='green', alpha=0.5, label='warm ana 1')

        """cold analytical line"""
        if plot_ana[1]:
            ax.axvline(1e4, lw=2, ls='-', color='blue', alpha=0.5, label='cold ana')
        
        """warm analytical line 2"""
        if plot_ana[2]:
            ax.axvline(2e6, lw=2, ls='-', color='red', alpha=0.5, label='warm ana 2')

    """Plot the timescales"""
    # load the csv file aand the trial names
    csvpath = f'{pickle_path.split('.')[0]}.csv'
    df = pd.read_csv(csvpath, comment='#')
    trials = df['trial'].to_numpy()

    # loop through the trials
    t_cc_hc, t_cc_wc, t_cc_hw = [], [], []
    t_grow_hc, t_grow_wc, t_grow_hw = [], [], []
    t_cool_wc, t_cool_hw = [], []
    
    for i, trial_mach in enumerate(df['x_mach'].to_numpy()):
        if (trial_mach < mach+0.1) and (trial_mach > mach-0.1):  # same criteria
            trial = trials[i]
            datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'
            with open(f'{datapath}/params.pickle', 'rb') as handle:
                rp = pickle.load(handle)
            # cooling function
            import sys
            sys.path.append(os.path.abspath('/freya/ptmp/mpa/wuze/multiphase_turb/athena/cooling_scripts'))
            import cooling_fn as cf
            t_cool_func = lambda T : cf.tcool_calc(T_hot * 0.1 / T, T, Zsol=1.0, Lambda_fac=1.0, fit_type="max")
                
            # calculate t_cc of warm/cold gas
            T_cold = rp['T_cloud']
            T_hot = rp['T_hot']
            T_warm = 8000#np.sqrt(T_cold * T_hot)
            T_mix = np.sqrt(T_cold * T_hot)
            
            # calculate other quantities
            chi_hc = T_hot / T_cold
            chi_wc = T_warm / T_cold
            chi_hw = T_hot / T_warm
            R_cl = rp['cloud_radius']
            cloud_box_ratio = 1/50
            
            # load other values
            t_cool_min = 0.0001217175017901
            l_shat = 2.4989679118099205e-06
            vturb = rp['mach'] * calc_cs(T_hot)
            
            # calculate the four timescales
            t_cc_hc.append( (chi_hc ** (1/2) * R_cl / vturb) )
            t_cc_wc.append( (chi_wc ** (1/2) * R_cl / vturb) )
            t_cc_hw.append( (chi_hw ** (1/2) * R_cl / vturb) )
            
            t_grow_hc.append( chi_hc * rp['mach'] ** (-1/2) * (R_cl / l_shat)**(1/2) * cloud_box_ratio**(-1/6) * t_cool_func(T_warm) )
            t_grow_wc.append( chi_wc * rp['mach'] ** (-1/2) * (R_cl / l_shat)**(1/2) * cloud_box_ratio**(-1/6) * t_cool_func(T_cold) )
            t_grow_hw.append( chi_hw * rp['mach'] ** (-1/2) * (R_cl / l_shat)**(1/2) * cloud_box_ratio**(-1/6) * t_cool_min )

            # the cooling functions are altered
            t_cool_wc.append(t_cool_func(800) / 0.6 * mach)
            t_cool_hw.append(t_cool_func(8000) / 0.6 * mach)  # the cooling function limit for warm and cold
            
            # all the relevant timescales
            if verbose:
                print(f't_cc_hc   = {t_cc_hc:>10.3f}; rate = {1/t_cc_hc:>10.3e}')
                print(f't_cc_wc   = {t_cc_wc:>10.3f}; rate = {1/t_cc_wc:>10.3e}')
                print(f't_cc_hw   = {t_cc_hw:>10.3f}; rate = {1/t_cc_hw:>10.3e}')
                print(f't_grow_hc = {t_grow_hc:>10.3f}; rate = {1/t_grow_hc:>10.3e}')
                print(f't_grow_wc = {t_grow_wc:>10.3f}; rate = {1/t_grow_wc:>10.3e}')
                print(f't_grow_hw = {t_grow_hw:>10.3f}; rate = {1/t_grow_hw:>10.3e}')
                print(f't_cool_cloud = {t_cool_func(800):>10.3f}; rate = {1/t_cool_func(800):>10.3e}')
                print(f't_cool_peak = {t_cool_func(8000):>10.5f}; rate = {1/t_cool_func(8000):>10.3e}')

    ylist = ylist[:int(len(ylist)/2)]
    ylist_sorted = np.sort(ylist)
    """Plot the timescales"""
    ax.plot(ylist_sorted, sort_parallel(t_cc_hc, ylist), lw=1, ls='--', alpha=0.5, color='purple', label=r'$t_{\rm cc, hc}$')
    ax.plot(ylist_sorted, sort_parallel(t_cc_wc, ylist), lw=1, ls='--', alpha=0.5, color='green', label=r'$t_{\rm cc, wc}$')
    ax.plot(ylist_sorted, sort_parallel(t_cc_hw, ylist), lw=1, ls='--', alpha=0.5, color='red', label=r'$t_{\rm cc, hw}$')
    ax.plot(ylist_sorted, sort_parallel(t_grow_hc, ylist), lw=1, ls=':', alpha=0.5, color='purple', label=r'$t_{\rm grow, hc}$')
    ax.plot(ylist_sorted, sort_parallel(t_grow_wc, ylist), lw=1, ls=':', alpha=0.5, color='green', label=r'$t_{\rm grow, wc}$')
    ax.plot(ylist_sorted, sort_parallel(t_grow_hw, ylist), lw=1, ls=':', alpha=0.5, color='red', label=r'$t_{\rm grow, hw}$')
    ax.plot(ylist_sorted, sort_parallel(t_cool_wc, ylist), lw=1, ls='-.', alpha=0.5, color='green', label=r'$t_{\rm cool, cloud}$')
    ax.plot(ylist_sorted, sort_parallel(t_cool_hw, ylist), lw=1, ls='-.', alpha=0.5, color='red', label=r'$t_{\rm cool, peak}$')

    
    ax.legend(loc='best', bbox_to_anchor=[0.9, 0.55, 0.5, 0.5], fontsize=10)
    # axis
    ax.set_xlim(np.power(10., 1), np.power(10., 8.5))
    ax.set_xscale('log')
    # ax.set_ylim(1e-4, 1e5)
    ax.set_yscale('log')
    ax.set_xlabel(r'$\frac{R_{\rm cl}}{l_{\rm shatter}}$')
    ax.set_ylabel('Timescales')

    """color bar"""
    # cold
    cbar_ax1 = fig.add_axes([0.925, 0.05, 0.02, 0.3])  # [left, bottom, width, height]
    cbar1 = plt.colorbar(scs[0], cax=cbar_ax1, extend='both')
    # cbar1.set_ticklabels([])
    cbar1.ax.set_xlabel('c', labelpad=10)
    # warm
    cbar_ax2 = fig.add_axes([1.025, 0.05, 0.02, 0.3])  # [left, bottom, width, height]
    cbar2 = plt.colorbar(scs[1], cax=cbar_ax2, extend='max')
    cbar2.ax.set_ylabel(r'$\log_{10} \frac{M_{\rm phase}}{M_{\rm cold, 0}}$', rotation=90, labelpad=10)
    cbar2.ax.set_xlabel('w', labelpad=1)
    
    plt.show()

"""
Plot all params in the tcc / tchar & Mach space
Produces 7 panels

Corresponds to Figure 7 in the paper
"""

def plot_params_coolmix_cc(csvpath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/saves/cloud_8e2_old.csv',
                           cg_st_epoch = 0, log_ylimu = 9, ms = 30,
                           real_mach = False, plot_ana = False, show_text = True,
                           cm_cold = None, cm_warm = None, verbose = False):
    # load the trials
    from matplotlib.gridspec import GridSpec
    import pandas as pd
    df = pd.read_csv(csvpath, comment='#')
    trials = df['trial'].to_numpy()
    
    # extra parameters
    actual_mach = df['x_mach'].to_numpy()
    stop_time = df['stop_time'].to_numpy()
    grid_dim = df['grid_dim'].to_numpy()
    l_shat = df['l_shat'].to_numpy()
    
    # initialize the plot
    fig = plt.figure(figsize=(14, 4))
    gs = GridSpec(1, 31, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0:9])
    ax2 = fig.add_subplot(gs[0, 11:20])
    ax3 = fig.add_subplot(gs[0, 22:31])
    # make the markers
    lmarker = MarkerStyle("o")
    rmarker = MarkerStyle("H")

    all_vals = []
    for i, trial in enumerate(trials):
        # load the hst file
        datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'
        fname = f'{datapath}/cloud/Turb.hst'
        with open(fname, 'r') as file: keys_raw = file.readlines()[1]
        keys = [a.split('=')[1] for a in keys_raw.split()[1:]]
        fname = f'{datapath}/cloud/Turb.hst'
        data = np.loadtxt(fname).T
        hst_data = {keys[i]: data[i] for i in range(len(keys))}

        """Get the data"""
        rp, x, y, t_cool_mix, t_cool_min, [t_cool_func, T_tcoolmin, T_mix] = add_point(trial=trial, verbose=verbose)

        # replace x with actual mach
        if real_mach: x = actual_mach[i]
        log_cold_frac, log_warm_frac, hst_time = add_color_time(rp=rp, trial=trial, verbose=verbose, cg_st_epoch=cg_st_epoch, return_cropped=False)  # do not crop, but use the 

        # normalize for all fractions
        log_cold_frac = np.nan_to_num(log_cold_frac, posinf=1, neginf=-1)
        log_warm_frac = np.nan_to_num(log_warm_frac, posinf=1, neginf=-1)

        # plotting params from dataframe
        pt_size = grid_dim[i] / 128 * ms
        stop_ind = int(np.ceil(stop_time[i] * rp['t_cc'] / (rp['dt_hdf5'] / 100)))  # find the index to stop

        all_vals_trial = []
        """Plot both cold & hot gas plots"""
        scs = [None, None, None]
        for j, (log_frac, marker, cm, ax) in \
        enumerate(zip([log_cold_frac, log_warm_frac, log_warm_frac],
                      [lmarker, rmarker, rmarker],
                      [cm_cold, cm_warm, cm_warm],
                      [ax1, ax2, ax3])):
            # replace y with log10(t_cc / t_cool,mix)
            vturb = x * calc_cs(rp['T_hot'])
            t_cc = (rp['chi'] ** (1/2) * rp['cloud_radius'] / vturb)

            # find the stable points
            t_cc_lim = 1  # when is the point stable
            stable_ind = int(np.ceil(t_cc_lim * rp['t_cc'] / (rp['dt_hdf5'] / 100)))  # normalize to hst time
            log_frac_stable = log_frac[stable_ind:stop_ind]  # select all points onward, up to stable point
            # take the mean as color value

            """scaling"""
            T_peak = 8000
            
            if j == 1:
                """Warm 1"""
                T_cold = rp['T_cloud']
                T_peak = 8.57e+03
                T_warm = 8e3
                T_hot = rp['T_hot']
                
                # calculate a different t_cc (hot / warm)
                t_cc_hw = (T_hot / T_warm) ** (1/2) * rp['cloud_radius'] / vturb
                t_cool_peak = t_cool_func(T_peak)  # temperature of interest
                y = np.log10(t_cc_hw / t_cool_peak)

                # make color linear
                log_frac_mean = np.power(10., np.mean(log_frac_stable))
                # log_frac_mean = np.mean(np.diff(log_frac_stable))#np.power(10., np.mean(log_frac_stable))
                # log_frac_mean, _ = np.polyfit(hst_time[stable_ind:stop_ind], log_frac_stable, 1)
                # scatter the points
                scs[j] = ax.scatter(x, y, marker=marker, s=pt_size,
                                    c=log_frac_mean, vmin=0, vmax=1, ec='k', cmap=cm)#, label='warm')
            elif j == 2:
                """Warm 2"""
                T_cold = rp['T_cloud']
                T_peak = 8.57e+03
                T_warm = 8e3
                T_hot = rp['T_hot']
                
                # calculate t_cc for warm and cold
                t_cc_wc = (T_warm / T_cold) ** (1/2) * rp['cloud_radius'] / vturb
                # calculate growth time for warm gas in hot
                chi_wc = T_warm / T_cold
                R_cl = rp['cloud_radius']
                l_shat_trial = l_shat[i]
                cloud_box_ratio = 1/50
                t_grow_wc = chi_wc * rp['mach'] ** (-1/2) * (R_cl / l_shat_trial)**(1/2) * cloud_box_ratio**(-1/6) * t_cool_func(T_cold)

                # calculate the new y
                y = np.log10(t_cc_wc / t_grow_wc)

                # make color linear
                log_frac_mean = np.power(10., np.mean(log_frac_stable))
                # log_frac_mean = np.mean(np.diff(log_frac_stable))#np.power(10., np.mean(log_frac_stable))
                # log_frac_mean, _ = np.polyfit(hst_time[stable_ind:stop_ind], log_frac_stable, 1)
                # scatter the points
                scs[j] = ax.scatter(x, y, marker=marker, s=pt_size,
                                    c=log_frac_mean, vmin=0, vmax=1, ec='k', cmap=cm)#, label='not final')
            else:
                """Cold"""
                T_cold = rp['T_cloud']
                T_peak = 8.57e+03
                T_warm = 8e3
                T_hot = rp['T_hot']
                
                # calculate a different t_cc (warm / cold)
                t_cc_wc = (T_warm / T_cold) ** (1/2) * rp['cloud_radius'] / vturb
                t_cool_peak = t_cool_func(T_peak)  # temperature of interest
                
                y = np.log10(t_cc_wc / t_cool_peak)
                log_frac_mean = np.clip(np.mean(log_frac_stable), -0.5, 1)
                # scatter the point
                scs[j] = ax.scatter(x, y, marker=marker, s=pt_size,
                                    c=log_frac_mean, vmin=-0.3, vmax=0.3, ec='k', cmap=cm)#, label='cold')

            # append the values
            all_vals_trial.append([x, y, pt_size, log_frac_mean])
            
            # color label with text
            col = log_frac_mean
            if np.isclose(col, -.5): col = -np.inf  # catchs only the cold values
            col = f'{col:.1f}'.replace('-', '\N{MINUS SIGN}')

            if show_text:
                if j:  # for warm
                    ax.text(x + 0.3 * np.sqrt(pt_size)/ms, y, col, fontsize=6, ha='center', va='center', path_effects=[pe.withStroke(linewidth=0.5, foreground="white")])
                else:  # for cold
                    ax.text(x - 0.3 * np.sqrt(pt_size)/ms, y, col, fontsize=6, ha='center', va='center', path_effects=[pe.withStroke(linewidth=0.5, foreground="white")])

        # append all values in a trial
        all_vals.append(all_vals_trial)
    # save the values as a pickle
    import pickle
    with open(f'{csvpath.split('.')[0]}_timescales.pkl', 'wb') as handle:
        pickle.dump(all_vals, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # y axis label for t_cc / t_cool
    ax1.set_ylabel(r'$\log_{10}(t_{\rm cc, wc} / t_{\rm cool, peak})$', rotation=90)
    ax2.set_ylabel(r'$\log_{10}(t_{\rm cc, hw} / t_{\rm cool, peak})$', rotation=90)
    ax3.set_ylabel(r'$\log_{10}(t_{\rm cc, wc} / t_{\rm grow, wc})$', rotation=90)
    # axis
    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.set_xlim(0.2, 1)
        # ax.set_ylim(0, 8)
        ax.set_xlabel(r'$\mathcal{M}_{\rm hot,\ turb}$')
        # ax.legend(loc='best', frameon=True, fancybox=False, shadow=True)

        if plot_ana:
            ana_x = np.linspace(0, 1.2, 100)
            if i:  # warm
                ax.plot(ana_x, 0.6 * ana_x, color='gray', ls='-.', alpha=0.5, )
            else:  # cold
                ax.plot(ana_x, 0.6 * ana_x, color='gray', ls='-.', alpha=0.5, label='og')

    """color bar"""
    # cold
    cbar_ax1 = fig.add_axes([0.915, 0.15, 0.005, 0.7])  # [left, bottom, width, height]
    cbar1 = plt.colorbar(scs[0], cax=cbar_ax1, extend='both')
    # cbar1.set_ticklabels([])
    cbar1.ax.set_xlabel('c', labelpad=10)
    # warm
    cbar_ax2 = fig.add_axes([0.945, 0.15, 0.005, 0.7])  # [left, bottom, width, height]
    cbar2 = plt.colorbar(scs[1], cax=cbar_ax2, extend='max')
    cbar2.ax.set_ylabel(r'$\log_{10} \frac{M_{\rm phase}}{M_{\rm cold, 0}}$', rotation=90, labelpad=10)
    cbar2.ax.set_xlabel('w', labelpad=1)
    
    plt.show()

"""
^^^ Plots the above from the saved .pkl file
"""

def plot_params_coolmix_cc_save(pickle_path = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/saves/cloud_8e2_new_timescales.pkl',
                                log_ylimu = 9, ms = 30, plot_ana = False, show_text = True,
                                cm_cold = None, cm_warm = None, verbose = False):
    fs = 14
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