import numpy as np
import pandas
import pickle
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath('/freya/ptmp/mpa/wuze/athena_pp/vis/python'))
from athena_read import athdf
import h5py

"""Define constants"""

class unit():
    def __init__(self):        
        # length, time, and mass constants
        self.CONST_pc  = 3.086e18
        self.CONST_yr  = 3.154e7
        self.CONST_amu = 1.66053886e-24
        self.CONST_kB  = 1.3806505e-16
        self.unit_length = self.CONST_pc*1e3  # 1 kpc
        self.unit_time   = self.CONST_yr*1e6  # 1 Myr
        self.unit_density = self.CONST_amu    # 1 mp/cm-3
        self.unit_velocity = self.unit_length/self.unit_time
        self.KELVIN = self.unit_velocity*self.unit_velocity*self.CONST_amu/self.CONST_kB
        self.unit_q = (self.unit_density * (self.unit_velocity**3))/self.unit_length
        self.g = 5/3
        
        # avg atomic mass
        Xsol = 1.0
        Zsol = 1.0
        
        X = Xsol * 0.7381
        Z = Zsol * 0.0134
        Y = 1 - X - Z
        
        self.mu  = 1.0/(2.*X+ 3.*(1.-X-Z)/4.+ Z/2.);
        self.mue = 2.0/(1.0+X);
        self.muH = 1.0/X;
        self.mH = 1.0

        # alpha values for different sims
        self.alpha_hyd = 2 ** (1 / 3)  # 1.26 # 1.383
        self.alpha_mhd = (2 * 4.2 / 0.77) ** (1 / 3)

u = unit()

def calc_T(P, rho):
    """
    Calculates temeprature from constants
    ----------
    P: gas pressure
    rho: gas density
    """
    T = P/rho * u.KELVIN * u.mu
    return T

def calc_cs(T):
    """
    Calculates sound speed
    ----------
    T: temperature
    mu: avg atomic number of the gas
    """
    # convert to cm
    m_to_cm = 100

    # return np.sqrt(g.g*R*T_hot/M) * m_to_cm/g.unit_velocity
    # return np.sqrt(u.g * u.CONST_kB / (u.mu * u.CONST_amu) * T) * m_to_cm / u.unit_velocity
    return np.sqrt(u.g * u.CONST_kB / (u.mu * u.CONST_amu) * T) / u.unit_velocity

def calc_mach(v_turb, P, rho):
    """
    Calculates the Mach number
    ----------
    v_turb: turbulence velocity
    P, rho
    """
    T = calc_T(P, rho)
    cs = calc_cs(T)
    print(f'cs = {cs}')
    return v_turb / cs

def calc_dedt_mach(mach, P, rho, L):
    """
    Returns required dedt for a given Mach number, density, temperature, and box size
    """
    # calculate sound speed first
    T = calc_T(P, rho)
    cs_new = calc_cs(T)
    print(f"cs_hot: {cs_new}")

    dedt_req = rho * (cs_new**3) * (L**2) * (mach**3) / (u.alpha_hyd**3)
    return dedt_req


def calc_dedt_vturb(v_turb, rho, L):
    """
    Returns required dedt for a given turbulent velocity, density, and box size

    Does NOT calculate sound speed, so no pressure required
    """
    
    dedt_req = rho * v_turb**3 * (L**2) / (u.alpha_hyd**3)
    return dedt_req

def find_ind_l(seq, val):
    seq = list(seq)
    return seq.index(list(filter(lambda x: x>val, seq))[0])

def get_datamd(fname=None,
               key='rho', verbose=False):
    """
    Get and plot data for 2d/3d runs
    """
    data = athdf(filename=fname,)
    if verbose: print(list(data.keys()))
    try:
        len(data[key])
        return data[key][0] if len(data[key]) == 1 else data[key]
    except:
        return data[key]
        

# mass radios of different temperature trials

missing_trials_mass_ratios = {}
cold_cloud_ratio_list = [2, 3, 5, 10]  # pick from these ratios for the cold gas mass


from tqdm import tqdm
trials_missing = ['240612_0.5_1000', '240612_0.8_1200']

for trial in trials_missing:  # np.concatenate([trials_12, trials_18])
    print(trial)
    print('-' * 50)
    print('\n' * 5)
    datapath = f'/freya/ptmp/mpa/wuze/data/{trial}'

    """read parameters"""
    with open(f'{datapath}/params.pickle', 'rb') as handle:
        rp = pickle.load(handle)
    
    """construct a list of files"""
    file_list = np.sort([f'{datapath}/cloud/{f}' for f in os.listdir(f'{datapath}/cloud') if f.startswith('Turb.out2') and f.endswith('.athdf')])
    end_epoch = int(file_list[-1].split('.')[-2])
    
    time_athdf = []  # time array for the athdfs
    mass_athdf = []  # mass array for the athdfs
    temperature_athdf = []  # temperature dist for the athdfs
    
    mass_fl = []  # mass dist of FIRST AND LAST athdfs
    temperature_fl = []  # mass dist of FIRST AND LAST athdfs
    
    # calculate grid volume
    grid_size = 1 / rp['grid_dim'] ** 3
    
    i = 0
    for fname in tqdm(file_list):
        t = get_datamd(fname=fname, verbose=False, key='Time')
        rho = get_datamd(fname=fname, verbose=False, key='rho').flatten()
        press = get_datamd(fname=fname, verbose=False, key='press').flatten()
        temperature = calc_T(press, rho)
    
        # append
        # if first or last file, get ratio
        if i in [0, len(file_list)-1]:
            print(f'FL recorded at {fname.split('/')[-1]}')
            mass_fl.append(rho * grid_size)
            temperature_fl.append(temperature)
        # regular intervals
        time_athdf.append(t)
        mass_athdf.append(rho * grid_size)
        temperature_athdf.append(temperature)
        i += 1

    # save as pickles
    import pickle
    with open(f'{datapath}/cloud/time_temperature', 'wb') as handle:
        # append the time, temperature, and mass evolutions
        pickle.dump([time_athdf, temperature_athdf, mass_athdf], handle, protocol=pickle.HIGHEST_PROTOCOL)

    """dictionary of cold / warm / hot gas mass ratios"""
    # corresponds to different definitions of cold gas mass
    mass_ratios = {} # temperature: [cold, warm, hot]
    
    for cold_cloud_ratio in cold_cloud_ratio_list:
        # calculate cold gas temperature
        T_cold = rp['T_cloud'] * cold_cloud_ratio
        
        # select cold and hot gas densities
        cgr = np.sum(mass_fl[-1][temperature_fl[-1] <= T_cold])\
        / np.sum(mass_fl[0][temperature_fl[0] <= T_cold])
        wgr = np.sum(mass_fl[-1][np.logical_and(temperature_fl[-1] > T_cold, temperature_fl[-1] <= rp['T_warm'])])\
        / np.sum(mass_fl[0][temperature_fl[0] <= T_cold])  # divide warm gas by cold gas mass
        hgr = np.sum(mass_fl[-1][temperature_fl[-1] > rp['T_cut']])\
        / np.sum(mass_fl[0][temperature_fl[0] > rp['T_cut']])
    
        mass_ratios[cold_cloud_ratio] = [cgr, wgr, hgr]
    
    print(mass_ratios)
    missing_trials_mass_ratios[trial] = mass_ratios

# print the trials
print(list(missing_trials_mass_ratios.keys()))
with open(f'/freya/ptmp/mpa/wuze/data/missing_trials_mass_ratios', 'wb') as handle:
    pickle.dump(missing_trials_mass_ratios, handle, protocol=pickle.HIGHEST_PROTOCOL)

# concatenate
with open('/freya/ptmp/mpa/wuze/data/all_trials_mass_ratios', 'rb') as handle:
    all_trials_mass_ratios = pickle.load(handle)
for key, val in missing_trials_mass_ratios.items():
    all_trials_mass_ratios[key] = val
# save everything
with open(f'/freya/ptmp/mpa/wuze/data/all_trials_mass_ratios', 'wb') as handle:
    pickle.dump(all_trials_mass_ratios, handle, protocol=pickle.HIGHEST_PROTOCOL)