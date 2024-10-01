import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import os

from codes.jason import plotting_def, plot_prettier
plotting_def()

"""
----------------------------------------
Simple functions
----------------------------------------
"""

def _into_dict(start_key, end_key):
    """
    Parses variables into a dictionary
    NOT standard practice
    """
    all_keys = list(globals().keys())
    dict_keys = all_keys[all_keys.index(start_key) : all_keys.index(end_key)+1]
    parse_dict = {}
    for key in dict_keys:
        parse_dict[key] = globals()[key]
    return parse_dict

def timer(func):
    """
    A function designed to be used as @timer
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def sort_parallel(x, y):
    """
    sort x by y
    """
    return [ele for _, ele in sorted(zip(y, x))]

def s_n(num):
    """
    returns coefficient and exponent of a number under base 10
    """
    expo = np.floor(np.log10(abs(num)))
    coeff = num / (10 ** expo)
    return coeff, expo



"""
----------------------------------------
Define constants
----------------------------------------

The units here are:
length — kpc
time — Myr
mass — AMU

"""

class unit():
    """
    Converts from cgs units to code units
    """
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
        self.alpha_hyd = 2 ** (1 / 3)  # 1.26
        
        self.alpha_mhd = (2 * 4.2 / 0.77) ** (1 / 3)
# define the unit
u = unit()



"""
----------------------------------------
Define unit calculations
----------------------------------------
"""

def calc_T(P, rho):
    """
    Calculates temeprature from constants
    ----------
    P: gas pressure
    rho: gas density
    """
    T = P/rho * u.KELVIN * u.mu
    return T

def calc_P(T, rho):
    """
    Calculates pressure from constants
    ----------
    T: gas temperature
    rho: gas density
    """
    P = (T / 64**3) * (rho / 64**3) / (u.KELVIN * u.mu)
    return P

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

def calc_vturb(mach, P, rho):
    """
    Calculates the turbulent velocity from desired Mach number M
    ----------
    mach: mach number
    P, rho
    """
    T = calc_T(P, rho)
    cs = calc_cs(T)
    return mach * cs
def calc_vturb(mach, T):
    """
    Calculates the turbulent velocity from desired Mach number M
    ----------
    mach: mach number
    T
    """
    cs = calc_cs(T)
    return mach * cs

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
    Returns required dedt for a given Mach number, pressure, density, and box size
    """
    # calculate sound speed first
    T = calc_T(P, rho)
    cs_new = calc_cs(T)

    dedt_req = rho * (cs_new**3) * (L**2) * (mach**3) / (u.alpha_hyd**3)
    return dedt_req
def calc_dedt_mach(mach, T, rho, L):
    """
    Returns required dedt for a given Mach number, temperature, and box size
    """
    # calculate sound speed first
    cs_new = calc_cs(T)

    dedt_req = rho * (cs_new**3) * (L**2) * (mach**3) / (u.alpha_hyd**3)
    return dedt_req


def calc_dedt_vturb(v_turb, rho, L):
    """
    Returns required dedt for a given turbulent velocity, density, and box size

    Does NOT calculate sound speed, so no pressure required
    """
    
    dedt_req = rho * v_turb**3 * (L**2) / (u.alpha_hyd**3)
    return dedt_req







"""
----------------------------------------
I/O related functions
----------------------------------------

Deals with reading files and extracting information
"""

sys.path.append(os.path.abspath('/freya/ptmp/mpa/wuze/multiphase_turb/athena_pp/vis/python'))
from athena_read import athdf  # import the athena I/O function
import h5py

trial = '240613_0.1_10'  # rp.trial
datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'

def get_rp(trial = trial):
    datapath = f'/freya/ptmp/mpa/wuze/multiphase_turb/data/{trial}'
    with open(f'{datapath}/params.pickle', 'rb') as handle:
        rp = pickle.load(handle)
    return rp


def get_hst(trial = trial):
    # grab the hst file
    # accomodate for both the old and new convention
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
    
    return dataf

def get_datamd(fname=f'{datapath}/cloud/Turb.out2.00101.athdf',
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
    

def plot_data2d(fname=f'{datapath}/cloud/Turb.out2.00101.athdf',
                key='rho'):
    # not only retrieves teh data but also plots it
    plt.imshow(get_datamd(key=key, verbose=False), interpolation='none')
    plt.colorbar()
    plt.title(key)
    plt.show()

def get_datamds(fname=f'{datapath}/cloud/Turb.out2.00101.athdf',
                keys=['rho'], verbose=False):
    """
    Get and plot data for 2d/3d runs
    """
    data = athdf(filename=fname,)
    if verbose: print(list(data.keys()))
    vals = []
    for key in keys:
        try:
            len(data[key])
            vals.append(data[key][0] if len(data[key]) == 1 else data[key])
        except:
            vals.append(data[key])
    return vals


# finds the index of the first element in seq LARGER than val
def find_ind_l(seq, val):
    seq = list(seq)
    return seq.index(list(filter(lambda x: x>val, seq))[0])

# finds the index of the first element in seq SMALLER than val
def find_ind_s(seq, val):
    seq = list(seq)
    return seq.index(list(filter(lambda x: x < val, seq))[0])


# the binary search function for pressure condition checks
def press_binary_search(arr, condition):
    # initialize the indices
    low = 0
    high = len(arr) - 1

    # the final index
    final_ind = -1  # If condition is never met, return -1
    counter = 0
    
    while low <= high:
        mid = (low + high) // 2
        print(f'{counter:<5}: fnum {mid} / {len(arr)}')  # print the middle index
        
        if condition(arr[mid]):  # if the condition is met
            final_ind = mid  # update final_ind
            high = mid - 1  # GO LEFT
        else:
            low = mid + 1  # GO RIGHT
        counter += 1

    # get the final time
    if final_ind == -1:
        # return the last entry
        final_athdf_fname = arr[-1]
    else:
        final_athdf_fname = arr[final_ind]
    stop_time = get_datamd(fname=f'{datapath}/cloud/{final_athdf_fname}', verbose=False, key='Time')
    return stop_time