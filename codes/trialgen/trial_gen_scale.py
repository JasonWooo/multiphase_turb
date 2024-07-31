import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# import funcs
from codes.funcs import *

u = unit()
print(u)


"""
Information about trials
"""

trial_old = '240711_0.4_16000'

# turb_hdf_path = f'/freya/ptmp/mpa/wuze/data/cloud_hdfs/mach_{trial_old.split('_')[1]}.athdf'
trial_dep = '240711_0.4_320'; turb_hdf_path = f'/freya/ptmp/mpa/wuze/data/{trial_dep}/turb/Turb.cons.{30:05d}.athdf'
# turb_hdf_path = '../turb/Turb.cons.00100.athdf'

trial_mach = 0  # leave as 0 to be the same

"""SCALING PARAMETERS"""
radius_scale = 2 # the scaling of radius relative to the previous run
rho_scale = 1 # the scaling of density relative to the previous run
clf_number = 0.3  # the Courant, Friedrichs, & Lewy (CFL) Number

# generate new trial
import datetime; date = str(datetime.datetime.now().date()).replace('-', '')[2:]  # six digit date
trial_mach = trial_old.split('_')[1] if trial_mach == 0 else trial_mach

trial_new = f'{date}_{trial_mach}_{float(trial_old.split('_')[-1]) * radius_scale * rho_scale:.0f}'

print(f'Old trial {trial_old}')
print(f'New trial {trial_new}')

import pickle
datapath = f'/freya/ptmp/mpa/wuze/data/{trial_old}'
with open(f'{datapath}/params.pickle', 'rb') as handle:
    run_params_old = pickle.load(handle)


"""
Load and change parameters
"""

# new params

"""
RUN & BOX PARAMS
"""

## copy and change
run_params_all = run_params_old.copy()
run_params_all['trial'] = trial_new
run_params_all['cloud_radius'] = run_params_old['cloud_radius'] * radius_scale  # multiply radius by scale
run_params_all['rho_hot'] = run_params_old['rho_hot'] * rho_scale  # multiply density by scale
run_params_all['mach'] = float(trial_mach)

## dimensions
run_params_all['grid_dim'] = 256  # scale up the dimensions
run_params_all['grid_mesh'] = 32  # scale up the grid mesh
run_params_all['pfloor'] = 1e-9  # DECREASE the pressure floor

## scale of the box
# run_params_all['box_scale'] = run_params_old['box_scale']  # keep the original box scale
run_params_all['box_scale'] = 50
run_params_all['box_size'] = run_params_all['cloud_radius'] * run_params_all['box_scale']  # get actual box size

run_params_all['turb_cons_hdf_path'] = turb_hdf_path

"""
DERIVED PARAMS
"""
run_params_all['chi'] = run_params_all['T_hot'] / run_params_all['T_cloud']
run_params_all['P_hot'] = calc_P(run_params_all['T_hot'], run_params_all['rho_hot'])  # scaled hot gas pressure
run_params_all['v_turb'] = calc_vturb(run_params_all['mach'], T=run_params_all['T_hot'])  # turbulent velocity from desired mach number
run_params_all['t_eddy'] = run_params_all['box_size']/run_params_all['v_turb']
run_params_all['cs_hot'] = calc_cs(run_params_all['T_hot'])  # sound speed for hot gas

run_params_all['T_mix'] = np.sqrt(run_params_all['T_cloud'] * run_params_all['T_hot'])
run_params_all['T_warm'] = run_params_all['T_mix']  # set to mixed gas temperature, for tracking

# dedt = 0.5 * v_turb^3 = (Mach_num*cs_hot)^3 given the density & box size
run_params_all['dedt'] = calc_dedt_mach(run_params_all['mach'], T=run_params_all['T_hot'], rho=run_params_all['rho_hot'], L=run_params_all['box_size'])

# tcorr ~ t_eddy
run_params_all['t_corr'] = run_params_all['t_eddy']

# dtdrive << t_eddy
run_params_all['dt_drive'] = run_params_all['t_eddy'] / 1e3

# the end time of simulation, multiples of time_start
# reaches >5 t_cc or t_eddysqu
run_params_all['t_cc'] = (run_params_all['chi'] ** (1/2) * run_params_all['cloud_radius'] / run_params_all['v_turb'])
run_params_all['t_maxc'] = np.max([run_params_all['t_eddy'], run_params_all['t_cc']])  # 1 for floored, 1 for start
# the start time of CLOUD simulation; the end time of the TURB simulation
run_params_all['time_start'] = 3 * run_params_all['t_eddy']  # make it 3 x t_eddy
# the end time of CLOUD simulation
run_params_all['time_end'] = 5 * run_params_all['t_maxc'] + run_params_all['time_start']  # make the duration of the cloud runs 5 x t_cc
run_params_all['dt_hdf5'] = run_params_all['t_maxc'] / 10  # the time interval between hdf5 outputs
dt_hdf5_turb = run_params_all['t_eddy'] / 10  # tim interval between hdf5 outputs for TURB
run_params_all['time_cloud'] = run_params_all['time_start']  # introduce cloud when simulation starts

# compare the two
print(f"{'PARAMETER':<20}{'NEW':<30}{'OLD':<30}")
print('-' * 70)
for entry in run_params_all.keys():
    try:  # for floats
        if run_params_all[entry] > 1:  # if a fraction
            print(f"{entry:<20}{run_params_all[entry]:<30.10f}{run_params_old[entry]:<30.10f}")
        else:
            print(f"{entry:<20}{run_params_all[entry]:<30.5e}{run_params_old[entry]:<30.5e}")
    except:  # for strings
        try:
            print(f"{entry:<20}{str(run_params_all[entry]):<30}{str(run_params_old[entry]):<30}")
        except:  # if old does not exist
            try:
                run_params_old[entry]
            except:
                print(f"{entry:<20}{str(run_params_all[entry]):<30}{'NONE EXISTENT':<30}")


"""Define run parameters"""

class run_params():
    def __init__(self, run_params_all):
        self.trial = run_params_all['trial']
        self.cloud_flag = run_params_all['cloud_flag']
        self.T_cloud = run_params_all['T_cloud']
        self.T_cold = run_params_all['T_cold']
        self.cloud_radius = run_params_all['cloud_radius']
        self.cloud_pos = run_params_all['cloud_pos']
        self.box_size = run_params_all['box_size']
        self.grid_dim = run_params_all['grid_dim']
        self.grid_vol = run_params_all['grid_vol']
        self.grid_mesh = run_params_all['grid_mesh']
        self.pfloor = run_params_all['pfloor']
        self.T_floor = run_params_all['T_floor']
        self.T_ceil = run_params_all['T_ceil']
        self.cooling_flag = run_params_all['cooling_flag']
        self.T_hot = run_params_all['T_hot']
        self.rho_hot = run_params_all['rho_hot']
        self.mach = run_params_all['mach']
        self.T_cut = run_params_all['T_cut']
        self.chi = run_params_all['chi']
        self.P_hot = run_params_all['P_hot']
        self.v_turb = run_params_all['v_turb']
        self.t_eddy = run_params_all['t_eddy']
        self.cs_hot = run_params_all['cs_hot']
        self.T_mix = run_params_all['T_mix']
        self.T_warm = run_params_all['T_warm']
        self.dedt = run_params_all['dedt']
        self.t_corr = run_params_all['t_corr']
        self.dt_drive = run_params_all['dt_drive']
        self.t_cc = run_params_all['t_cc']
        self.t_maxc = run_params_all['t_maxc']
        self.time_start = run_params_all['time_start']
        self.time_end = run_params_all['time_end']
        self.dt_hdf5 = run_params_all['dt_hdf5']
        self.time_cloud = run_params_all['time_cloud']

rp = run_params(run_params_all)


# make datapath
import os
datapath = f'/freya/ptmp/mpa/wuze/data/{rp.trial}'

if not os.path.exists(datapath):
    os.makedirs(datapath)
    os.makedirs(f'{datapath}/cloud')
    os.makedirs(f'{datapath}/turb')
    print(f'Made path {datapath}')

# save as pickles
# the dictionary saved here is separate from the rp object
import pickle

with open(f'{datapath}/params.pickle', 'wb') as handle:
    pickle.dump(run_params_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(run_params_all)


"""
Generate files for the new run
"""

#----------------------------------------
# Generate readme
#----------------------------------------

def infgen_readme(datapath, readme_msg="This run"):
    f = open(f"{datapath}/readme", "w")
    
    # generate text
    readme_text =\
    f"""
{readme_msg}

{'-' * 50}
This run is auto-generated by the trial_gen.ipynb

This run uses:
    """
    
    # generate variables
    readme_vars = f'\n{'-' * 20}\n'
    for key, val in run_params_all.items():
        if isinstance(val, float):
            app = f'{key} = {val:.4f}\n'
        else:
            app = f'{key} = {val}\n'
        readme_vars += app
    
    f.write(readme_text)
    f.write('-' * 50)
    f.write(readme_vars)
    f.close()

#----------------------------------------
# Generate TURB .turb file
#----------------------------------------

def infgen_init_turb(datapath):
    f = open(f"{datapath}/turb/athinput_init.turb", "w")
    
    # generate text
    turb_config =\
    f""" 
<comment>
problem   = Adding turbulence
configure = --prob=turb_v2 -fft 

<job>
problem_id = Turb # problem ID: basename of output filenames


#~----------OUTPUTS-----------~#
<output1>
file_type  = hst        # history data dump
dt         = {dt_hdf5_turb / 100 :<20.10f}# time increment between outputs

<output2>
file_type = hdf5        # HDF5 data dump
variable  = prim        # variables to be output
dt        = {dt_hdf5_turb:<20.10f}# time increment between outputs

<output3>
file_type = rst         # restart file dump                       
dt        = {dt_hdf5_turb * 10 :<20.10f}# time increment between outputs

<output4>
file_type = hdf5        # HDF5 data dump
variable  = cons        # variables to be output
dt        = {dt_hdf5_turb:<20.10f}# time increment between outputs
id        = cons

#~----------SETUP-----------~#
<time>
cfl_number  = {clf_number:<20}# The Courant, Friedrichs, & Lewy (CFL) Number
nlim        = -1        # cycle limit
# time limit of second trial should be larger
tlim        = {run_params_all['time_start']:<20.10f}# time limit for the turb run, where the cloud run starts
integrator  = rk2       # time integration algorithm
xorder      = 2         # order of spatial reconstruction
ncycle_out  = 1         # interval for stdout summary info

<mesh>
nx1    = {run_params_all['grid_dim']}             # number of zones in x1-direction
x1min  = {-run_params_all['box_size']/2}           # minimum value of x1
x1max  = {run_params_all['box_size']/2}            # maximum value of x1
ix1_bc = periodic       # inner-x1 boundary condition
ox1_bc = periodic       # outer-x1 boundary condition

nx2    = {run_params_all['grid_dim']}             # number of zones in x2-direction
x2min  = {-run_params_all['box_size']/2}           # minimum value of x2
x2max  = {run_params_all['box_size']/2}            # maximum value of x2
ix2_bc = periodic       # inner-x2 boundary condition
ox2_bc = periodic       # outer-x2 boundary condition

nx3    = {run_params_all['grid_dim']}             # number of zones in x3-direction
x3min  = {-run_params_all['box_size']/2}           # minimum value of x3
x3max  = {run_params_all['box_size']/2}            # maximum value of x3
ix3_bc = periodic       # inner-x3 boundary condition
ox3_bc = periodic       # outer-x3 boundary condition

refinement  = none      # type of mesh refinement to use

<meshblock>
nx1 = {run_params_all['grid_mesh']}                # block size in x1-direction
nx2 = {run_params_all['grid_mesh']}                # block size in x2-direction
nx3 = {run_params_all['grid_mesh']}                # block size in x3-direction

<hydro>
gamma      = 1.6666666666666667  # gamma = C_p/C_v
pfloor     = {run_params_all['pfloor']:<20.10f}  # pressure floor

#~----------TURBULENCE PARAMS-----------~#
<turbulence>
dedt       = {run_params_all['dedt']:<20.15f}# Energy injection rate (for driven) or Total energy (for decaying)
nlow       = 0                   # cut-off wavenumber at low-k
nhigh      = 2                   # cut-off wavenumber at high-k
expo       = 2.0                 # power-law exponent
tcorr      = {run_params_all['t_corr']:<20.15f}# correlation time for OU process (both impulsive and continuous)
dtdrive    = {run_params_all['dt_drive']:<20.15f}# time interval between perturbation (impulsive)
f_shear    = 0.3                 # the ratio of the shear component
rseed      = 1                   # if non-negative, seed will be set by hand (slow PS generation)
# dedt should be calibrated by        dedt = 0.5 * v_turb^3 = (Mach_num*cs_hot)^3
# tcorr ~ t_eddy
# dtdrive << t_eddy

<problem>
turb_flag    = 2                 # 1 for decaying, 2 (impulsive) or 3 (continuous) for driven turbulence
rescale_flag = 1                 # 1 for cloud runs


#~----------HEATING & COOLING-----------~#
# User-defined variables:

heating = 0.001                  # constant volume heating rate

cs_hot = {run_params_all['cs_hot']:<20.10f}# hot gas sound speed for T~4e6K

# turn on cooling for cloud runs
cooling_flag = 0                 # set to 1 to turn on cooling, 0 to turn it off
global_cooling_flag = 0          # set to 1 to turn on uniform global cooling, 0 to turn it off
# turn on cloud
restart_cloud_flag   = 0                 # set to 1 to add the cloud on restart, 0 to not 
hdf_cloud_flag       = 0                 # set to 1 to add the cloud after reading HDF5 file, 0 to not 

amb_rho      = {run_params_all['rho_hot']}               # density of the ambient medium, in code units


#~----------CLOUD PROPERTIES-----------~#
cloud_radius = {run_params_all['cloud_radius']:<20.10f}# radius of the cloud, in code units
# this should be the same as simulation start
start_time   = {0:<20.10f}# time of insertion of cloud, in code units
cloud_time   = {run_params_all['time_cloud']:<20.10f}# time of insertion of cloud, in code units
# temperature ratio T_init / T_cloud
# TO SET COLD GAS TEMPERATURE, CHANGE HERE
cloud_chi    = {run_params_all['chi']}               # density contrast of the cloud, rho_cloud/amb_rho

cloud_pos_x  = {run_params_all['cloud_pos'][0]}               # cloud center position x-coordinate
cloud_pos_y  = {run_params_all['cloud_pos'][1]}               # cloud center position y-coordinate
cloud_pos_z  = {run_params_all['cloud_pos'][2]}               # cloud center position z-coordinate

#~-----------TEMPERATURE----------~#
T_floor      = {run_params_all['T_floor']:<10.0f}# floor temperature in the simulation
T_ceil       = {run_params_all['T_ceil']:<10.0f}# ceiling temperature in the simulation

# medium mass is integrated from 2 x temp
T_hot_req    = {run_params_all['T_hot']:<10.0f}# hot medium temperature required, reset to this on restart, if cloud_flag is 1
T_hot        = {run_params_all['T_hot']:<10.0f}# initial hot medium temperature (box heats up due to turbulence)
T_cold       = {run_params_all['T_cold']:<10.0f}# cold medium temperature, ONLY used in cold gas mass calculation
T_cut_mul    = 0.5               # T_cut = T_hot_req * T_cut_mul, gas higher than T_cut is not cooled
# infinite cooling time for T > T_cut, want hot gas to stay hot for the medium
T_cut        = {run_params_all['T_cut']:<10.0f}# gas higher than T_cut is not cooled
T_warm       = {run_params_all['T_warm']:<10.2f}# warm medium temperature, ONLY used in warm gas mass calculation

Xsol         = 1.0               # Change H-abundance src/utils/code_units
Zsol         = 1.0               # Change metallicity src/utils/code_units

B_x          = 0.0               # initial magnetic field in x-direction
B_y          = 0.0               # initial magnetic field in y-direction
B_z          = 0.0               # initial magnetic field in z-direction

# only read if hdf_cloud_flag = 1
cons_input_filename = /dev/null  # name of HDF5 file containing initial conditions
dataset_cons   = cons       # name of dataset containing conserved values
index_dens     = 0          # index of density in conserved dataset
index_etot     = 1          # index of energy in conserved dataset (for nonbarotropic EOS)
index_mom1     = 2          # index of x1-momentum in conserved dataset
index_mom2     = 3          # index of x2-momentum in conserved dataset
index_mom3     = 4          # index of x3-momentum in conserved dataset

# only used if -b flag is on for compiling
b1_input_filename = /dev/null  # name of HDF5 file containing initial conditions
b2_input_filename = /dev/null  # name of HDF5 file containing initial conditions
b3_input_filename = /dev/null  # name of HDF5 file containing initial conditions
    """
    
    # generate variables
    f.write(turb_config)
    f.write('-' * 50)
    f.close()

#----------------------------------------
# Generate CLOUD .turb file
#----------------------------------------

def infgen_cloud_turb(datapath):
    f = open(f"{datapath}/cloud/athinput_cloud.turb", "w")
    
    # generate text
    cloud_config =\
    f"""
<comment>
problem   = Adding cloud
configure = --prob=turb_v2 -fft 

<job>
problem_id = Turb # problem ID: basename of output filenames


#~----------OUTPUTS-----------~#
<output1>
file_type  = hst        # history data dump
dt         = {run_params_all['dt_hdf5'] / 100 :<20.10f}# time increment between outputs

<output2>
file_type = hdf5        # HDF5 data dump
variable  = prim        # variables to be output
dt        = {run_params_all['dt_hdf5']:<20.10f}# time increment between outputs

<output3>
file_type = rst         # restart file dump                       
dt        = {run_params_all['dt_hdf5'] * 10 :<20.10f}# time increment between outputs

<output4>
file_type = hdf5        # HDF5 data dump
variable  = cons        # variables to be output
dt        = {run_params_all['dt_hdf5']:<20.10f}# time increment between outputs
id        = cons

#~----------SETUP-----------~#
<time>
cfl_number  = {clf_number:<20}# The Courant, Friedrichs, & Lewy (CFL) Number
nlim        = -1        # cycle limit
# time limit of second trial should be larger
tlim        = {run_params_all['time_end']:<20.10f}# time limit
integrator  = rk2       # time integration algorithm
xorder      = 2         # order of spatial reconstruction
ncycle_out  = 1         # interval for stdout summary info

<mesh>
nx1    = {run_params_all['grid_dim']}             # number of zones in x1-direction
x1min  = {-run_params_all['box_size']/2}           # minimum value of x1
x1max  = {run_params_all['box_size']/2}            # maximum value of x1
ix1_bc = periodic       # inner-x1 boundary condition
ox1_bc = periodic       # outer-x1 boundary condition

nx2    = {run_params_all['grid_dim']}             # number of zones in x2-direction
x2min  = {-run_params_all['box_size']/2}           # minimum value of x2
x2max  = {run_params_all['box_size']/2}            # maximum value of x2
ix2_bc = periodic       # inner-x2 boundary condition
ox2_bc = periodic       # outer-x2 boundary condition

nx3    = {run_params_all['grid_dim']}             # number of zones in x3-direction
x3min  = {-run_params_all['box_size']/2}           # minimum value of x3
x3max  = {run_params_all['box_size']/2}            # maximum value of x3
ix3_bc = periodic       # inner-x3 boundary condition
ox3_bc = periodic       # outer-x3 boundary condition

refinement  = none      # type of mesh refinement to use

<meshblock>
nx1 = {run_params_all['grid_mesh']}                # block size in x1-direction
nx2 = {run_params_all['grid_mesh']}                # block size in x2-direction
nx3 = {run_params_all['grid_mesh']}                # block size in x3-direction

<hydro>
gamma      = 1.6666666666666667  # gamma = C_p/C_v
pfloor     = {run_params_all['pfloor']:<20.10f}  # pressure floor

#~----------TURBULENCE PARAMS-----------~#
<turbulence>
dedt       = {run_params_all['dedt']:<20.15f}# Energy injection rate (for driven) or Total energy (for decaying)
nlow       = 0                   # cut-off wavenumber at low-k
nhigh      = 2                   # cut-off wavenumber at high-k
expo       = 2.0                 # power-law exponent
tcorr      = {run_params_all['t_corr']:<20.15f}# correlation time for OU process (both impulsive and continuous)
dtdrive    = {run_params_all['dt_drive']:<20.15f}# time interval between perturbation (impulsive)
f_shear    = 0.3                 # the ratio of the shear component
rseed      = 1                   # if non-negative, seed will be set by hand (slow PS generation)
# dedt should be calibrated by        dedt = 0.5 * v_turb^3 = (Mach_num*cs_hot)^3
# tcorr ~ t_eddy
# dtdrive << t_eddy

<problem>
turb_flag    = 2                 # 1 for decaying, 2 (impulsive) or 3 (continuous) for driven turbulence
rescale_flag = 1                 # 1 for cloud runs


#~----------HEATING & COOLING-----------~#
# User-defined variables:

heating = 0.001                  # constant volume heating rate

cs_hot = {run_params_all['cs_hot']:<20.10f}# hot gas sound speed for T~4e6K

# turn on cooling for cloud runs
cooling_flag = {1 if run_params_all['cooling_flag'] else 0}                 # set to 1 to turn on cooling, 0 to turn it off
global_cooling_flag = 0          # set to 1 to turn on uniform global cooling, 0 to turn it off
# turn on cloud
restart_cloud_flag   = 0                 # set to 1 to add the cloud on restart, 0 to not 
hdf_cloud_flag       = {1 if run_params_all['cloud_flag'] else 0}                 # set to 1 to add the cloud after reading HDF5 file, 0 to not 

amb_rho      = {run_params_all['rho_hot']}               # density of the ambient medium, in code units


#~----------CLOUD PROPERTIES-----------~#
cloud_radius = {run_params_all['cloud_radius']:<20.10f}# radius of the cloud, in code units
# this should be the same as simulation start
start_time   = {run_params_all['time_start']:<20.10f}# time of insertion of cloud, in code units
cloud_time   = {run_params_all['time_cloud']:<20.10f}# time of insertion of cloud, in code units
# temperature ratio T_init / T_cloud
# TO SET COLD GAS TEMPERATURE, CHANGE HERE
cloud_chi    = {run_params_all['chi']}               # density contrast of the cloud, rho_cloud/amb_rho

cloud_pos_x  = {run_params_all['cloud_pos'][0]}               # cloud center position x-coordinate
cloud_pos_y  = {run_params_all['cloud_pos'][1]}               # cloud center position y-coordinate
cloud_pos_z  = {run_params_all['cloud_pos'][2]}               # cloud center position z-coordinate

#~-----------TEMPERATURE----------~#
T_floor      = {run_params_all['T_floor']:<10.0f}# floor temperature in the simulation
T_ceil       = {run_params_all['T_ceil']:<10.0f}# ceiling temperature in the simulation

# medium mass is integrated from 2 x temp
T_hot_req    = {run_params_all['T_hot']:<10.0f}# hot medium temperature required, reset to this on restart, if cloud_flag is 1
T_hot        = {run_params_all['T_hot']:<10.0f}# initial hot medium temperature (box heats up due to turbulence)
T_cold       = {run_params_all['T_cold']:<10.0f}# cold medium temperature, ONLY used in cold gas mass calculation
T_cut_mul    = 0.5               # T_cut = T_hot_req * T_cut_mul, gas higher than T_cut is not cooled
# infinite cooling time for T > T_cut, want hot gas to stay hot for the medium
T_cut        = {run_params_all['T_cut']:<10.0f}# gas higher than T_cut is not cooled
T_warm       = {run_params_all['T_warm']:<10.2f}# warm medium temperature, ONLY used in warm gas mass calculation

Xsol         = 1.0               # Change H-abundance src/utils/code_units
Zsol         = 1.0               # Change metallicity src/utils/code_units

B_x          = 0.0               # initial magnetic field in x-direction
B_y          = 0.0               # initial magnetic field in y-direction
B_z          = 0.0               # initial magnetic field in z-direction

# only read if hdf_cloud_flag = 1
cons_input_filename = {run_params_all['turb_cons_hdf_path']}  # name of HDF5 file containing initial conditions
dataset_cons   = cons       # name of dataset containing conserved values
index_dens     = 0          # index of density in conserved dataset
index_etot     = 1          # index of energy in conserved dataset (for nonbarotropic EOS)
index_mom1     = 2          # index of x1-momentum in conserved dataset
index_mom2     = 3          # index of x2-momentum in conserved dataset
index_mom3     = 4          # index of x3-momentum in conserved dataset

# only used if -b flag is on for compiling
b1_input_filename = /dev/null  # name of HDF5 file containing initial conditions
b2_input_filename = /dev/null  # name of HDF5 file containing initial conditions
b3_input_filename = /dev/null  # name of HDF5 file containing initial conditions
    """
    
    # generate variables
    f.write(cloud_config)
    f.write('-' * 50)
    f.close()


#----------------------------------------
# readme
#----------------------------------------

readme_msg = """

THIS IS ONE OF THE NEW RUNS,
check notebook on 24/7/12!

These runs use:
256 cell
1/50 cloud radius
1e-9 pfloor
0.3 cfl

"""


"""
Write the files
"""

infgen_readme(datapath=datapath, readme_msg=readme_msg)
# init / turb
infgen_init_turb(datapath=datapath)
# cloud
infgen_cloud_turb(datapath=datapath)

import datetime
print(datetime.datetime.now())
print('\n' * 10)





"""
Generate .sh files
"""

# Get trial params
import datetime
date = str(datetime.datetime.now().date()).replace('-', '')[2:]  # six digit date
run_dp = trial_new[7:]  # x_y

# Machine params

executable = 'athena_turb'
server = '24h' #test
if server == '24h':
    comp_time = '3:55:00'#'23:55:00'
    comp_time_save = '3:50:00'#'23:50:00'
elif server == 'test':
    comp_time = '00:20:00'
    comp_time_save = '00:19:00'

print('\n\nMachine params for TURB .sh file:\n')
print(date, run_dp, executable, server, comp_time, comp_time_save)

sh_content = f"""#!/usr/bin/env bash
# source ~/.bashrc

#SBATCH -J {date[2:]}{trial_new[6:]}_turb
#SBATCH -o ./out/{date}/{run_dp}_turb."%j".out
#SBATCH -e ./out/{date}/{run_dp}_turb."%j".err
#SBATCH --mail-user wuz@mpa-garching.mpg.de
#-- wuz@mpa-garching.mpg.de
#SBATCH --partition=p.24h
#-- p.24h, p.test, p.gpu & p.gpu.ampere
#SBATCH --mail-type=ALL
#SBATCH --nodes=16
#-- 2 at most?
#SBATCH --ntasks-per-node=32
#-- 40 at most
#SBATCH --time={comp_time}
#-- in format hh:mm:ss

set -e
SECONDS=0

module purge
module load intel/19.1.2
module load impi/2019.8
module load fftw-mpi/3.3.8
module load hdf5-mpi/1.8.21
module load ffmpeg/4.4
module list

echo {trial_new}
cd $DP/{trial_new}/turb

srun ../../{executable} -i athinput_init.turb -t {comp_time_save}
#-- !!TIME_LIMIT_RST!! should be ten minutes before the end, to generate a restart file

echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"

echo "Boom!"

cd ~
"""

# write the file
file_name = f"{datapath}/js_turb_24h.sh" if server == '24h' else f"{datapath}/js_turb.sh"
with open(file_name, 'w') as file:
    file.write(sh_content)

# print the squeue command
squeue_cmd = f"""
sbatch {datapath}/js_turb_24h.sh
"""
print('-' * 50)
print(squeue_cmd)





"""
Machine params
"""
executable = 'athena_stop'#'athena_turb'
server = '24h' #test
if server == '24h':
    comp_time = '5:55:00'#'23:55:00'
    comp_time_save = '5:50:00'#'23:50:00'
elif server == 'test':
    comp_time = '00:20:00'
    comp_time_save = '00:19:00'

print('\n\nMachine params for CLOUD .sh file:\n')
print(date, run_dp, executable, server, comp_time, comp_time_save)

sh_content = f"""#!/usr/bin/env bash
# source ~/.bashrc

#SBATCH -J {date[2:]}{trial_new[6:]}_cloud
#SBATCH -o ./out/{date}/{run_dp}_cloud."%j".out
#SBATCH -e ./out/{date}/{run_dp}_cloud."%j".err
#SBATCH --mail-user wuz@mpa-garching.mpg.de
#-- wuz@mpa-garching.mpg.de
#SBATCH --partition=p.24h
#-- p.24h, p.test, p.gpu & p.gpu.ampere
#SBATCH --mail-type=ALL
#SBATCH --nodes=16
#-- 2 at most?
#SBATCH --ntasks-per-node=32
#-- 40 at most
#SBATCH --time={comp_time}
#-- in format hh:mm:ss

set -e
SECONDS=0

module purge
module load intel/19.1.2
module load impi/2019.8
module load fftw-mpi/3.3.8
module load hdf5-mpi/1.8.21
module load ffmpeg/4.4
module list

echo {trial_new}
cd $DP/{trial_new}/cloud

srun ../../{executable} -i athinput_cloud.turb -t {comp_time_save}
#-- !!TIME_LIMIT_RST!! should be ten minutes before the end, to generate a restart file

echo "Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"

echo "Boom!"

cd ~
"""

# write the file
file_name = f"{datapath}/js_cloud_24h.sh" if server == '24h' else f"{datapath}/js_cloud.sh"
with open(file_name, 'w') as file:
    file.write(sh_content)

# print the squeue command
squeue_cmd = f"""
sbatch {datapath}/js_cloud_24h.sh
"""
print('-' * 50)
print(squeue_cmd)

# print the squeue command
squeue_cmd = f"""
sbatch --dependency=afterok:XXX {datapath}/js_cloud_24h.sh
"""
print('-' * 50)
print(squeue_cmd)

print('\n' * 20)