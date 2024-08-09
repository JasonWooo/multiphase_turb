### import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

import sys, os
sys.path.append(os.path.abspath('/freya/ptmp/mpa/wuze/multiphase_turb'))
import codes
from codes.funcs import *  # import everything in functions
from codes.plotting.checks import *

# read the trial name
fig_trial = sys.argv[1]

_ = \
tracer_avg_evol_load(trial=fig_trial, nbins_temp = 73,
                     ncores = [1, 1])