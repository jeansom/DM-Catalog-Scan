# This code makes a series of fake data maps, for all quartile data
# Number of each map made is nsim
# It makes maps with DM cross section from 1e-21 down to 1e-28 (steps of 0.5 in log space)
# These are based on the DM J-factor map (PP factor is for 100 GeV b-bbar)
# Also makes no dm
# out string is appended to output

from global_variables import *
from global_var import *
import numpy as np
import healpy as hp
import pandas as pd
from scipy.interpolate import interp1d
from scipy import integrate
import fermi.fermi_plugin as fp
import argparse

nsim = 100
outstring = 'allhalos_p7'

nside=128
eventclass=5 # 2 (Source) or 5 (UltracleanVeto)
eventtype=0 # 0 (all), 3 (bestpsf) or 5 (top3 quartiles)
emin_bin=0
emax_bin=40 # Must match the norm file!

f_global = fp.fermi_plugin(maps_dir,fermi_data_dir=fermi_data_dir,work_dir=work_dir,CTB_en_min=emin_bin,CTB_en_max=emax_bin,nside=nside,eventclass=eventclass,eventtype=eventtype,newstyle=1,data_July16=True)

f_global.add_ps_model(comp='ps_model')

ps_model_old = np.array(f_global.template_dict['ps_model'])

np.save('/tigress/nrodd/2mass2furious/PSmaps/ps_model_old.npy',ps_model_old)
