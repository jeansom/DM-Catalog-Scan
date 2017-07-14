# This code makes a series of fake data maps, for all quartile data
# Number of each map made is nsim
# It makes maps with DM cross section from 1e-21 down to 1e-28 (steps of 0.5 in log space)
# These are based on the DM J-factor map (PP factor is for 100 GeV b-bbar)
# Also makes no dm
# Input J-map should not be smoothed or exposure corrected, we'll do that below
# out string is appended to output

from global_variables import *
from global_var import *
import numpy as np
import healpy as hp
import pandas as pd
from scipy.interpolate import interp1d
from scipy import integrate
import fermi.fermi_plugin as fp
import smooth_maps as sm

nsim = 100
outstring = 'allhalos'

nside=128
eventclass=5 # 2 (Source) or 5 (UltracleanVeto)
eventtype=0 # 0 (all), 3 (bestpsf) or 5 (top3 quartiles)
emin_bin=0
emax_bin=40 # Must match the norm file!

f_global = fp.fermi_plugin(maps_dir,fermi_data_dir=fermi_data_dir,work_dir=work_dir,CTB_en_min=emin_bin,CTB_en_max=emax_bin,nside=nside,eventclass=eventclass,eventtype=eventtype,newstyle=1,data_July16=True)

# Code to make fake data in a single energy bin, with or without a DM contribution

f_global.add_diffuse_newstyle(comp = 'p7', eventclass = f_global.eventclass, eventtype = f_global.eventtype)
f_global.add_bubbles(comp='bubs') #bubbles
f_global.add_iso(comp='iso')  #iso
f_global.add_ps_model(comp='ps_model')

norm_file = './P8UCVA_base_norm.npy' 

# Pre Norm:
print "Pre Norm:"
for key in f_global.template_dict.keys():
    print "key:",key
    print "shape:",np.shape(f_global.template_dict[key])
    print "mean:",np.mean(f_global.template_dict[key])
np.load('fake')
f_global.use_template_normalization_file(norm_file,key_suffix='-0')

print "Post Norm:"
for key in f_global.template_dict.keys():
    print "key:",key
    print "shape:",np.shape(f_global.template_dict[key])
    print "mean:",np.mean(f_global.template_dict[key])

sum_sim_normed = {}
sum_sim_normed['sim']=np.zeros((emax_bin-emin_bin,hp.nside2npix(nside)))
for key in f_global.template_dict.keys():
    sum_sim_normed['sim']+=np.array(f_global.template_dict[key])

nodm_map = sum_sim_normed['sim']

nodm_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))

for i in range(40):
    print "Ebin:",i
    print "Mean:",np.mean(nodm_map[i])
