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
import argparse

nsim = 100
outstring = 'allhalos'

nside=128
eventclass=5 # 2 (Source) or 5 (UltracleanVeto)
eventtype=0 # 0 (all), 3 (bestpsf) or 5 (top3 quartiles)
emin_bin=0
emax_bin=40 # Must match the norm file!

f_global = fp.fermi_plugin(maps_dir,fermi_data_dir=fermi_data_dir,work_dir=work_dir,CTB_en_min=emin_bin,CTB_en_max=emax_bin,nside=nside,eventclass=eventclass,eventtype=eventtype,newstyle=1,data_July16=True)

# Set up J_map
J_map = hp.ud_grade(np.load('/tigress/smsharma/public/GenMaps/GenMapsJumpAround/Jfactor_DS_true_map_100.0_100.0_100.0b2e+20_a3.16e+17.npy'),nside,power=-2)
J_map /= GeV**2*Centimeter**-5

# Exposure correct then smooth J_map latter must be done first
J_map_arr_ps = np.zeros(shape=(emax_bin-emin_bin,len(J_map)))
for en in range(emax_bin-emin_bin):
    J_map_arr_ps[en] = J_map*f_global.CTB_exposure_maps[en]

smi = sm.smooth_map(maps_dir,CTB_en_min=emin_bin,CTB_en_max=emax_bin,data_name='p8',nside=nside,is_p8=True,eventclass=eventclass,eventtype=eventtype)
J_map_arr = smi.smooth_map_array(J_map_arr_ps)

# Code to make fake data in a single energy bin, with or without a DM contribution

f_global.add_diffuse_newstyle(comp = 'p7', eventclass = f_global.eventclass, eventtype = f_global.eventtype)
f_global.add_bubbles(comp='bubs') #bubbles
f_global.add_iso(comp='iso')  #iso
f_global.add_ps_model(comp='ps_model')

norm_file = './P8UCVA_norm.npy' 

f_global.use_template_normalization_file(norm_file,key_suffix='-0')

sum_sim_normed = {}
sum_sim_normed['sim']=np.zeros((emax_bin-emin_bin,hp.nside2npix(nside)))
for key in f_global.template_dict.keys():
    sum_sim_normed['sim']+=np.array(f_global.template_dict[key])

nodm_map = sum_sim_normed['sim']

# Load PP factor elements
mass = 100
dNdLogx_df=pd.read_csv(work_dir+'AdditionalData/AtProduction_gammas.dat', delim_whitespace=True)
channel = 'b'
dNdLogx_ann_df = dNdLogx_df.query('mDM == ' + (str(np.int(float(mass)))))[['Log[10,x]',channel]]
Egamma = np.array(mass*(10**dNdLogx_ann_df['Log[10,x]']))
dNdEgamma = np.array(dNdLogx_ann_df[channel]/(Egamma*np.log(10)))
dNdE_interp = interp1d(Egamma, dNdEgamma)
ebins=2*np.logspace(-1,3,41)

DM_count_map_1=np.zeros((emax_bin-emin_bin,hp.nside2npix(nside)))
DM_count_map_2=np.zeros((emax_bin-emin_bin,hp.nside2npix(nside)))
DM_count_map_3=np.zeros((emax_bin-emin_bin,hp.nside2npix(nside)))
DM_count_map_4=np.zeros((emax_bin-emin_bin,hp.nside2npix(nside)))

# When doing so convert to human units from natural units
for i in range(emax_bin-emin_bin):
    # First Calculate PP factor - want counts, not counts/GeV
    PPnoxsec = 0.0
    if ebins[i] < mass:
        if ebins[i+1] < mass:
            # Whole bin is inside
            PPnoxsec = 1.0/(8*np.pi*mass**2)*integrate.quad(lambda x: dNdE_interp(x), ebins[i], ebins[i+1])[0]
        else:
            # Bin only partially contained
            PPnoxsec = 1.0/(8*np.pi*mass**2)*integrate.quad(lambda x: dNdE_interp(x), ebins[i], mass)[0]

    DM_count_map_1[i] = PPnoxsec*J_map_arr[i]*1e-21
    DM_count_map_2[i] = PPnoxsec*J_map_arr[i]*1e-22
    DM_count_map_3[i] = PPnoxsec*J_map_arr[i]*1e-23
    DM_count_map_4[i] = PPnoxsec*J_map_arr[i]*1e-24

dm_map1 = nodm_map + DM_count_map_1
dm_map2 = nodm_map + DM_count_map_2
dm_map3 = nodm_map + DM_count_map_3
dm_map4 = nodm_map + DM_count_map_4

nodm_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))
dm1_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))
dm2_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))
dm3_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))
dm4_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))

for n in range(nsim):
    for i in range(len(nodm_out)):
        for j in range(len(nodm_out[0])):
            nodm_out[i,j] = np.random.poisson(nodm_map[i,j])
            dm1_out[i,j] = np.random.poisson(dm_map1[i,j])
            dm2_out[i,j] = np.random.poisson(dm_map2[i,j])
            dm3_out[i,j] = np.random.poisson(dm_map3[i,j])
            dm4_out[i,j] = np.random.poisson(dm_map4[i,j])

    #np.save(work_dir + 'FakeMaps/MC_'+str(outstring)+'_nodm_v'+str(n)+'.npy',nodm_out)
    np.save(work_dir + 'FakeMaps/MC_'+str(outstring)+'_dm1_v'+str(n)+'_t2.npy',dm1_out)
    np.save(work_dir + 'FakeMaps/MC_'+str(outstring)+'_dm2_v'+str(n)+'_t2.npy',dm2_out)
    np.save(work_dir + 'FakeMaps/MC_'+str(outstring)+'_dm3_v'+str(n)+'_t2.npy',dm3_out)
    np.save(work_dir + 'FakeMaps/MC_'+str(outstring)+'_dm4_v'+str(n)+'_t2.npy',dm4_out)
