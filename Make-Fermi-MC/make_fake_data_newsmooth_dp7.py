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

# Set up J_map - this is in human units and nside=128 
J_map_arr = np.load('/tigress/nrodd/2mass2furious/AccurateSmoothing/SmoothedMaps/Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a1e+13_final_smoothed.npy')

# Correct for exposure
J_map_arr *= f_global.CTB_exposure_maps

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
DM_count_map_5=np.zeros((emax_bin-emin_bin,hp.nside2npix(nside)))
DM_count_map_6=np.zeros((emax_bin-emin_bin,hp.nside2npix(nside)))
DM_count_map_7=np.zeros((emax_bin-emin_bin,hp.nside2npix(nside)))
DM_count_map_8=np.zeros((emax_bin-emin_bin,hp.nside2npix(nside)))
DM_count_map_9=np.zeros((emax_bin-emin_bin,hp.nside2npix(nside)))
DM_count_map_10=np.zeros((emax_bin-emin_bin,hp.nside2npix(nside)))
DM_count_map_11=np.zeros((emax_bin-emin_bin,hp.nside2npix(nside)))
DM_count_map_12=np.zeros((emax_bin-emin_bin,hp.nside2npix(nside)))
DM_count_map_13=np.zeros((emax_bin-emin_bin,hp.nside2npix(nside)))
DM_count_map_14=np.zeros((emax_bin-emin_bin,hp.nside2npix(nside)))
DM_count_map_15=np.zeros((emax_bin-emin_bin,hp.nside2npix(nside)))

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
    DM_count_map_2[i] = PPnoxsec*J_map_arr[i]*3.16227766017e-22
    DM_count_map_3[i] = PPnoxsec*J_map_arr[i]*1e-22
    DM_count_map_4[i] = PPnoxsec*J_map_arr[i]*3.16227766017e-23
    DM_count_map_5[i] = PPnoxsec*J_map_arr[i]*1e-23
    DM_count_map_6[i] = PPnoxsec*J_map_arr[i]*3.16227766017e-24
    DM_count_map_7[i] = PPnoxsec*J_map_arr[i]*1e-24
    DM_count_map_8[i] = PPnoxsec*J_map_arr[i]*3.16227766017e-25
    DM_count_map_9[i] = PPnoxsec*J_map_arr[i]*1e-25
    DM_count_map_10[i] = PPnoxsec*J_map_arr[i]*3.16227766017e-26
    DM_count_map_11[i] = PPnoxsec*J_map_arr[i]*1e-26
    DM_count_map_12[i] = PPnoxsec*J_map_arr[i]*3.16227766017e-27
    DM_count_map_13[i] = PPnoxsec*J_map_arr[i]*1e-27
    DM_count_map_14[i] = PPnoxsec*J_map_arr[i]*3.16227766017e-28
    DM_count_map_15[i] = PPnoxsec*J_map_arr[i]*1e-28

dm_map1 = nodm_map + DM_count_map_1
dm_map2 = nodm_map + DM_count_map_2
dm_map3 = nodm_map + DM_count_map_3
dm_map4 = nodm_map + DM_count_map_4
dm_map5 = nodm_map + DM_count_map_5
dm_map6 = nodm_map + DM_count_map_6
dm_map7 = nodm_map + DM_count_map_7
dm_map8 = nodm_map + DM_count_map_8
dm_map9 = nodm_map + DM_count_map_9
dm_map10 = nodm_map + DM_count_map_10
dm_map11 = nodm_map + DM_count_map_11
dm_map12 = nodm_map + DM_count_map_12
dm_map13 = nodm_map + DM_count_map_13
dm_map14 = nodm_map + DM_count_map_14
dm_map15 = nodm_map + DM_count_map_15

nodm_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))
dm1_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))
dm2_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))
dm3_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))
dm4_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))
dm5_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))
dm6_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))
dm7_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))
dm8_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))
dm9_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))
dm10_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))
dm11_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))
dm12_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))
dm13_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))
dm14_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))
dm15_out = np.zeros(shape=(len(nodm_map),hp.nside2npix(nside)))

for n in range(nsim):
    for i in range(len(nodm_out)):
        for j in range(len(nodm_out[0])):
            nodm_out[i,j] = np.random.poisson(nodm_map[i,j])
            dm1_out[i,j] = np.random.poisson(dm_map1[i,j])
            dm2_out[i,j] = np.random.poisson(dm_map2[i,j])
            dm3_out[i,j] = np.random.poisson(dm_map3[i,j])
            dm4_out[i,j] = np.random.poisson(dm_map4[i,j])
            dm5_out[i,j] = np.random.poisson(dm_map5[i,j])
            dm6_out[i,j] = np.random.poisson(dm_map6[i,j])
            dm7_out[i,j] = np.random.poisson(dm_map7[i,j])
            dm8_out[i,j] = np.random.poisson(dm_map8[i,j])
            dm9_out[i,j] = np.random.poisson(dm_map9[i,j])
            dm10_out[i,j] = np.random.poisson(dm_map10[i,j])
            dm11_out[i,j] = np.random.poisson(dm_map11[i,j])
            dm12_out[i,j] = np.random.poisson(dm_map12[i,j])
            dm13_out[i,j] = np.random.poisson(dm_map13[i,j])
            dm14_out[i,j] = np.random.poisson(dm_map14[i,j])
            dm15_out[i,j] = np.random.poisson(dm_map15[i,j])

    np.save(work_dir + 'FakeMaps/MC_'+str(outstring)+'_nodm_v'+str(n)+'.npy',nodm_out)
    #np.save(work_dir + 'FakeMaps/MC_'+str(outstring)+'_dm1_v'+str(n)+'.npy',dm1_out)
    #np.save(work_dir + 'FakeMaps/MC_'+str(outstring)+'_dm2_v'+str(n)+'.npy',dm2_out)
    #np.save(work_dir + 'FakeMaps/MC_'+str(outstring)+'_dm3_v'+str(n)+'.npy',dm3_out)
    #np.save(work_dir + 'FakeMaps/MC_'+str(outstring)+'_dm4_v'+str(n)+'.npy',dm4_out)
    #np.save(work_dir + 'FakeMaps/MC_'+str(outstring)+'_dm5_v'+str(n)+'.npy',dm5_out)
    #np.save(work_dir + 'FakeMaps/MC_'+str(outstring)+'_dm6_v'+str(n)+'.npy',dm6_out)
    #np.save(work_dir + 'FakeMaps/MC_'+str(outstring)+'_dm7_v'+str(n)+'.npy',dm7_out)
    #np.save(work_dir + 'FakeMaps/MC_'+str(outstring)+'_dm8_v'+str(n)+'.npy',dm8_out)
    #np.save(work_dir + 'FakeMaps/MC_'+str(outstring)+'_dm9_v'+str(n)+'.npy',dm9_out)
    #np.save(work_dir + 'FakeMaps/MC_'+str(outstring)+'_dm10_v'+str(n)+'.npy',dm10_out)
    #np.save(work_dir + 'FakeMaps/MC_'+str(outstring)+'_dm11_v'+str(n)+'.npy',dm11_out)
    #np.save(work_dir + 'FakeMaps/MC_'+str(outstring)+'_dm12_v'+str(n)+'.npy',dm12_out)
    #np.save(work_dir + 'FakeMaps/MC_'+str(outstring)+'_dm13_v'+str(n)+'.npy',dm13_out)
    #np.save(work_dir + 'FakeMaps/MC_'+str(outstring)+'_dm14_v'+str(n)+'.npy',dm14_out)
    #np.save(work_dir + 'FakeMaps/MC_'+str(outstring)+'_dm15_v'+str(n)+'.npy',dm15_out)
