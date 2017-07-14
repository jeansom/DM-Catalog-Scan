#
# NAME:
#  get_spec.py

import numpy as np
import healpy as hp
from global_variables import *
import sys
sys.path.insert(0, work_dir + 'AnalyseScan/code/')
from inj_spec import inj_spec
from find_max_flux import find_max_flux

# Number of MC files
mcfiles = 100
# Array of tags associated with the runs
tagarr = ['Jci1','Jci3','Jci5']
# Array of J-maps used in the runs
jmaparr = ['Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut']
# Array of cross sections to test
xarr = ['-21.0','-22.0','-23.0']
xarrd = np.linspace(-21.,-23.,3)

### Basic variables
band_mask_val = 30 # At what distance to mask the plane
nside=128
eventclass=5 # 2 (Source) or 5 (UltracleanVeto)
eventtype=0 # 0 (all), 3 (bestpsf) or 5 (top3 quartiles)
emin_bin=0
emax_bin=40 # Must match the norm file!

evals = 2*10**(np.linspace(-1,3,41)+0.05)[0:40] 

for jfile in range(len(jmaparr)):
    J_map_smoothed = np.load('/tigress/smsharma/public/GenMaps/SmoothedMapsLocsCut/Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut.npy')
    J_map_unsmoothed = hp.ud_grade(np.load('/tigress/smsharma/public/GenMaps/SmoothedMapsLocsCut/Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut_unsmoothed.npy'),nside,power=-2)
    #J_map_unsmoothed = hp.ud_grade(np.load('/tigress/smsharma/public/GenMaps/SmoothedMapsLocsCut/Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a1e+13_final.npy'),nside,power=-2)

    # unsmoothed map in natural units, convert
    GeV = 10**6
    Centimeter = 5.0677*10**13/GeV
    J_map_unsmoothed /= GeV**2*Centimeter**-5

    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside,range(npix))
    barr = np.pi/2 - theta
    keep = np.where(np.abs(barr) > band_mask_val*np.pi/180.)[0]

    J_ratio = np.zeros(emax_bin-emin_bin)
    J_map_inj_arr = np.zeros(shape=(emax_bin-emin_bin,len(J_map_unsmoothed)))
    for en in range(emax_bin-emin_bin):
        J_ratio[en] = np.mean(J_map_unsmoothed)/np.mean(J_map_smoothed[en][keep])
        J_map_inj_arr[en] = J_map_unsmoothed

    # Get the injected spectrum independent of cross section
    fluxwoxsec = inj_spec(J_map=J_map_inj_arr,band_mask=band_mask_val,mass=100)

    if jfile == 0:
        injflux = np.zeros(shape=(40,2))
        injflux[:,0] = evals
        injflux[:,1] = fluxwoxsec*10**xarrd[jfile]
        np.savetxt('./Output/'+tagarr[jfile]+'-inj-spec.dat',injflux)
    
    # Now get the max flux values
    #for xval in range(len(xarr)):
    loadflux = np.zeros(shape=(mcfiles,40))
    for mc in range(mcfiles):
        # Adjust for J-factor
        loadflux[mc] = find_max_flux(tag=tagarr[jfile]+'_v'+str(mc))*J_ratio

    outflux = np.zeros(shape=(40,4))
    for i in range(40):
        outflux[i,0] = evals[i]
        outflux[i,1] = np.percentile(loadflux[:,i],50)
        #outflux[i,2] = np.percentile(loadflux[:,i],100)-np.percentile(loadflux[:,i],50)
        #outflux[i,3] = np.percentile(loadflux[:,i],50)-np.percentile(loadflux[:,i],0)
        outflux[i,2] = np.max(loadflux[:,i])-np.percentile(loadflux[:,i],50)
        outflux[i,3] = np.percentile(loadflux[:,i],50)-np.min(loadflux[:,i])

    np.savetxt('./Output/'+tagarr[jfile]+'-out-spec.dat',outflux)
