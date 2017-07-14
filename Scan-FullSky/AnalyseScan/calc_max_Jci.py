#
# NAME:
#  run-scan.py
#
# PURPOSE:
#  To use J-factor catalog maps (e.g. 2MASS) to obtain DM ID limits from
#  Fermi data or simulated data
#
# HISTORY:
#  Written by Nick Rodd, MIT, 18 July 2016

import numpy as np
import healpy as hp
from global_variables import *
import sys
sys.path.insert(0, work_dir + 'AnalyseScan/code/')
from find_max_LL import find_max_LL

# Number of MC files
mcfiles = 100
# Array of tags associated with the runs
tagarr = ['Jci1','Jci2','Jci3','Jci4','Jci5','Jci6','Jci7','Jci8','Jci9','Jci10','Jci11','Jci12','Jci13','Jci14','Jci15']
# Array of J-maps used in the runs
jmaparr = ['Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut']
# Array of cross sections to test
xarr = ['-21.0','-21.5','-22','-22.5','-23','-23.5','-24','-24.5','-25','-25.5','-26','-26.5','-27','-27.5','-28']

### Basic variables
band_mask_val = 30 # At what distance to mask the plane

for jfile in range(len(jmaparr)):
    # Load in J-factor map, pre-smoothed and in human units (GeV**2 cm**-5)
    J_map_arr = np.load('/tigress/smsharma/public/GenMaps/SmoothedMapsLocsCut/' + jmaparr[jfile] + '.npy')

    for mc in range(mcfiles):
        xsecarr = np.zeros(len(xarr))
        xmaxarr = np.zeros(len(xarr))
        xminarr = np.zeros(len(xarr))
        xtsarr = np.zeros(len(xarr))
        tsarr = np.zeros(len(xarr))

        for xval in range(len(xarr)):
            # Find max
            maxLL = -1e10
            maxm = 0
            maxxsec = 0

            marr = [100]
            xsecs = np.logspace(-33,-18,301) 
            for i in range(len(marr)):
                LLs = find_max_LL(J_map=J_map_arr,tag=tagarr[jfile]+'_v'+str(mc),band_mask=band_mask_val,mass=marr[i])
                if (max(LLs) > maxLL):
                    maxLL = max(LLs)
                    maxm = marr[i]
                    maxxsec = xsecs[np.where(LLs == max(LLs))[0]]
                # 2 sigma error; LL = sigma**2/2
                finderror = LLs-maxLL+2.0
                keep = np.where(finderror > 0)[0]
                minerr = np.min(xsecs[keep])
                maxerr = np.max(xsecs[keep])
                errp = np.log10(maxerr)-np.log10(maxxsec[0])
                errm = np.log10(maxxsec[0])-np.log10(minerr)
                TS = 2*(maxLL-LLs[0])

                # Only set central value if TS > 9
                if TS > 9:
                    xtsarr[xval] = 1.0
                
                xsecarr[xval] = np.log10(maxxsec[0])
                xmaxarr[xval] = errp
                xminarr[xval] = errm
                tsarr[xval] = np.log10(TS)
        
        outdir ='/tigress/nrodd/2mass2furious/AnalyseScan/Output/'
        np.savetxt(outdir + tagarr[jfile]+'_v'+str(mc) + '_xs.dat',xsecarr)
        np.savetxt(outdir + tagarr[jfile]+'_v'+str(mc) + '_xp.dat',xmaxarr)
        np.savetxt(outdir + tagarr[jfile]+'_v'+str(mc) + '_xm.dat',xminarr)
        np.savetxt(outdir + tagarr[jfile]+'_v'+str(mc) + '_xts.dat',xtsarr)
        np.savetxt(outdir + tagarr[jfile]+'_v'+str(mc) + '_ts.dat',tsarr)
