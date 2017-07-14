#
# NAME:
#  calc_lim.py
#

import numpy as np
import healpy as hp
from global_variables import *
import sys
sys.path.insert(0, work_dir + 'AnalyseScan/code/')
from calc_limits import calc_limits

# Number of MC files
mcfiles = 100
# Array of tags associated with the runs
tagarr = ['Jci1','Jci2','Jci3','Jci4','Jci5','Jci6','Jci7','Jci8','Jci9','Jci10','Jci11','Jci12','Jci13','Jci14','Jci15']
# Array of J-maps used in the runs
jmaparr = ['Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2e+17_skyloccut']

### Basic variables
band_mask_val = 30 # At what distance to mask the plane

for jfile in range(len(jmaparr)):
    # Load J-factor map
    # Load in J-factor map, pre-smoothed and in human units (GeV**2 cm**-5)
    J_map_arr = np.load('/tigress/smsharma/public/GenMaps/SmoothedMapsLocsCut/' + jmaparr[jfile] + '.npy')

    for mc in range(mcfiles):
        # Now run in the requested mode
        # Masses in GeV
        marr = np.array([1.00000000e+02])
        output = np.zeros(shape=(len(marr),2))
        for i in range(len(marr)):
            output[i,0] = marr[i]
            output[i,1] = calc_limits(J_map=J_map_arr,tag=tagarr[jfile]+'_v'+str(mc),band_mask=band_mask_val,mass=marr[i])
        np.savetxt('./Output/' + tagarr[jfile]+'_v'+str(mc) + '-lim.dat',output)
