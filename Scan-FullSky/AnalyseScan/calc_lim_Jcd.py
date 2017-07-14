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
tagarr = ['Jcd1','Jcd2','Jcd3','Jcd4','Jcd5','Jcd6','Jcd7','Jcd8','Jcd9','Jcd10','Jcd11','Jcd12','Jcd13','Jcd14']
# Array of J-maps used in the runs
jmaparr = ['Jfactor_DS_true_map_200.0_200.0_200.0b3.16e+18_a1e+13_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b1.28e+18_a1e+13_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b5.18e+17_a1e+13_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b2.1e+17_a1e+13_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b8.48e+16_a1e+13_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b3.43e+16_a1e+13_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b1.39e+16_a1e+13_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b5.62e+15_a1e+13_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b2.28e+15_a1e+13_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b9.21e+14_a1e+13_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b3.73e+14_a1e+13_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b1.51e+14_a1e+13_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b6.11e+13_a1e+13_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b2.47e+13_a1e+13_final_smoothed']

### Basic variables
band_mask_val = 30 # At what distance to mask the plane

for jfile in range(len(jmaparr)):
    # Load in J-factor map, pre-smoothed and in human units (GeV**2 cm**-5)
    J_map_arr = np.load('/tigress/smsharma/public/DarkSkyMaps/CutMaps/' + jmaparr[jfile] + '.npy')

    for mc in range(mcfiles):
        # Now run in the requested mode
        # Masses in GeV
        marr = np.array([1.00000000e+02])
        output = np.zeros(shape=(len(marr),2))
        for i in range(len(marr)):
            output[i,0] = marr[i]
            output[i,1] = calc_limits(J_map=J_map_arr,tag=tagarr[jfile]+'_v'+str(mc),band_mask=band_mask_val,mass=marr[i])
        np.savetxt('./Output/' + tagarr[jfile]+'_v'+str(mc) + '-lim.dat',output)
