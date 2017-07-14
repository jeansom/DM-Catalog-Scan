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
#tagarr = ['Jcu1','Jcu2','Jcu3','Jcu4','Jcu5','Jcu6','Jcu7','Jcu8','Jcu9','Jcu10','Jcu11','Jcu12','Jcu13','Jcu14']
tagarr = ['Jcu14','Jcu15','Jcu16','Jcu17']
# Array of J-maps used in the runs
#jmaparr = ['Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2.47e+13_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a6.11e+13_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a1.51e+14_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a3.73e+14_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a9.21e+14_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2.28e+15_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a5.62e+15_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a1.39e+16_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a3.43e+16_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a8.48e+16_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a2.1e+17_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a5.18e+17_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a1.28e+18_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a3.16e+18_final_smoothed']
jmaparr = ['Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a1e+18_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a1.29e+18_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a1.9e+18_final_smoothed','Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a3.16e+18_final_smoothed']

### Basic variables
band_mask_val = 30 # At what distance to mask the plane

for jfile in range(len(jmaparr)):
    # Load in J-factor map, pre-smoothed and in human units (GeV**2 cm**-5)
    J_map_arr = np.load('/tigress/smsharma/public/GenMaps/GenMapsHigh/' + jmaparr[jfile] + '.npy')

    for mc in range(mcfiles):
        print "mc:",mc
        # Now run in the requested mode
        # Masses in GeV
        marr = np.array([1.00000000e+02])
        output = np.zeros(shape=(len(marr),2))
        for i in range(len(marr)):
            output[i,0] = marr[i]
            output[i,1] = calc_limits(J_map=J_map_arr,tag=tagarr[jfile]+'_v'+str(mc),band_mask=band_mask_val,mass=marr[i])
        np.savetxt('./Output/' + tagarr[jfile]+'_v'+str(mc) + '-lim.dat',output)
