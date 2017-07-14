#
# NAME:
#  calc_lim.py
#

import numpy as np
import healpy as hp
from global_variables import *
import sys
sys.path.insert(0, work_dir + 'AnalyseScan/code/')
from calc_limits_int import calc_limits
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mcval",action="store", dest="mcval", default=0,type=int)
results = parser.parse_args()
mcval=results.mcval

# Number of MC files
mcfiles = 100
# Array of tags associated with the runs
tagarr = ['all_']
# Array of J-maps used in the runs
jmaparr = ['Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a1e+13_final_smoothed']


### Basic variables
band_mask_val = 30 # At what distance to mask the plane

for jfile in range(len(jmaparr)):
    # Load in J-factor map, pre-smoothed and in human units (GeV**2 cm**-5)
    J_map_arr = np.load('/tigress/smsharma/public/DarkSkyMaps/CutMaps/' + jmaparr[jfile] + '.npy')

    #for mc in range(mcfiles):
    if 1==1:
        mc = mcval
        # Now run in the requested mode
        # Masses in GeV
        marr = np.array([1.00000000e+01,1.50000000e+01,2.00000000e+01,2.50000000e+01,3.00000000e+01,4.00000000e+01,5.00000000e+01,6.00000000e+01,7.00000000e+01,8.00000000e+01,9.00000000e+01,1.00000000e+02,1.10000000e+02,1.20000000e+02,1.30000000e+02,1.40000000e+02,1.50000000e+02,1.60000000e+02,1.80000000e+02,2.00000000e+02,2.20000000e+02,2.40000000e+02,2.60000000e+02,2.80000000e+02,3.00000000e+02,3.30000000e+02,3.60000000e+02,4.00000000e+02,4.50000000e+02,5.00000000e+02,5.50000000e+02,6.00000000e+02,6.50000000e+02,7.00000000e+02,7.50000000e+02,8.00000000e+02,9.00000000e+02,1.00000000e+03,1.10000000e+03,1.20000000e+03,1.30000000e+03,1.50000000e+03,1.70000000e+03,2.00000000e+03,2.50000000e+03,3.00000000e+03,4.00000000e+03,5.00000000e+03,6.00000000e+03,7.00000000e+03,8.00000000e+03,9.00000000e+03,1.00000000e+04])
        output = np.zeros(shape=(len(marr),2))
        for i in range(len(marr)):
            output[i,0] = marr[i]
            output[i,1] = calc_limits(J_map=J_map_arr,tag=tagarr[jfile]+'0_v'+str(mc),band_mask=band_mask_val,mass=marr[i])
        np.savetxt('./Output/' + tagarr[jfile]+'v'+str(mc) + '-lim-v2.dat',output)
