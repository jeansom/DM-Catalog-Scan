#
# NAME:
#  run_proflike_noscan_p.py
#
# PURPOSE:
#  To use J-factor catalog maps (e.g. 2MASS) to obtain DM ID limits from
#  Fermi data or simulated data
#  Here using DarkSky
#
# HISTORY:
#  Written by Nick Rodd, MIT, 25 November 2016

import numpy as np
import healpy as hp
from global_variables import *
import sys
sys.path.insert(0, work_dir + 'PerformScan/code/')
import calc_llflux_noscan_p as clf
import argparse

### Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ebin",
                  action="store", dest="ebin", default=0, type=int)
parser.add_argument("--tag",
                  action="store",dest='tag', default="ds_boost_run",type=str)
parser.add_argument("--mcfile",
                  action="store",dest='mcfile', default="mcfile",type=str)
parser.add_argument("--jmap",
                  action="store",dest='jmap', default="DarkSky_J_true",type=str)

results=parser.parse_args()
ebin=results.ebin
tag=results.tag
mcfile=results.mcfile
jmap=results.jmap

### Basic variables
band_mask_val = 30 # At what distance to mask the plane
nside=128

# Load J-factor map
J_map_arr = np.load('/tigress/smsharma/public/GenMaps/SmoothedMapsLocsCut/' + jmap + '.npy')

# Load fake data
fake_data = np.load(work_dir + 'FakeMaps/' + mcfile)

# Now run
cli = clf.calc_llflux_noscan(J_map_arr=J_map_arr,tag=tag,band_mask=band_mask_val,external_data=fake_data,calc_flux_array=True,flux_array_ebin=ebin,bin_min=-8,bin_max=1,nbins=500)
