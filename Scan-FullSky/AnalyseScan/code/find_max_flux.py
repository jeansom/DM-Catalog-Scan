#
# NAME:
#  find_max_flux.py
#
# PURPOSE:
#  return maximum flux
#
# HISTORY:
#  Written by Nick Rodd, MIT, 26 September 2016

import sys,os
import numpy as np
from global_variables import *

def find_max_flux(tag):

    ### Setup Flux -> LL arrays
    # Load one to determine binning: 
    maxfluxarr = np.zeros(40)
    for i in range(40):
        loadflux = np.load(work_dir+'ScanOutput/'+tag+'/Flux_array-'+str(i)+'.npy')
        loadLL = np.load(work_dir+'ScanOutput/'+tag+'/LL_array-'+str(i)+'.npy')
        maxfluxarr[i] = loadflux[np.where(loadLL == np.max(loadLL))[0]]

    return maxfluxarr
