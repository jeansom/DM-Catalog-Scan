#
# NAME:
#  inj_spec.py
#
# PURPOSE:
#  code to return the injected spectrum
#
# HISTORY:
#  Written by Nick Rodd, MIT, 26 September 2016

import sys,os
import numpy as np
import healpy as hp
import pandas as pd
from scipy.interpolate import interp1d
from scipy import integrate
from global_variables import *

def inj_spec(J_map,band_mask,mass):

    # Flux maps are in ph/GeV/cm^2/s/sr
    # Must make sure predicted flux is at the same point

    # Flux maps were created by looking at the mean flux of the map
    # within the ROI, not including the PS mask
    # Thus want average J-factor within the ROI
    nside = hp.npix2nside(len(J_map[0]))
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside,range(npix))
    barr = np.pi/2 - theta
    keep = np.where(np.abs(barr) > band_mask*np.pi/180.)[0]

    # Get an array of means and convert from mean per pixel to per sr
    mean_J = np.zeros(len(J_map))
    for i in range(len(J_map)):
        mean_J[i] = np.mean(J_map[i][keep]) / float(hp.nside2pixarea(nside))
        #mean_J[i] = np.mean(J_map[i]) / float(hp.nside2pixarea(nside))

    ### Calculate Flux/<sigmav> in each bin
    # Load PPPC, assuming b-bbar
    dNdLogx_df=pd.read_csv(work_dir+'AdditionalData/AtProduction_gammas.dat', delim_whitespace=True)
    channel = 'b'
    dNdLogx_ann_df = dNdLogx_df.query('mDM == ' + (str(np.int(float(mass)))))[['Log[10,x]',channel]]

    Egamma = np.array(mass*(10**dNdLogx_ann_df['Log[10,x]']))
    dNdEgamma = np.array(dNdLogx_ann_df[channel]/(Egamma*np.log(10)))
    dNdE_interp = interp1d(Egamma, dNdEgamma)

    ebins=2*np.logspace(-1,3,41)
    fluxwoxsec = np.zeros(40)
    for i in range(40):
        # Only have flux if m > Ebin
        if ebins[i] < mass:
            if ebins[i+1] < mass:
                # Whole bin is inside 
                fluxwoxsec[i] = mean_J[i]/(8*np.pi*mass**2*(ebins[i+1]-ebins[i]))*integrate.quad(lambda x: dNdE_interp(x), ebins[i], ebins[i+1])[0] 
            else:
                # Bin only partially contained
                fluxwoxsec[i] = mean_J[i]/(8*np.pi*mass**2*(ebins[i+1]-ebins[i]))*integrate.quad(lambda x: dNdE_interp(x), ebins[i], mass)[0]
     
    return fluxwoxsec
