#
# NAME:
#  find_max_LL.py
#
# PURPOSE:
#  return full LL array versus xsec so we can determine maximum LL 
#
# HISTORY:
#  Written by Nick Rodd, MIT, 18 July 2016

import sys,os
import numpy as np
import healpy as hp
import pandas as pd
from scipy.interpolate import interp1d
from scipy import integrate
from global_variables import *

def find_max_LL(J_map,tag,band_mask,mass):

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

    ### Setup Flux -> LL arrays
    # Load one to determine binning: 
    nbins = len(np.load(work_dir+'ScanOutput/'+tag+'/LL_array-0.npy'))
    fluxarr = np.zeros(shape=(40,nbins))
    minflux = np.zeros(40)
    maxflux = np.zeros(40)
    LLarr = np.zeros(shape=(40,nbins))
    minLL = np.zeros(40)
    for i in range(40):
        loadflux = np.load(work_dir+'ScanOutput/'+tag+'/Flux_array-'+str(i)+'.npy')
        minflux[i] = np.min(loadflux)
        maxflux[i] = np.max(loadflux)
        loadLL = np.load(work_dir+'ScanOutput/'+tag+'/LL_array-'+str(i)+'.npy')
        # Replace 0 values
        #loadLL[np.where(loadLL == 0)[0]] = np.min(loadLL)-100
        minLL[i] = loadLL[np.where(loadflux == np.min(loadflux))[0]]
        for j in range(nbins):
            fluxarr[i,j] = loadflux[j]
            LLarr[i,j] = loadLL[j]

    # Calculate LL vals versus xsec
    # First compute Delta LL values
    xsecs = np.logspace(-33,-18,301)
    DLL2vals = np.zeros(len(xsecs))
    for i in range(len(xsecs)):
        # Add LL up energy bin by energy bin
        for j in range(40):
            if ebins[j] < mass:
                fluxval = fluxwoxsec[j]*xsecs[i]
                if fluxval <= minflux[j]:
                    DLL2vals[i] += minLL[j]
                elif fluxval >= maxflux[j]:
                    DLL2vals[i] += -1e10
                else:
                    fluxint = interp1d(fluxarr[j],LLarr[j])
                    DLL2vals[i] += fluxint(fluxval)

            else:
                DLL2vals[i] += minLL[j]

    return DLL2vals
