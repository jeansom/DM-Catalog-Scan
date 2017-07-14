#
# NAME:
#  calc_limits.py
#
# PURPOSE:
#  Calculate limits given an array of fluxes and LL values
#
# HISTORY:
#  Written by Nick Rodd, MIT, 18 July 2016

# HERE WE INTEGRATE THE LIKELIHOOD

import numpy as np
import healpy as hp
import pandas as pd
from scipy.interpolate import interp1d
from scipy import integrate
from global_variables import *

def calc_limits(J_map,tag,band_mask,mass):

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
        # But always do /dE for the full energy bin as that's how
        # the flux is normalised
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
        loadLL[np.where(loadLL == 0)[0]] = np.min(loadLL)-100
        minLL[i] = loadLL[np.where(loadflux == np.min(loadflux))[0]]
        for j in range(nbins):
            fluxarr[i,j] = loadflux[j]
            LLarr[i,j] = loadLL[j]

    ### Compute where increasing the cross section crosses the 95% limit

    # Create an array of likelihood values as a function of xsec
    xsecs = np.logspace(-33,-18,301)
    lvals = np.zeros(len(xsecs))
    # If there is no flux, LL is just the sum of the mins
    zerofluxLL = np.sum(minLL)
    for i in range(len(xsecs)):
        # Add LL up energy bin by energy bin
        LLtemp = 0.0
        for j in range(40):
            if ebins[j] < mass:
                fluxval = fluxwoxsec[j]*xsecs[i]
                if fluxval <= minflux[j]:
                    LLtemp += minLL[j]
                elif fluxval >= maxflux[j]:
                    LLtemp += -1e10
                else:
                    fluxint = interp1d(fluxarr[j],LLarr[j])
                    LLtemp += fluxint(fluxval)
            else:
                LLtemp += minLL[j]
        # Remove zero point before exponentiating or just get 0.0
        LLtemp -= zerofluxLL
        lvals[i] = np.exp(LLtemp)

    # Now normalize Likelihood to 1
    # Assume linear flat prior to correspond to the frequentist approach
    lvals_int_pn = interp1d(np.linspace(-33,-18,301),lvals)
    lnorm = integrate.quad(lambda x: np.log(10)*10**x*lvals_int_pn(x), np.log10(xsecs[0]), np.log10(xsecs[-1]))[0]
    lvals /= lnorm

    np.save('lvals.npy',lvals)

    # Now redo this integral and determine when we get 0.95
    lvals_int = interp1d(np.linspace(-33,-18,301),lvals)
    for i in range(1,len(xsecs)):
        intval = integrate.quad(lambda x: np.log(10)*10**x*lvals_int(x), np.log10(xsecs[0]), np.log10(xsecs[i]))[0]
        if intval > 0.95:
            # Figure out what fraction of distance from i-1 to i hits 0.95
            intval2 = integrate.quad(lambda x: np.log(10)*10**x*lvals_int(x), np.log10(xsecs[0]), np.log10(xsecs[i-1]))[0] 
            fracint = 1.-(intval-0.95)/(intval-intval2)
            lim = 10**(np.log10(xsecs[i-1])+fracint*(np.log10(xsecs[i])-np.log10(xsecs[i-1])))
            break
        
    ### Return the limit
    return lim
