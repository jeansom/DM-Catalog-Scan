###############################################################################
# Dangpdf.py
###############################################################################
# HISTORY:
#   2017-06-22 - Written - Nick Rodd (MIT)
###############################################################################

# Code to calculate the D-factor and the associated error for the case where
# we do not integrate over the full halo, to account for a finite ROI effect

# Code here assumes a NFW halo profile

# Throughout we use the following notation:
#  - z = log10(J)
#  - x = log10(M)
#  - y = log10(c)

import numpy as np
import halo as hl

class Dc:
    def __init__(self):

        self.hm = hl.HaloModel(boost_model='bartels', concentration_model='correa_Planck15')

    def zfull(self, x, dA):
        """ z = log10(D) for a given x and dA for the full halo
        """

        return -2.*np.log10(dA) + x

    def dzdxfull(self, x):
        """ dz/dx for the full halo
        """

        return 1.

    def zpartial(self, x, y, rang, murvir, dA):
        """ z = log10(D) for a given x, y, rang, murvir, and dA for a partial halo
        """

        return -2.*np.log10(dA) + x + np.log10(np.log(1. + rang/murvir*10**y) 
               - 10**y / (murvir/rang + 10**y)) \
               - np.log10(np.log(1. + 10**y) - 10**y / (1. + 10**y))

    def dzdxpartial(self, x, y, rang, murvir):
        """ dz/dx for the partial halo
        """

        return 1. - 10**(2*y)*rang**2 / (3*(murvir + rang*10**y) *
               (murvir + rang*10**y) * np.log(1. + 10**y*rang/murvir) 
               - rang*10**y) 

    def dzdypartial(self, x, y, rang, murvir):
        """ dz/dy for the partial halo
        """

        return 10**(2*y)*rang**2 / ((murvir + rang*10**y) *
               (murvir + rang*10**y) * np.log(1. + 10**y*rang/murvir)\
               - rang*10**y)\
               - 10**(2*y) / ((1. + 10**y)*((1. + 10**y)*np.log(1. + 10**y)\
               - 10**y))

    def log10Dmusig(self, mux, sigmax, muy, sigmay, red, theta):
        """ mu and sigma associated with log normal distribution J follows

        Input:
            - mux: mean of log10(M [Msun])
            - sigmax: std. dev. of log10(M [Msun])
            - muy: mean of log10(c)
            - sigmay: std. dev. of log10(c)
            - red: redshift
            - theta: distance integrated out from center (degrees)

        Returns:
            - mean of log10(Jnb)
            - std. dev. of log10(Jnb)
            - mean of log10(J)
            - std. dev. of log10(J)
        """

        # Convert masses from [Msun] to [GeV]
        Msun = 1.99*10**30*5.6085*10**35*10**-9 # GeV
        mux += np.log10(Msun)

        # Calculate distance to object
        dA = self.hm.universe.angular_diameter_distance(red) # Mpc
        dA *= 3.086 * 10**24 # cm

        # Calculate distance we integrate out to
        theta = theta*np.pi/180. # radians
        rang = dA*theta # cm
        
        # Calculate rvir
        self.hm.rvir = -999.
        # Note Sids code expects masses in keV
        murvir = self.hm.r_vir(10**(mux+6), red) # 1/keV
        hbarc = (6.582119514*10**(-16))*(299792458)*10**-3*10**2 # keV.cm
        murvir *= hbarc # cm

        # Determine rho_c*Delta_c(z)
        rhocDeltac = self.hm.rho_c * self.hm.Delta_c(red)

        # Calculate the D factor and associated error
        if rang >= murvir: # Have entire halo, return simple value
            muz = self.zfull(mux, dA)
            sigmaz = np.abs(self.dzdxfull(mux))*sigmax
        else: # Need the partial machinery
            muz = self.zpartial(mux, muy, rang, murvir, dA) 
            sigmaz = np.sqrt( (np.abs(self.dzdxpartial(mux, muy, rang, murvir))*sigmax)**2
                        + (np.abs(self.dzdypartial(mux, muy, rang, murvir))*sigmay)**2 ) 
        
        return np.array([muz, sigmaz])
