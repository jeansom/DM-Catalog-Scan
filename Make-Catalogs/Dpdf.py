###############################################################################
# Dpdf.py
###############################################################################
# HISTORY:
#   2017-06-07 - Written - Nick Rodd (MIT)
###############################################################################

# Using the point like approximation for the D-factor of a cluster or galaxy
# gives a PDF for the D-factor that is log normal. This code calculates the 
# mean and standard deviation of log10(D)

# Note for the decay factor the result is independent of the halo profile

# Throughout we use the following notation:
#  - z = log10(D)
#  - x = log10(M)

import numpy as np
import halo as hl

class Dc:
    def __init__(self):
        
        self.hm = hl.HaloModel(boost_model='bartels', concentration_model='correa_Planck15')

    def z(self, x, dA):
        """ z = log10(D) for a given x and dA
        """
        
        return -2.*np.log10(dA) + x 
        
    def dzdx(self, x):
        """ dz/dx
        """

        return 1. 

    def log10Dmusig(self, mux, sigmax, red):
        """ mu and sigma associated with log normal distribution D follows
        
        Input:
            - mux: mean of log10(M [Msun])
            - sigmax: std. dev. of log10(M [Msun])
            - red: redshift

        Returns:
            - mean of log10(D)
            - std. dev. of log10(D)
        """

        # Convert masses from [Msun] to [GeV]
        Msun = 1.99*10**30*5.6085*10**35*10**-9 # GeV
        mux += np.log10(Msun)
        
        # Calculate distance to object
        dA = self.hm.universe.angular_diameter_distance(red) # Mpc
        dA *= 3.086 * 10**24 # cm
        
        # Calculate boosted values
        muz = self.z(mux, dA)
        sigmaz = np.abs(self.dzdx(mux))*sigmax 
        
        return np.array([muz, sigmaz])
