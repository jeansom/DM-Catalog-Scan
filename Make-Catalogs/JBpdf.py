###############################################################################
# JBpdf.py
###############################################################################
# HISTORY:
#   2017-05-14 - Written - Nick Rodd (MIT)
###############################################################################

# Using the point like approximation for the J-factor of a cluster or galaxy
# gives a PDF for the J-factor that is approximately log normal. This code
# calculates the mean and standard deviation of log10(J)

# Code here assumes a Burkert halo profile, but input parameters concentration
# (if any) should be for an NFW

# Throughout we use the following notation:
#  - z = log10(J)
#  - x = log10(M)
#  - y = log10(c)

import numpy as np
import halo as hl

class Jc:
    def __init__(self):
        
        self.hm = hl.HaloModel(boost_model='bartels', concentration_model='correa_Planck15')

    def z(self, x, y, dA, rvir, red):
        """ z = log10(J) for a given x, y, dA and rvir for the boosted case
        """
        
        # NB: when using hm.bsh, need to convert mass from GeV to keV
        return np.log10((1. + self.hm.bsh(10**(x+6.),red)) / (np.pi * dA**2 * rvir**3)) \
            + 2*x + 3*y \
            + np.log10( 10**y*(1. + 10**y + 2*10**(2*y))/(1. + 10**y + 10**(2*y) + 10**(3*y)) \
                       - np.arctan(10**y) ) \
            - 2*np.log10(np.log((1. + 10**y)**2*(1+10**(2*y))) - 2*np.arctan(10**y))

    def znb(self, x, y, dA, rvir):
        """ z = log10(J) for a given x, y, dA and rvir for the unboosted case
        """

        return np.log10(1. / (np.pi * dA**2 * rvir**3)) \
            + 2*x + 3*y \
            + np.log10( 10**y*(1. + 10**y + 2*10**(2*y))/(1. + 10**y + 10**(2*y) + 10**(3*y)) \
                       - np.arctan(10**y) ) \
            - 2*np.log10(np.log((1. + 10**y)**2*(1+10**(2*y))) - 2*np.arctan(10**y)) 

    def dznbdx(self, x, y):
        """ dz/dx for the unboosted case - must remember rvir depnds on x
        """

        return 1.

    def dzdx(self, x, y, red, dratio=1000000.):
        """ dz/dx for the boosted case - must remember rvir depnds on x
        """

        # Evaluate derivative manually
        dval = (self.hm.bsh(10**(x+6.+x/dratio),red) - self.hm.bsh(10**(x+6.),red)) / \
               (x/dratio)

        return 1. + dval/(1+self.hm.bsh(10**(x+6.),red))

    def dzdy(self, x, y):
        """ dz/dy for the (un)boosted case - same for either one
        """

        return 3. - 2**(2+3*y)*125**y / (1. + 10**y) / (1. + 100**y) / \
               ( -10**y * (1. + 10**y + 2**(1.+2*y)*25**y)
               + (1. + 10**y) * (1 + 100**y) * np.arctan(10**y)) \
               + 8**(1.+y)*125**y / (1. + 10**y) / (1. + 100**y) / \
               ( 2 * np.arctan(10**y) - np.log((1. + 10**y)**2*(1. + 100**y)))

    def log10Jmusig(self, mux, sigmax, muy, sigmay, red):
        """ mu and sigma associated with log normal distribution J follows
        
        Input:
            - mux: mean of log10(M [Msun])
            - sigmax: std. dev. of log10(M [Msun])
            - muy: mean of log10(c)
            - sigmay: std. dev. of log10(c)
            - red: redshift

        Returns:
            - mean of log10(Jnb)
            - std. dev. of log10(Jnb)
            - mean of log10(J)
            - std. dev. of log10(J)
        """

        # Convert concentration from input NFW value to Burkert
        muy -= np.log10(0.666511)
        
        # Convert masses from [Msun] to [GeV]
        Msun = 1.99*10**30*5.6085*10**35*10**-9 # GeV
        mux += np.log10(Msun)

        # Calculate distance to object
        dA = self.hm.universe.angular_diameter_distance(red) # Mpc
        dA *= 3.086 * 10**24 # cm
         
        # Calculate rvir
        self.hm.rvir = -999.
        # Note Sids code expects masses in keV
        murvir = self.hm.r_vir(10**(mux+6), red) # 1/keV
        hbarc = (6.582119514*10**(-16))*(299792458)*10**-3*10**2 # keV.cm
        murvir *= hbarc # cm

        # Calculate unboosted values
        muznb = self.znb(mux, muy, dA, murvir)
        sigmaznb = np.sqrt( (np.abs(self.dznbdx(mux, muy))*sigmax)**2
                        + (np.abs(self.dzdy(mux, muy))*sigmay)**2 )

        # Calculate boosted values
        muz = self.z(mux, muy, dA, murvir, red)
        sigmaz = np.sqrt( (np.abs(self.dzdx(mux, muy, red))*sigmax)**2
                        + (np.abs(self.dzdy(mux, muy))*sigmay)**2 )


        return np.array([muznb, sigmaznb, muz, sigmaz])
