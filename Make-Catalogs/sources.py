"""
Module for calculating the source counts and intensities of 
astrophysical objects (blazar, SFG, mAGN) to the Fermi EGRB.

Created by Siddharth Mishra-Sharma. Last modified 05/04/2016
"""

import sys, os
import collections
import functools
import random

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.interpolate as ip
from scipy import integrate
import pandas as pd
import healpy as hp
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from scipy.special import hyp2f1

# Monte-carlo integration modules
import mcint
import vegas
from skmonaco import mcquad

from constants import *
import CosmologicalDistance as cld

class memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)

class LuminosityFunctionBL:
    """
    Class for calculating the luminosity function, source count function and intensity spectrum of blazars. 
    Based primarily on 1501.05301 (Ajello et al).
    """
    def __init__(self, model = 'blazars'):

        # Integration ranges, modified from equation (12)
        # self.Lmin = 10**43*erg*Sec**-1
        # self.Lmax = 10**51*erg*Sec**-1 # Upper limit taken down slightly to help with integration convergence
        # self.zmin = 10**-3
        # self.zmax = 6.
        # self.Gammamin=1.
        # self.Gammamax=3.5

        self.Lmin = 7*10**43*erg*Sec**-1
        self.Lmax = 10**49*erg*Sec**-1 # Upper limit taken down slightly to help with integration convergence
        self.zmin = 0.04
        self.zmax = 6.
        self.Gammamin=1.45
        self.Gammamax=2.80

        self.model = model

        self.cd = cld.CosmologicalDistance() # For cosmological distance calculations
        if self.model == 'blazars':
            self.params_BLL_FSRQ() # Initialize LF parameters
        if self.model == 'fsrq':
            self.params_FSRQ() # Initialize LF parameters
        if self.model == 'bll1':
            self.params_BLLacs1() # Initialize LF parameters

        # self.Gamma = self.mustar # For now, setting spectral index to average value
        self.make_Tau() # Create EBL interpolation based on Finke et al

    def params_BLL_FSRQ(self):
        """
        Parameters for blazars (BL Lacs + FSRQs), taken from Table 1 of 1501.05301 (model 1)
        assuming Luminosity-Dependent Density Evolution (LDDE)
        """
        self.A = 196*10**-2*(1000*Mpc)**-3*erg**-1*Sec
        self.gamma1 = 0.50
        self.Lstar = (1.05)*10**48*erg*Sec**-1
        self.gamma2 = 1.83
        self.zcstar = 1.25
        self.p1star = 3.39
        self.tau = 3.16
        self.p2star = -4.96
        self.alpha = 7.23*10**-2
        self.mustar = 2.22
        self.beta = 0.10
        self.sigma = 0.28
        self.delta = 0.64

    def params_FSRQ(self):
        """
        Parameters for FSRQs, taken from Table 3 of 1110.3787
        assuming Luminosity Dependent Density Evolution (LDDE)
        """
        self.A = (3.06*10**4)*(10**-13*Mpc**-3*erg**-1*Sec)
        self.gamma1 = 0.21
        self.Lstar = (.84)*10**48*erg*Sec**-1
        self.gamma2 = 1.58
        self.zcstar = 1.47
        self.p1star = 7.35
        self.tau = 0.
        self.p2star = -6.51
        self.alpha = 0.21
        self.mustar = 2.44
        self.beta = 0.
        self.sigma = 0.18
        self.delta = 0

    # def params_BLLacs1(self):
    #     """
    #     Parameters for BL Lacs, taken from Table 3 of 1301.0006 (model 1)
    #     assuming Luminosity Dependent Density Evolution (LDDE)
    #     """
    #     self.A = (9.20*10**2)*(10**-13*Mpc**-3*erg**-1*Sec)
    #     self.gamma1 = 1.12
    #     self.Lstar = (2.43)*10**48*erg*Sec**-1
    #     self.gamma2 = 3.71
    #     self.zcstar = 1.67
    #     self.p1star = 4.50
    #     self.tau = 0.
    #     self.p2star = -12.88
    #     self.alpha = 4.46*10**-2
    #     self.mustar = 2.12
    #     self.beta = 6.04*10**-2
    #     self.sigma = 0.26
    #     self.delta = 0

    def params_BLLacs1(self):
        """
        Parameters for BL Lacs, taken from Table 3 of 1301.0006 (model 2)
        assuming Luminosity Dependent Density Evolution (LDDE)
        """
        self.A = (3.39*10**4)*(10**-13*Mpc**-3*erg**-1*Sec)
        self.gamma1 = 0.27
        self.Lstar = (.28)*10**48*erg*Sec**-1
        self.gamma2 = 1.86
        self.zcstar = 1.34
        self.p1star = 2.24
        self.tau = 4.92
        self.p2star = -7.37
        self.alpha=4.53*10**-2
        self.mustar = 2.10
        self.beta = 6.46*10**-2
        self.sigma = 0.26
        self.delta = 0

    # @memoized
    def phi_LDDE(self, L, z, Gamma):
        """
        Parameterization of Luminosity-Dependent Density Evolution (LDDE) model. Equations (10)-(20) of Ajello et al.
        Returns Phi(L,V,Gamma) = d^3N/(dLdVdGamma).
        """
        self.zc = self.zcstar*(L/(10**48*erg*Sec**-1))**self.alpha
        self.p1 = self.p1star+self.tau*(np.log10(L/(erg*Sec**-1)) - 46)
        self.p2 = self.p2star+self.delta*(np.log10(L/(erg*Sec**-1)) - 46)
        self.e = (((1+z)/(1+self.zc))**-self.p1+((1+z)/(1+self.zc))**-self.p2)**-1
        self.mu = self.mustar + self.beta*(np.log10(L/(erg*Sec**-1))-46)
        
        self.phi = (self.A/(np.log(10)*L/(erg*Sec**-1)))*((L/self.Lstar)**self.gamma1+(L/self.Lstar)**self.gamma2)**-1*self.e*np.exp(-(Gamma - self.mu)**2/(2*self.sigma**2))
        return self.phi

    def make_Tau(self):
        """
        Create EBL interpolation (model based on Finke et al)
        """

        # Load and combined EBL files downloaded from http://www.phy.ohiou.edu/~finke/EBL/ 
        tau_files = '/group/hepheno/smsharma/Fermi_High_Latitudes/Fermi-HighLat/sim/tau_modelC_total/tau_modelC_total_z%.2f.dat'
        z_list = np.arange(0, 5, 0.01) # z values to load files for
        E_list, tau_list = [],[]
        for z in z_list:
            d = np.genfromtxt(tau_files % z, unpack=True)
            E_list = d[0]
            tau_list.append(d[1])
        self.Tau_ip = ip.RectBivariateSpline(z_list, np.log10(E_list), np.log10(np.array(tau_list))) # Create interpolation
    
    def Tau(self,E,z):
        """
        EBL attenuation of gamma rays using Finke et al
        """
        return np.float64(10**self.Tau_ip(z, np.log10(E/(1000*GeV))))
        # return (z/3.3)*(E/(10*GeV))**.8 # Analytic approximation from 1506.05118

    # def dFdE(self, E,z, L, Gamma):
    #     """
    #     Intrinsic flux of source. Use simple power law.
    #     """
    #     E1 = 100*MeV
    #     E2 = 100*GeV
    #     dL = self.cd.luminosity_distance(z)*Mpc

    #     N = (1+z)**(2-Gamma)*L/(4*np.pi*dL**2)/((1/E1)**-Gamma*(E2**(-Gamma+2)-E1**(-Gamma+2))/(-Gamma+2)) # Check it double deck it
    #     return N*((E/E1)**-Gamma)

    def Eb(self, Gamma):
        """
        From Ajello et al (text in page 7)
        """
        return 10**(9.25 - 4.11*Gamma)*GeV

    # @memoized
    def dFdE(self, E, z, L, Gamma):
        """
        Intrinsic flux of source for the full Ajello spectrum
        """
        gammaa=1.7
        gammab=2.6

        Kcorr = (1+z)**(2-Gamma)

        N = L/(4*np.pi*(self.cd.luminosity_distance(z)*Mpc)**2)/(0.00166667*self.Eb(Gamma)**2.6*(hyp2f1(0.666667, 1., 1.66667, -0.0000316228*self.Eb(Gamma)**0.9) - 0.0158489*hyp2f1(0.666667, 1.,1.66667, -6.30957*10**-8*self.Eb(Gamma)**0.9)))
        
        return N*Kcorr*((E/self.Eb(Gamma))**gammaa+(E/self.Eb(Gamma))**gammab)**-1*np.exp(-self.Tau(self.E,z)) # Includes EBL suppression
    
    """
    ************************************************************
    Monte carlo integration experimentation
    """

    def dIdE_integrand(self, x):
        Gamma = x[0]
        L = x[1]
        z = x[2]
        return self.dVdz(z)*self.phi_LDDE(L,z, Gamma)*self.dFdE(self.E,z,L, Gamma)#*np.exp(-self.Tau(self.E,z))

    # def sampler(self):
    #     while True:
    #         Gamma = random.uniform(self.Gammamin,self.Gammamax)
    #         L = random.uniform(self.Lmin,self.Lmax)
    #         z = random.uniform(self.zmin,self.zmax)
    #         yield (Gamma, L, z)

    # def dIdE_mc(self, E, nmc):
    #     self.E = E
    #     domainsize = (self.Gammamax-self.Gammamin)*(self.Lmax-self.Lmin)*(self.zmax-self.zmin)
    #     random.seed(1)
    #     result, error = mcint.integrate(self.dIdE_integrand, self.sampler(), measure=domainsize, n=nmc)
    #     return result

    def dIdE_mc_vegas(self, E, nitn=15, neval=2e4, verbose=False):
        self.E = E

        integ = vegas.Integrator([[self.Gammamin,self.Gammamax], [self.Lmin,self.Lmax], [self.zmin,self.zmax]])
        result = integ(self.dIdE_integrand, nitn=nitn, neval=neval)
        if verbose:
            print result.summary()
        return result.mean

    # def dIdE_mc_monaco(self, E):
    #     self.E = E
    #     result, error = mcquad(lambda x: self.integrand(x), npoints=1e5, xl=[self.Gammamin, self.Lmin, self.zmin], xu=[self.Gammamax, self.Lmax, self.zmax])
    #     return result

    """
    ************************************************************
    """
    
    def dIdE(self, E):
        """
        Return intensity spectrum of blazars. Since this is only used for sub-bin apportioning of photons, 
        we use a single index approximation (the source count function uses the full form)
        """

        if self.model == 'fsrq':
            Gamma = 2.4
        else:
            Gamma = 2.1 # Assumed spectral index for blazars

        self.dIdEval = integrate.nquad(lambda L,z: self.dVdz(z)*self.phi_LDDE(L,z, Gamma)*self.dFdE(E,z,L, Gamma)*np.exp(-self.Tau(E,z)),[[self.Lmin,self.Lmax], [self.zmin, self.zmax]], opts=[{'epsrel':1e-2,'epsabs':0},{'epsrel':1e-2,'epsabs':0}])[0]

        return self.dIdEval

    def dVdz(self,z):
        """
        Return comoving volument element
        """
        return self.cd.comoving_volume_element(z)*Mpc**3

    def Lgamma(self, Fgamma, Gamma,z):
        """
        Return luminosity flux given energy flux Fgamma
        """
        E1=100*MeV
        E2=100*GeV
        dL = self.cd.luminosity_distance(z)*Mpc
        return 4*np.pi*dL**2*(1+z)**(-2+Gamma)*Fgamma*((E2**(-Gamma+2)-E1**(-Gamma+2))/(E2**(-Gamma+1)-E1**(-Gamma+1)))*((-Gamma+1)/(-Gamma+2))
        # return 4*np.pi*dL**2*(1+z)**(-2+Gamma)*Fgamma*(0.00166667*self.Eb(Gamma)**2.6*(1.*hyp2f1(0.666667, 1., 1.66667, -0.0000316228*self.Eb(Gamma)**0.9) - 0.0158489*hyp2f1(0.666667, 1., 1.66667, -6.30957*10**-8*self.Eb(Gamma)**0.9)))/(6.25*10**-9*self.Eb(Gamma)**2.6*(1.*hyp2f1(1., 1.77778, 2.77778, -0.0000316228*self.Eb(Gamma)**0.9) - 0.0000158489*hyp2f1(1., 1.77778, 2.77778, -6.30957*10**-8*self.Eb(Gamma)**0.9)))

    def dNdF(self,Fgamma):
        """
        Returns the differential source counts function in units of Centimeter**2*Sec
        """    
        dFgamma = Fgamma/1000
        return (1/dFgamma)*integrate.quad(lambda Gamma: integrate.quad(lambda z: integrate.quad(lambda Lgamma_var: 4*np.pi*self.dVdz(z)*self.phi_LDDE(Lgamma_var,z, Gamma), self.Lgamma(Fgamma , Gamma,z), self.Lgamma(Fgamma+dFgamma, Gamma,z))[0],self.zmin,self.zmax)[0],self.Gammamin,self.Gammamax)[0]

    def set_dIdE(self, Evals, dIdEvals):
        """
        Make interpolating function from calculated energy spectrum
        """
        self.dIdE_interp = ip.InterpolatedUnivariateSpline(Evals, dIdEvals)

    def Fpgamma(self,Fgamma,E1,E2):
        """
        Stretch flux for a given range to observed value over 0.1-100 GeV
        """
        return Fgamma*integrate.quad(lambda E: self.dIdE_interp(E), 100*MeV, 100*GeV)[0]/integrate.quad(lambda E: self.dIdE_interp(E), E1, E2)[0]

    # def Fpgamma(self,Fgamma,E1,E2):
    #     """
    #     Stretch flux for a given range to observed value over 0.1-100 GeV
    #     """
    #     Gamma = 2.1
    #     E10 = 100*MeV
    #     E20 = 100*GeV
    #     return Fgamma*((E20**(-Gamma+1)-E10**(-Gamma+1))/(E2**(-Gamma+1)-E1**(-Gamma+1)))

    # def Fpgamma(self,Fgamma,E1,E2, Gamma):
    #     """
    #     Stretch flux for a given range to observed value over 0.1-100 GeV
    #     """
    #     # Gamma = 2.1
    #     E10 = 100*MeV
    #     E20 = 100*GeV
    #     # return Fgamma*((E20**(-Gamma+1)-E10**(-Gamma+1))/(E2**(-Gamma+1)-E1**(-Gamma+1)))
    #     return Fgamma*(6.25*10**-9*self.Eb(Gamma)**2.6*(1.*hyp2f1(1., 1.77778, 2.77778, -0.0000316228*self.Eb(Gamma)**0.9) - 0.0000158489*hyp2f1(1., 1.77778, 2.77778, -6.30957*10**-8*self.Eb(Gamma)**0.9)))/((0.625*E2**1.6*hyp2f1(1., 1.77778, 2.77778, -(1/(E1/self.Eb(Gamma))**0.9)) - 0.625*E1**1.6*hyp2f1(1., 1.77778, 2.77778, -(1/(E2/self.Eb(Gamma))**0.9)))/((E1*E2)**1.6*(1/self.Eb(Gamma))**2.6))
        
    # def dNdFp(self,Fgamma,E1,E2):
    #     """
    #     Returns the differential source counts function in units of Centimeter**2*Sec
    #     """    
        # dFgamma = Fgamma/1000
        # # Fpgamma_L = self.Fpgamma(Fgamma,E1,E2) # Uncomment if using simplified form
        # # Fpgamma_H = self.Fpgamma(Fgamma+dFgamma,E1,E2)
        # return (1/dFgamma)*integrate.quad(lambda Gamma: integrate.quad(lambda z: integrate.quad(lambda Lgamma_var: 4*np.pi*self.dVdz(z)*self.phi_LDDE(Lgamma_var,z, Gamma), self.Lgamma(self.Fpgamma(Fgamma,E1,E2, Gamma) , Gamma,z), self.Lgamma(self.Fpgamma(Fgamma+dFgamma,E1,E2, Gamma), Gamma,z))[0],self.zmin,self.zmax)[0],self.Gammamin,self.Gammamax,epsabs=0,epsrel=10**-2)[0]
        
    def dNdFp(self,Fgamma,E1,E2):
        """
        Returns the differential source counts function in units of Centimeter**2*Sec
        """    
        dFgamma = Fgamma/1000
        Fpgamma_L = self.Fpgamma(Fgamma,E1,E2) # Uncomment if using simplified form
        Fpgamma_H = self.Fpgamma(Fgamma+dFgamma,E1,E2)
        return (1/dFgamma)*integrate.quad(lambda Gamma: integrate.quad(lambda z: integrate.quad(lambda Lgamma_var: 4*np.pi*self.dVdz(z)*self.phi_LDDE(Lgamma_var,z, Gamma), self.Lgamma(Fpgamma_L , Gamma,z), self.Lgamma(Fpgamma_H, Gamma,z))[0],self.zmin,self.zmax)[0],self.Gammamin,self.Gammamax,epsabs=0,epsrel=10**-2)[0]

class LuminosityFunctionSFG:
    """
    Class for calculating the luminosity function and source counts of SFGs.
    Based on 1404.1189, 1302.5209 and 1206.1346.
    """
    def __init__(self, source, pionic_peak = True):
        
        self.source = source # Set the source type
        self.pionic_peak = pionic_peak # Do we want the pionic peak energy spectrum

        # Set a few general parameters
        self.alpha = 1.17 # These are values from 1206.1346 (also given after eq. 2.4 of 1404.1189)
        self.beta = 39.28 
        self.Ls = 3.828*10**33*erg*Sec**-1 # Verify this is the right value for the solar luminosity -- from https://en.wikipedia.org/wiki/Solar_luminosity
        
        self.set_params() # Set the LF parameters

        # Integration ranges
        self.Lmin = 10**34*erg*Sec**-1 
        self.Lmax = 10**44*erg*Sec**-1
        self.zmin = 10**-3.
        self.zmax = 4.

        self.CTB_en_bins = 10**np.linspace(np.log10(0.3), np.log10(300),31) # CTB energy bins

        self.cd = cld.CosmologicalDistance() # For cosmological distance calculations

        self.make_Tau()

    def set_params(self):
        """
        Parameters from Table 8 of 1302.5209
        """    
        if self.source == 'NG':

            self.phi_IR_star_0 = 10**-1.95*Mpc**-3
            self.LIR_star_0 = 10**9.45*self.Ls

            self.alpha_IR = 1.00
            self.sigma_IR = 0.5

            self.kL1 = 4.49
            self.kL2 = 0.00
            self.zbL = 1.1

            self.kRh1 = -0.54
            self.kRh2 = -7.13
            self.zbRh = 0.53

            self.Gamma = 2.7

        elif self.source == 'SB':

            self.phi_IR_star_0 = 10**-4.59*Mpc**-3
            self.LIR_star_0 = 10**11.0*self.Ls

            self.alpha_IR = 1.00
            self.sigma_IR = 0.35

            self.kL1 = 1.96

            self.kRh1 = 3.79
            self.kRh2 = -1.06
            self.zbRh = 1.1

            self.Gamma = 2.2

        elif self.source == 'SF-AGN':

            self.phi_IR_star_0 = 10**-3.00*Mpc**-3
            self.LIR_star_0 = 10**10.6*self.Ls

            self.alpha_IR = 1.20
            self.sigma_IR = 0.4

            self.kL1 = 3.17

            self.kRh1 = 0.67
            self.kRh2 = -3.17
            self.zbRh = 1.1

            self.GammaSB = 2.2
            self.GammaNG = 2.7

        elif self.source == 'ALL':

            self.phi_IR_star_0 = 10**-2.29*Mpc**-3
            self.LIR_star_0 = 10**10.12*self.Ls

            self.alpha_IR = 1.15
            self.sigma_IR = 0.52

            self.kL1 = 3.55
            self.kL2 = 1.62
            self.zbL = 1.85

            self.kRh1 = -0.57
            self.kRh2 = -3.92
            self.zbRh = 1.1

            self.Gamma = 2.475

    def phi_IR_star(self, z):
        """
        Redshift evolution of phi_IR (see section 3.5 of 1302.5209)
        """
        if z < self.zbRh:
            return self.phi_IR_star_0*(1+z)**self.kRh1
        else:
            return (self.phi_IR_star_0*(1+z)**self.kRh2)*(self.phi_IR_star_0*(1+self.zbRh)**self.kRh1)/((self.phi_IR_star_0*(1+self.zbRh)**self.kRh2))

    def LIR_star(self, z):
        """
        Redshift evolution of LIR (see section 3.5 of 1302.5209)
        """

        if (self.source == 'NG') | (self.source == 'ALL'):
            if z < self.zbL:
                return self.LIR_star_0*(1+z)**self.kL1
            else:
                return (self.LIR_star_0*(1+z)**self.kL2)*(self.LIR_star_0*(1+self.zbL)**self.kL1)/(self.LIR_star_0*(1+self.zbL)**self.kL2)
        else:
            return self.LIR_star_0*(1+z)**self.kL1

    def phi_IR(self, LIR, z):
        """
        Returns the IR luminosity for a give SFG sub-class.
        Based on eq. 2.2 of 1404.1189.
        """
        return self.phi_IR_star(z)*(LIR/self.LIR_star(z))**(1-self.alpha_IR)*np.exp(-(1/(2*self.sigma_IR**2))*(np.log10(1+(LIR/self.LIR_star(z))))**2)


    def LIR(self, Lgamma):
        """
        Returns the IR luminosity given the gamma luminosity.
        Based on eq. 2.4 of 1404.1189.
        """
        return ((Lgamma/(erg*Sec**-1))*(1/10**self.beta))**(1/self.alpha)*10**10*self.Ls

    def phi_gamma(self, Lgamma, z):
        """
        Returns the gamma-ray LF given the IR LF.
        Based on eq. 2.5 of 1404.1189.
        This is defined as Phi(Lgamma,z)= dN/dVdLog(Lgamma) according to text after eq. 2.1 so we divide by ln(10)*Lgamma to get standard form dN/dVdLgamma we use as per usual.
        """
        return self.phi_IR(self.LIR(Lgamma),z)*(1/self.alpha)*(1/(np.log(10)*Lgamma))

    def dFdE_unnorm(self, E, Gamma):
        if self.pionic_peak:
            if E < 600*MeV:
                return E**-1.5
            else:
                return E**-Gamma    
        else:
            return E**-Gamma

    def Lgamma(self, Fgamma, Gamma,z):
        """
        Return luminosity flux given energy flux Fgamma
        """
        dL = self.cd.luminosity_distance(z)*Mpc
        E1=100*MeV
        E2=100*GeV
        if self.pionic_peak:
            N = Fgamma/(((600*MeV)**-0.5 - (100*MeV)**-0.5)/(-0.5*(600*MeV)**-1.5)*(1+z)**-1.5 + ((100*GeV)**(-Gamma+1) - (600*MeV)**(-Gamma+1))/((-Gamma+1)*(600*MeV)**-Gamma)*(1+z)**-Gamma) # Normalization for BPL form in Tamborra et al
            L_N = (4*np.pi*dL**2)*(((600*MeV)**0.5 - (100*MeV)**0.5)/(0.5*(600*MeV)**-1.5)*(1+z)**-1.5 + ((100*GeV)**(-Gamma+2) - (600*MeV)**(-Gamma+2))/((-Gamma+2)*(600*MeV)**-Gamma)*(1+z)**-Gamma)
        else:
            N = Fgamma/(E2**-(Gamma+1)-E1**-(Gamma+1)) # Normalization for PL # This is wrong -- check
            L_N = (4*np.pi*dL**2)*(E2**-(Gamma+2)-E1**-(Gamma+2)) # This is wrong -- check

        return N*L_N

    def fSB(self, z):
        """
        Fraction of SF-AGN sources contributing SB and non-SB type 
        spectra, Table 2 of Tamborra et al
        """
        if self.source == 'SF-AGN':
            if 0.0 <= z < 0.3:
                return .15
            elif 0.3 <= z < 0.45:
                return .09
            elif 0.45 <= z < 0.6:
                return .01
            elif 0.6 <= z < 0.8:
                return .13
            elif 0.8 <= z < 1.0:
                return .27
            elif 1.0 <= z < 1.2:
                return .68
            elif 1.2 <= z < 1.7:
                return .25
            elif 1.7 <= z < 2.0:
                return .25
            elif 2.0 <= z < 2.5:
                return .81
            elif 2.5 <= z < 3.0:
                return .76
            elif 3.0 <= z < 4.2:
                return .72

    def dNdF(self, Fgamma):
        """
        Returns the differential source counts function in units of Centimeter**2*Sec
        """    
        dFgamma = Fgamma/1000

        if self.source == 'SF-AGN':
            return (1/dFgamma)*(integrate.quad(lambda z: integrate.quad(lambda Lgamma: self.fSB(z)*4*np.pi*self.dVdz(z)*self.phi_gamma(Lgamma,z), self.Lgamma(Fgamma,self.GammaSB,z), self.Lgamma(Fgamma+dFgamma,self.GammaSB,z))[0], self.zmin,self.zmax)[0]+integrate.quad(lambda z: integrate.quad(lambda Lgamma: (1-self.fSB(z))*4*np.pi*self.dVdz(z)*self.phi_gamma(Lgamma,z), self.Lgamma(Fgamma,self.GammaNG,z), self.Lgamma(Fgamma+dFgamma,self.GammaNG,z))[0], self.zmin,self.zmax)[0])
        else:
            return (1/dFgamma)*integrate.quad(lambda z: integrate.quad(lambda Lgamma: 4*np.pi*self.dVdz(z)*self.phi_gamma(Lgamma,z), self.Lgamma(Fgamma,self.Gamma,z), self.Lgamma(Fgamma+dFgamma,self.Gamma,z))[0], self.zmin,self.zmax)[0]

    def Fpgamma(self,Fgamma,E1,E2):
        """
        Stretch flux for a given range to observed value over 0.1-100 GeV
        """
        return Fgamma*integrate.quad(lambda E: self.dIdE_interp(E), 100*MeV, 100*GeV)[0]/integrate.quad(lambda E: self.dIdE_interp(E), E1, E2)[0]

    def dNdFp(self,Fgamma,E1,E2):
        """
        Returns the scaled differential source counts function in units of Centimeter**2*Sec
        """    
        dFgamma = Fgamma/1000
        Fpgamma_L = self.Fpgamma(Fgamma,E1,E2)
        Fpgamma_H = self.Fpgamma(Fgamma+dFgamma,E1,E2)

        if self.source == 'SF-AGN':
            return (1/dFgamma)*(integrate.quad(lambda z: integrate.quad(lambda Lgamma: self.fSB(z)*4*np.pi*self.dVdz(z)*self.phi_gamma(Lgamma,z), self.Lgamma(Fpgamma_L,self.GammaSB,z), self.Lgamma(Fpgamma_H,self.GammaSB,z),epsabs=0,epsrel=10**-2)[0], self.zmin,self.zmax,epsabs=0,epsrel=10**-2)[0]+integrate.quad(lambda z: integrate.quad(lambda Lgamma: (1-self.fSB(z))*4*np.pi*self.dVdz(z)*self.phi_gamma(Lgamma,z), self.Lgamma(Fpgamma_L,self.GammaNG,z), self.Lgamma(Fpgamma_H,self.GammaNG,z),epsabs=0,epsrel=10**-2)[0], self.zmin,self.zmax,epsabs=0,epsrel=10**-2)[0])
        else:
            return (1/dFgamma)*integrate.quad(lambda z: integrate.quad(lambda Lgamma: 4*np.pi*self.dVdz(z)*self.phi_gamma(Lgamma,z), self.Lgamma(Fpgamma_L,self.Gamma,z), self.Lgamma(Fpgamma_H,self.Gamma,z),epsabs=0,epsrel=10**-2)[0], self.zmin,self.zmax,epsabs=0,epsrel=10**-2)[0]
    
    def opts0(self,*args, **kwargs):
            return {'epsrel':1e-2,'epsabs':0}

    def dFdE(self, E,z, L, Gamma):
        """
        Intrinsic flux of source. Use simple power law.
        """
        E1 = 100*MeV
        E2 = 100*GeV
        dL = self.cd.luminosity_distance(z)*Mpc
        if self.pionic_peak:
            Gamma_L = 1.5
            N = L/(4*np.pi*dL**2)/((((600*MeV)**(-Gamma_L + 2) - (100*MeV)**(-Gamma_L + 2))/((-Gamma_L+2)*(600*MeV)**-Gamma_L))*(1+z)**-Gamma_L+(((100*GeV)**(-Gamma + 2)-(600*MeV)**(-Gamma + 2))/((-Gamma+2)*(600*MeV)**-Gamma))*(1+z)**-Gamma)
            if E < 600*MeV:
                return N*(E**-Gamma_L)/(600*MeV)**-Gamma_L
            else:
                return N*(E**-Gamma)/(600*MeV)**-Gamma

        else:
            N = L/(4*np.pi*dL**2)*(-Gamma+2)/(E2**(-Gamma+2)-E1**(-Gamma+2)) # This is wrong I think -- change.
            return N*E**-Gamma

    def dIdE(self, E):
        if self.source == 'SF-AGN':
            self.dIdEval = integrate.quad(lambda z: self.fSB(z)*self.dVdz(z)*integrate.quad(lambda L: self.phi_gamma(L,z)*self.dFdE((1+z)*E,z,L, self.GammaSB)*np.exp(-self.Tau(E,z)), self.Lmin,self.Lmax,epsabs=0,epsrel=10**-2)[0], self.zmin, self.zmax,epsabs=0,epsrel=10**-2)[0]+integrate.quad(lambda z: (1-self.fSB(z))*self.dVdz(z)*integrate.quad(lambda L: self.phi_gamma(L,z)*self.dFdE((1+z)*E,z,L, self.GammaNG)*np.exp(-self.Tau(E,z)), self.Lmin,self.Lmax,epsabs=0,epsrel=10**-2)[0], self.zmin, self.zmax,epsabs=0,epsrel=10**-2)[0]
        else:
            self.dIdEval = integrate.quad(lambda z: self.dVdz(z)*integrate.quad(lambda L: self.phi_gamma(L,z)*self.dFdE((1+z)*E,z,L, self.Gamma)*np.exp(-self.Tau(E,z)), self.Lmin,self.Lmax,epsabs=0,epsrel=10**-2)[0], self.zmin, self.zmax,epsabs=0,epsrel=10**-2)[0]

        return self.dIdEval

    def set_dIdE(self, Evals, dIdEvals):
        """
        Make interpolating function from calculated energy spectrum
        """
        self.dIdE_interp = ip.InterpolatedUnivariateSpline(Evals, dIdEvals)

    def dVdz(self,z):
        """
        Return comoving volument element
        """
        return self.cd.comoving_volume_element(z)*Mpc**3

    def make_Tau(self):
        """
        Create EBL interpolation (model based on Finke et al)
        """

        # Load and combined EBL files downloaded from http://www.phy.ohiou.edu/~finke/EBL/ 
        tau_files = '/group/hepheno/smsharma/Fermi_High_Latitudes/Fermi-HighLat/sim/tau_modelC_total/tau_modelC_total_z%.2f.dat'
        z_list = np.arange(0, 5, 0.01) # z values to load files for
        E_list, tau_list = [],[]
        for z in z_list:
            d = np.genfromtxt(tau_files % z, unpack=True)
            E_list = d[0]
            tau_list.append(d[1])
        self.Tau_ip = ip.RectBivariateSpline(z_list, E_list, np.array(tau_list)) # Create interpolation
    
    def Tau(self,E,z):
        """
        EBL attenuation of gamma rays using Finke et al
        """
        return np.float64(self.Tau_ip(z, float(E)/(1000*GeV)))
        # return (z/3.3)*(E/(10*GeV))**.8 # Analytic approximation from 1506.05118

class LuminosityFunctionmAGN:
    """
    Class for calculating the luminosity function and source counts of mAGNs.
    Based on 1304.0908 (Di Mauro et al) and astro-ph/0010419 (Willot et al).
    """
    def __init__(self):
                
        self.set_params() # Set the radio LF parameters

        # Integration ranges
        self.Lmin = 10**41*erg*Sec**-1 # These are from the text after eq. (23)
        self.Lmax = 10**48*erg*Sec**-1
        self.zmin = 10**-3. # These are from the text after eq. (22) (using 10**-3 rather than 0)
        self.zmax = 4.
        self.Gammamin = 1.0
        self.Gammamax = 3.5

        self.Gamma_mean = 2.37 # Spectral index characteristics -- from text before equation (1) of Di Mauro et al
        self.Gamma_sigma = 0.32

        self.CTB_en_bins = 10**np.linspace(np.log10(0.3), np.log10(300),31) # CTB energy bins

        self.cd = cld.CosmologicalDistance() # For cosmological distance calculations -- default flat Lambda CDM parameters
        self.cdW = cld.CosmologicalDistance(omega_m = 0., omega_l = 0,h0=.5) # Cosmology used in Willot et al

        self.make_Tau()

    def set_params(self):
        """
        Parameters from the radio LF from Table 1 of Willot et al. Model C, omega_m = 0.
        """    
        self.rho_l0 = 10**-7.523*Mpc**-3
        self.alphal = 0.586
        self.Llstar = 10**26.48*W/Hz/sr
        self.zl0 = 0.710
        self.kl = 3.48
        self.rho_h0 = 10**-6.757*Mpc**-3
        self.alphah = 2.42
        self.Lhstar = 10**27.39*W/Hz/sr

        self.zh0 = 2.03
        self.zh1 = 0.568
        self.zh2 = 0.956

    def phi_R(self, LR151f, z):
        """
        Returns the Willot et al radio luminosity function (RLF), provided the radio luminosity at 151 MHz in 
        units of W/Hz/sr (LR151f) and redshift z. This is based on equations (7)-(14) of Willot et al.
        """

        if z < self.zl0:
            phi_l = self.rho_l0*(LR151f/self.Llstar)**-self.alphal*np.exp(-LR151f/self.Llstar)*(1+z)**self.kl
        if z >= self.zl0:
            phi_l = self.rho_l0*(LR151f/self.Llstar)**-self.alphal*np.exp(-LR151f/self.Llstar)*(1+self.zl0)**self.kl

        if z < self.zh0:
            phi_h = self.rho_h0*(LR151f/self.Lhstar)**-self.alphah*np.exp(-self.Lhstar/LR151f)*np.exp(-.5*((z-self.zh0)/self.zh1)**2)
        if z >= self.zh0:
            phi_h = self.rho_h0*(LR151f/self.Lhstar)**-self.alphah*np.exp(-self.Lhstar/LR151f)*np.exp(-.5*((z-self.zh0)/self.zh2)**2)

        eta = self.cdW.comoving_volume_element(z)/self.cd.comoving_volume_element(z) # ratio of differential volume slices to convert between cosmologies (see equation 14)

        return eta*(phi_l + phi_h)

    def phi_gamma(self, Lgamma, z):
        """
        Returns the gamma-ray LF given gamma-ray luminosity. Based on equation (20) of 1304.0908. First pre-factor 
        is from equation (5), second from equation (13). Post-factor converts to dN/dLdV form.

        Factor of 0.496 in the argument converts total radio luminosity from 5 GHz to 151 MHz. Divide by (151*MHz*sr) to get in appropriate 
        units [W/Hz/sr] for phi_R above.

        """
        return (1/1.008)*(1/0.77)*self.phi_R(((151*MHz)/(5*GHz))**-.8*self.LR5tot(self.LR5core(Lgamma))/(5*GHz*sr), z)*(1/(np.log(10)*Lgamma))

    def LR5tot(self, LR5core):
        """
        Returns total radio Luminosity at 5 GHz given core Luminosity at 5 GHz. Equation (13) of 
        Di Mauro et al and L ~ f**(-alpha+1) to convert 1.4 -> 5 GHz total luminosity (factor of 1.2899).
        """

        return 1.2899*(10**((np.log10(LR5core/(5*GHz)/(W/Hz))-4.2)/0.77))*(W/Hz)*(1.4*GHz)

    def LR5core(self, Lgamma):
        """
        Returns gamma-ray Luminosity given 5 GHz core radio Luminosity. Equation (8) of 1304.0908
        """

        return 10**((np.log10(Lgamma/(erg*Sec**-1))-2.00)/1.008)*(erg*Sec**-1)

    def Lgamma(self, Fgamma, Gamma,z):
        """
        Return luminosity flux given energy flux Fgamma
        """
        dL = self.cd.luminosity_distance(z)*Mpc

        E2 = 100*GeV
        E1 = 100*MeV
        
        N = Fgamma/((E2**(-Gamma+1)-E1**(-Gamma+1))/(-Gamma+1)) # Normalization for PL
        L_N = (4*np.pi*dL**2)*(E2**(-Gamma+2)-E1**(-Gamma+2))*(1+z)**(-2+Gamma)/(-Gamma+2) # Check it double deck it

        return N*L_N

    def dNdF(self, Fgamma):
        """
        Returns the differential source counts function
        """    
        dFgamma = Fgamma/1000

        return (1/dFgamma)*integrate.quad(lambda Gamma: integrate.quad(lambda z: integrate.quad(lambda Lgamma: 4*np.pi*self.dVdz(z)*self.phi_gamma(Lgamma,z)*(1/(np.sqrt(2*np.pi)*self.Gamma_sigma))*np.exp(-(Gamma - self.Gamma_mean)**2/(2*self.Gamma_sigma**2)), self.Lgamma(Fgamma,Gamma,z), self.Lgamma(Fgamma+dFgamma,Gamma,z))[0], self.zmin,self.zmax)[0], self.Gammamin, self.Gammamax)[0]

    def NF(self, Fgamma):
        """
        Returns the cumulative source counts function
        """    

        return integrate.quad(lambda Gamma: integrate.quad(lambda z: integrate.quad(lambda Lgamma: 4*np.pi*self.dVdz(z)*self.phi_gamma(Lgamma,z)*np.exp(-(Gamma - self.Gamma_mean)**2/(2*self.Gamma_sigma**2))*(1/(np.sqrt(2*np.pi)*self.Gamma_sigma)), self.Lgamma(Fgamma,Gamma,z), self.Lmax)[0], self.zmin,self.zmax)[0], self.Gammamin, self.Gammamax)[0]

    def dVdz(self,z):
        """
        Return comoving volument element
        """
        return self.cd.comoving_volume_element(z)*Mpc**3

    def make_Tau(self):
        tau_ary = np.load("tau_ary.npy")
        self.Tau_ip = ip.SmoothBivariateSpline(tau_ary[0],tau_ary[2],tau_ary[1])

    def Tau(self,E,z):
        """
        EBL attenuation of gamma rays (mention references)
        """
        return np.float64(self.Tau_ip(E/(1000*GeV),z))
        # return (z/3.3)*(E/(10*GeV))**.8

    def dFdE(self, E,z, L, Gamma):
        """
        Intrinsic flux of source. Use simple power law.
        """
        E1 = 100*MeV
        E2 = 100*GeV
        dL = self.cd.luminosity_distance(z)*Mpc

        N = (1+z)**(2-Gamma)*L/(4*np.pi*dL**2)*(2-Gamma)/((1/E1)**-Gamma*(E2**(-Gamma+2)-E1**(-Gamma+2))) # Check it double deck it
        return N*((E/E1)**-Gamma)

    def opts0(self,*args, **kwargs):
        """
        Integration parameters
        """
        return {'epsrel':1e-2,'epsabs':0}
    
    def dIdE(self, E):
        """
        Return intensity spectrum of blazars. Since this is only used for sub-bin apportioning of photons, 
        we use a single index approximation (the source count function uses the full form)
        """

        Gamma = 2.37 # Assumed spectral index for mAGN

        self.dIdEval = integrate.nquad(lambda L,z: self.dVdz(z)*self.phi_gamma(L,z)*self.dFdE(E,z,L, Gamma)*(1/(np.sqrt(2*np.pi)*self.Gamma_sigma))*np.exp(-(Gamma - self.Gamma_mean)**2/(2*self.Gamma_sigma**2))*np.exp(-self.Tau(E,z)),[[self.Lmin,self.Lmax], [self.zmin, self.zmax]], opts=[self.opts0,self.opts0,self.opts0])[0]

        return self.dIdEval

    """
    ************************************************************
    Monte carlo integration experimentation
    """

    def dIdE_integrand(self, x):
        Gamma = x[0]
        L = x[1]
        z = x[2]
        return self.dVdz(z)*self.phi_gamma(L,z)*self.dFdE(self.E,z,L, Gamma)*(1/(np.sqrt(2*np.pi)*self.Gamma_sigma))*np.exp(-(Gamma - self.Gamma_mean)**2/(2*self.Gamma_sigma**2))*np.exp(-self.Tau(self.E,z))

    def dIdE_mc_vegas(self, E,nitn=10,neval=1e4):
        self.E = E

        integ = vegas.Integrator([[self.Gammamin,self.Gammamax], [self.Lmin,self.Lmax], [self.zmin,self.zmax]])
        result = integ(self.dIdE_integrand, nitn=nitn, neval=neval)
        print result.summary()
        return result.mean

    """
    ************************************************************
    """

    def Fpgamma(self,Fgamma,E1,E2, Gamma):
        """
        Stretch flux for a given range to observed value over 0.1-100 GeV
        """
        
        E10 = 100*MeV
        E20 = 100*GeV
        return Fgamma*((E20**(-Gamma+1)-E10**(-Gamma+1))/(E2**(-Gamma+1)-E1**(-Gamma+1)))

    def dNdFp(self, Fgamma, E1, E2):
        """
        Returns the differential source counts function
        """    
        dFgamma = Fgamma/1000

        return (1/dFgamma)*integrate.quad(lambda Gamma: integrate.quad(lambda z: integrate.quad(lambda Lgamma: 4*np.pi*self.dVdz(z)*self.phi_gamma(Lgamma,z)*(1/(np.sqrt(2*np.pi)*self.Gamma_sigma))*np.exp(-(Gamma - self.Gamma_mean)**2/(2*self.Gamma_sigma**2)), self.Lgamma(self.Fpgamma(Fgamma,E1,E2, Gamma),Gamma,z), self.Lgamma(self.Fpgamma(Fgamma+dFgamma,E1,E2, Gamma),Gamma,z))[0], self.zmin,self.zmax)[0], self.Gammamin, self.Gammamax)[0]

    def set_dIdE(self, Evals, dIdEvals):
        """
        Make interpolating function from calculated energy spectrum
        """
        self.dIdE_interp = ip.InterpolatedUnivariateSpline(Evals, dIdEvals)

