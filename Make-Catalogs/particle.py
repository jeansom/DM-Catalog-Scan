"""
DM annihilation spectra and associated functions
"""
import os, sys
from scipy.special import erf
from scipy.interpolate import interp1d 
import numpy as np
import pandas as pd
import hmf
from units import *
from global_variables import *
from scipy import integrate

class Particle:
    def __init__(self, channel='b',CTB_en_min=8,CTB_en_max=16,m_chi=100*GeV, sigma_v = sigma_v):

        self.work_dir = work_dir

        self.work_dir = '/group/hepheno/smsharma/Fermi-LSS/'
        
        self.dNdLogx_df=pd.read_csv(self.work_dir+'AdditionalData/AtProduction_gammas.dat', delim_whitespace=True)
       
        self.m_chi = m_chi
        self.sigma_v = sigma_v
        self.CTB_en_max = CTB_en_max
        self.CTB_en_min = CTB_en_min
        self.channel = channel
      
        # self.CTB_en_bins = 10**np.linspace(np.log10(0.3), np.log10(300),31)
        self.CTB_en_bins = 10**np.linspace(np.log10(0.2), np.log10(2000),41)
        
        self.dNdE(channel, m_chi)
            
        self.Phis()

    def dNdE(self, channel='b', mDM = 100*GeV):
        """
        Make interpolated annihilation spectra for given mass and channel
        """
        dNdLogx_ann_df = self.dNdLogx_df.query('mDM == ' + (str(np.int(float(mDM)/GeV))))[['Log[10,x]',channel]]
        self.Egamma = np.array(mDM*(10**dNdLogx_ann_df['Log[10,x]']))
        self.dNdEgamma = np.array(dNdLogx_ann_df[channel]/(self.Egamma*np.log(10)))        
        self.dNdE_interp =  interp1d(self.Egamma, self.dNdEgamma)

    def Phi(self, mDM, sigma_v, Emin, Emax):
        """
        Integrated flux for a given energy range from dNdE
        """
        N = integrate.quad(lambda x: self.dNdE_interp(x), Emin, Emax)[0]
        return sigma_v/(8*np.pi*mDM**2)*np.array(N)#/(Emax-Emin)

    def Phis(self):
        """
        Integrated fluxes for required CTB bins from Phi
        """
        self.PhiE = []
        # print 'The cross section assumed is', self.sigma_v/(Centimeter**3*Sec**-1)
        for i in range(self.CTB_en_min, self.CTB_en_max+1):
            if (self.CTB_en_bins[i+1]*GeV >self.m_chi) or (self.CTB_en_bins[i]*GeV >self.m_chi): # No counts
                self.PhiE.append(0)
            else:
                self.PhiE.append(self.Phi(self.m_chi, self.sigma_v, self.CTB_en_bins[i]*GeV,self.CTB_en_bins[i+1]*GeV))