###############################################################################
# king_params.py
###############################################################################
# HISTORY:
#   2017-03-22 - Written - Nick Rodd (MIT)
###############################################################################

# Load the appropriate parameters for a King function (fcore, score, gcore,
# stail, gtail and the energy rescale factor SpE)
# For details see:
# http://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_LAT_IRFs/IRF_PSF.html

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d as interp1d

class PSF_king:
    """ A class to return various King function parameters """
    def __init__(self,maps_dir,eventclass,quartile):

        # First determine the appropriate parameters for this class and quartile
        if eventclass==2:
            psf_file_name = 'psf_P8R2_SOURCE_V6_PSF.fits'
        elif eventclass==5:
            psf_file_name = 'psf_P8R2_ULTRACLEANVETO_V6_PSF.fits'
        fits_file_name = maps_dir + 'psf_data/' + psf_file_name
        self.f = fits.open(fits_file_name)

        if quartile==1:
            self.params_index=10
            self.rescale_index=11
            if eventclass==2:
                theta_norm=[0.0000000,9.7381019e-06,0.0024811595,0.022328802,
                            0.080147663,0.17148392,0.30634315,0.41720551]
            if eventclass==5:
                theta_norm=[0.0000000,9.5028121e-07,0.00094418357,0.015514370,
                            0.069725775,0.16437751,0.30868705,0.44075016]
        if quartile==2:
            self.params_index=7
            self.rescale_index=8
            if eventclass==2:
                theta_norm=[0.0000000,0.00013001938,0.010239333,0.048691643,
                            0.10790632,0.18585539,0.29140913,0.35576811]
            if eventclass==5:
                theta_norm=[0.0000000,1.6070284e-05,0.0048551576,0.035358049,
                            0.091767466,0.17568974,0.29916159,0.39315185]
        if quartile==3:
            self.params_index=4
            self.rescale_index=5
            if eventclass==2:
                theta_norm=[0.0000000,0.00074299273,0.018672204,0.062317201,
                            0.12894928,0.20150553,0.28339386,0.30441893]
            if eventclass==5:
                theta_norm=[0.0000000,0.00015569366,0.010164870,0.048955837,
                            0.11750811,0.19840060,0.29488095,0.32993394]
        if quartile==4:
            self.params_index=1
            self.rescale_index=2
            if eventclass==2:
                theta_norm=[4.8923139e-07,0.011167475,0.092594658,0.15382001,
                            0.16862869,0.17309118,0.19837774,0.20231968]
            if eventclass==5:
                theta_norm=[0.0000000,0.0036816313,0.062240006,0.14027030,
                            0.17077023,0.18329804,0.21722594,0.22251374]
    
        self.fill_rescale() 
        self.fill_PSF_params()
    	self.theta_norm = np.transpose([theta_norm for i in range(23)])
        self.interp_bool = False
        self.interpolate_R()
    
    def fill_rescale(self):
        self.rescale_array = self.f[self.rescale_index].data[0][0]
        
    def fill_PSF_params(self):
        self.E_min = self.f[self.params_index].data[0][0] #size is 23
        self.E_max = self.f[self.params_index].data[0][1]
        self.theta_min = self.f[self.params_index].data[0][2] #size is 8
        self.theta_max = self.f[self.params_index].data[0][3]
        self.NCORE = np.array(self.f[self.params_index].data[0][4]) #shape is (8, 23)
        self.NTAIL = np.array(self.f[self.params_index].data[0][5])
        self.SCORE = np.array(self.f[self.params_index].data[0][6])
        self.STAIL = np.array(self.f[self.params_index].data[0][7])
        self.GCORE = np.array(self.f[self.params_index].data[0][8])
        self.GTAIL = np.array(self.f[self.params_index].data[0][9])
        # Now create fcore from the definition
        self.FCORE = np.array([[1/(1+self.NTAIL[i,j]*self.STAIL[i,j]**2/self.SCORE[i,j]**2) for j in range(np.shape(self.NCORE)[1])] for i in range(np.shape(self.NCORE)[0])])
         
    def rescale_factor(self,E): #E in GeV
        SpE = np.sqrt((self.rescale_array[0]*(E*10**3/100)**(self.rescale_array[2]))**2 + self.rescale_array[1]**2)
        return SpE
    
    def interpolate_R(self):
        self.FCORE_int = interp1d((self.E_max+self.E_min)/2.*10**-3, np.sum(self.theta_norm*self.FCORE,axis=0))
        self.SCORE_int = interp1d((self.E_max+self.E_min)/2.*10**-3, np.sum(self.theta_norm*self.SCORE,axis=0))
        self.STAIL_int = interp1d((self.E_max+self.E_min)/2.*10**-3, np.sum(self.theta_norm*self.STAIL,axis=0))
        self.GCORE_int = interp1d((self.E_max+self.E_min)/2.*10**-3, np.sum(self.theta_norm*self.GCORE,axis=0))
        self.GTAIL_int = interp1d((self.E_max+self.E_min)/2.*10**-3, np.sum(self.theta_norm*self.GTAIL,axis=0))
        self.interp_bool = True

    def return_king_params(self,energies,param): # Put E in in GeV
        if not self.interp_bool:
            self.interpolate_R()
        if param=='fcore':
            return self.FCORE_int(energies)
        elif param=='score':
            return self.SCORE_int(energies)
        elif param=='gcore':
            return self.GCORE_int(energies)
        elif param=='stail':
            return self.STAIL_int(energies)
        elif param=='gtail':
            return self.GTAIL_int(energies)
        else:
            print "Param must be fcore, score, gcore, stail or gtail"
