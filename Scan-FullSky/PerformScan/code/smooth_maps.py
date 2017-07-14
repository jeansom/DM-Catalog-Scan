# Code or smoothing maps

import numpy as np
from global_variables import *
from astropy.io import fits
import config.smooth_with_psf as swp
import fermi.PSF_class as PSFC

class smooth_map:
    def __init__(self,maps_dir,CTB_en_min=8,CTB_en_max=16,fits_file_path='False',data_name='p8',nside=128,is_p8=True,eventclass=5, eventtype=0):
        self.nside = nside
        self.maps_dir = maps_dir
        self.psf_data_dir = maps_dir + 'psf_data/'
        self.data_name = data_name
        self.fits_file_path = fits_file_path
        self.CTB_en_min = CTB_en_min
        self.CTB_en_max = CTB_en_max
        self.nbins = self.CTB_en_max - self.CTB_en_min
        self.eventclass = eventclass
        self.eventtype = eventtype

        self.CTB_en_bins = 10**np.linspace(np.log10(0.2), np.log10(2000),41)[self.CTB_en_min:self.CTB_en_max+1] #force energy bins over whole energy range

        self.load_psf(data_name = self.data_name,fits_file_path = self.fits_file_path)

    def smooth_map(self,the_map):
        return np.array([ self.smooth_map_1ebin(the_map,self.sigma_PSF_deg[i]) for i in range(self.nbins)])

    def smooth_map_array(self,the_map):
        return np.array([ self.smooth_map_1ebin(the_map[i],self.sigma_PSF_deg[i]) for i in range(self.nbins)])

    def smooth_map_1ebin(self,the_map,sigma_PSF_deg):
        self.swp_inst = swp.smooth_gaussian_psf_quick(sigma_PSF_deg,the_map)
        #swp_inst.smooth_the_map()
        return self.swp_inst.the_smooth_map

    def load_psf(self,data_name='p8',fits_file_path = 'False'):
        if fits_file_path !='False':
            self.fits_file_name = fits_file_path
        elif data_name=='p8':
            # Define the param and rescale indices for the various quartiles
            params_index_psf1=10
            rescale_index_psf1=11
            params_index_psf2=7
            rescale_index_psf2=8
            params_index_psf3=4
            rescale_index_psf3=5
            params_index_psf4=1
            rescale_index_psf4=2
            # Setup to load the correct PSF details depending on the dataset, and define appropriate theta_norm values, psf1 is bestpsf, psf4 is worst psf (but just a quartile each, so for Q1-3 need tocombine
            if self.eventclass==2:
                psf_file_name = 'psf_P8R2_SOURCE_V6_PSF.fits'
                theta_norm_psf1=[0.0000000,9.7381019e-06,0.0024811595,0.022328802,0.080147663,0.17148392,0.30634315,0.41720551]
                theta_norm_psf2=[0.0000000,0.00013001938,0.010239333,0.048691643,0.10790632,0.18585539,0.29140913,0.35576811]
                theta_norm_psf3=[0.0000000,0.00074299273,0.018672204,0.062317201,0.12894928,0.20150553,0.28339386,0.30441893]
                theta_norm_psf4=[4.8923139e-07,0.011167475,0.092594658,0.15382001,0.16862869,0.17309118,0.19837774,0.20231968]
            elif self.eventclass==5:
                psf_file_name = 'psf_P8R2_ULTRACLEANVETO_V6_PSF.fits'
                theta_norm_psf1=[0.0000000,9.5028121e-07,0.00094418357,0.015514370,0.069725775,0.16437751,0.30868705,0.44075016]
                theta_norm_psf2=[0.0000000,1.6070284e-05,0.0048551576,0.035358049,0.091767466,0.17568974,0.29916159,0.39315185]
                theta_norm_psf3=[0.0000000,0.00015569366,0.010164870,0.048955837,0.11750811,0.19840060,0.29488095,0.32993394]
                theta_norm_psf4=[0.0000000,0.0036816313,0.062240006,0.14027030,0.17077023,0.18329804,0.21722594,0.22251374]
            self.fits_file_name = self.psf_data_dir + psf_file_name
        if fits_file_path !='False' or data_name=='p8':
            self.f = fits.open(self.fits_file_name)
            # Now need to get the correct PSF for the appropriate quartile.
            # If anything other than best psf, need to combine quartiles.
            # Quartiles aren't exactly equal in size, but approximate as so.
            self.PSFC_inst = PSFC.PSF(self.f, theta_norm=theta_norm_psf1, rescale_index=rescale_index_psf1, params_index=params_index_psf1)
            calc_sigma_PSF_deg = np.array(self.PSFC_inst.return_sigma_gaussian(self.CTB_en_bins))
            if ((self.eventtype==4) or (self.eventtype==5) or (self.eventtype==0)):
                self.PSFC_inst = PSFC.PSF(self.f, theta_norm=theta_norm_psf2, rescale_index=rescale_index_psf2, params_index=params_index_psf2)
                calc_sigma_load = np.array(self.PSFC_inst.return_sigma_gaussian(self.CTB_en_bins))
                calc_sigma_PSF_deg = (calc_sigma_PSF_deg + calc_sigma_load)/2.
                if ((self.eventtype==5) or (self.eventtype==0)):
                    self.PSFC_inst = PSFC.PSF(self.f, theta_norm=theta_norm_psf3, rescale_index=rescale_index_psf3, params_index=params_index_psf3)
                    calc_sigma_load = np.array(self.PSFC_inst.return_sigma_gaussian(self.CTB_en_bins))
                    calc_sigma_PSF_deg = (2.*calc_sigma_PSF_deg + calc_sigma_load)/3.
                    if self.eventtype==0:
                        self.PSFC_inst = PSFC.PSF(self.f, theta_norm=theta_norm_psf4, rescale_index=rescale_index_psf4, params_index=params_index_psf4)
                        calc_sigma_load = np.array(self.PSFC_inst.return_sigma_gaussian(self.CTB_en_bins))
                        calc_sigma_PSF_deg = (3.*calc_sigma_PSF_deg + calc_sigma_load)/4.
            # Now take mean of array and extract first element, which will be the PSF
            self.sigma_PSF_deg = calc_sigma_PSF_deg
        else:
            pass
