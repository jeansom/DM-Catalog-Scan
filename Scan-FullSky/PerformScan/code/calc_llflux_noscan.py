#
# NAME:
#  calc_llflux.py
#
# PURPOSE:
#  For a given map create a LL array for each energy bin and flux
#
# HISTORY:
#  Written by Nick Rodd, MIT, 18 July 2016

import numpy as np
import healpy as hp
import copy
import fermi.fermi_plugin as fp  
import config.bayesian_scan_models as bsm
import analysis_classes.fermi_analysis as fa
import sys, os
sys.path.insert(0, './../')
from global_variables import *

class calc_llflux_noscan:
    def __init__(self,J_map_arr,tag,band_mask,external_data=False,calc_flux_array=False,flux_array_ebin=0,bin_min=-5,bin_max=1,nbins=30):

        self.J_map_arr = J_map_arr
        self.tag = tag
        self.band_mask = band_mask
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.nbins = nbins
        self.flux_array_ebin = flux_array_ebin

        # Setup basic variables
        self.nside = hp.npix2nside(len(self.J_map_arr[0]))
        if external_data is not False:
            self.use_external_data = True
        else:
            self.use_external_data = False
        self.external_data = external_data

        # Setup Fermi plugin, and calculate and apply base normalisation
        self.setup_fermi()
        # Use a precalculated norm file each time
        self.norm_file=work_dir+'MakeMC/P8UCVA_base_norm'

        # Apply the norm file - this requires base spec to have been created at some point
        self.f1.use_template_normalization_file(self.norm_file+'.npy',key_suffix='-0')

        # If asked calculate the flux array in a specific enery bin
        if calc_flux_array:
            self.f1.reduce_to_energy_subset(flux_array_ebin,flux_array_ebin+1)
            self.d = work_dir+'ScanOutput/'+self.tag
            self.make_dir()
            self.flux_array_ebin = flux_array_ebin
            self.calc_flux_array()

    def make_dir(self):
        """ Function to make a directory, setup so it won't crash if directory exists """
        if not os.path.exists(self.d):
            try:
                os.mkdir(self.d)
            except OSError, e:
                if e.errno != 17:
                    raise
                pass

    def setup_fermi(self):
        """ Setup the Fermi plugin """
        eventclass=5 # 2 (Source) or 5 (UltracleanVeto)
        eventtype=0 # 0 (all), 3 (bestpsf) or 5 (top3 quartiles)
        mask_type='top300'
        force_mask_at_bin_number=8

        self.f1 = fp.fermi_plugin(maps_dir,fermi_data_dir=fermi_data_dir,work_dir=work_dir,CTB_en_min=0,CTB_en_max=40,nside=self.nside,eventclass=eventclass,eventtype=eventtype,newstyle=1,data_July16=True)

        if mask_type != 'False':
            self.f1.make_ps_mask(mask_type = mask_type,force_energy = True,energy_bin = force_mask_at_bin_number)
        self.f1.add_diffuse_newstyle(comp = 'p7', eventclass = eventclass, eventtype = eventtype)
        self.f1.add_bubbles(comp='bubs') #bubbles
        self.f1.add_iso(comp='iso')  #iso
        self.f1.add_ps_model(comp='ps_model')

        # Exposure correct J_map_arr
        self.J_map_arr *= self.f1.CTB_exposure_maps

        # Add J-factor map with mean 1 in each energy bin
        self.f1.add_template_by_hand('J_map',np.array([self.J_map_arr[i]/np.mean(self.J_map_arr[i]) for i in range(40)]))

    def setup_b_instance(self,norm,add_ps_mask=True):
        """ Setup an instance of bayesian scan with a fixed J_map """
        inst_tag = self.tag + '_'+str(self.flux_array_ebin)
        b = bsm.bayesian_scan_NPTF(tag=inst_tag,nside=self.nside,work_dir=work_dir,psf_dir=psf_dir,nlive=700)
        # Input the data, using the external data if provided
        if self.use_external_data:
            b.load_external_data(self.f1.CTB_en_bins,[self.external_data[self.flux_array_ebin]],self.f1.CTB_exposure_maps)
        else:
            b.load_external_data(self.f1.CTB_en_bins,self.f1.CTB_count_maps,self.f1.CTB_exposure_maps)

        if add_ps_mask:
            b.make_mask_total(band_mask_range = [-self.band_mask,self.band_mask],mask_ring = False,ps_mask_array = self.f1.ps_mask_array)
        else:
            b.make_mask_total(band_mask_range = [-self.band_mask,self.band_mask],mask_ring = False)

        b.add_new_template(self.f1.template_dict)
        b.rebin_external_data(1)

        b.add_poiss_model('ps_model','$A_{ps}$',[0.0,3.0],False)
        b.add_poiss_model('p7','$A_{p7}$',[0.0,2.0],False)
        b.add_poiss_model('bubs','$A_{bubs}$',[0.0,2.0],False)
        b.add_poiss_model('iso','$A_{iso}$',[0.0,2.0],False)
        # Add in a fixed J_map template
        b.add_fixed_templates({'J_map':[norm*self.J_map_arr[self.flux_array_ebin]/np.mean(self.J_map_arr[self.flux_array_ebin])]})

        b.initiate_poissonian_edep()
        return b

    def compute_template_spectrum(self):
        """ Code to compute the spectrum in physical units 
            Also calculate the central energy bins
            NB: here store flux, not E^2*flux as in Ben's original code """
        b = self.setup_b_instance(1,add_ps_mask=False)
        En_min = b.CTB_en_bins[0]
        En_max = b.CTB_en_bins[1]
        dE = En_max - En_min
        self.En_center = 10**((np.log10(En_max)+np.log10(En_min))/2)

        flux_map = b.fixed_template_dict_nested['summed_templates'] / b.CTB_exposure_maps_masked_compressed[0]/float(hp.nside2pixarea(b.nside))
        self.spectrum = np.mean(flux_map)/dE


    def calc_flux_array(self):
        """ Function to calculate the LL for an array of fluxes in a given energy bin """
       
        # First determine the associated spectrum
        self.compute_template_spectrum()

        # Calculate baseline counts to normalise fluxes we scan over
        # Go from 10**(bin_min)*mean up to 10**(bin_max)*mean in nbins steps
        b = self.setup_b_instance(0,add_ps_mask=True)
        mean = np.sum(b.CTB_masked_compressed[0])/len(b.CTB_masked_compressed[0])
        A_array = mean*10**np.linspace(self.bin_min,self.bin_max,self.nbins)

        # Array to get LLs when no profile likelihood run
        norun = np.array([1.0, 1.0, 1.0, 1.0])

        # Now setup and compute the arrays
        LL_array = np.array([]) 
        A_array_short = np.array([])
        spect_array = np.array([])
        LL_array_norun = np.array([])

        for i in range(len(A_array)):
            print "on i =",i
            # Calculate LL
            if i == 0:
                b1 = self.setup_b_instance(A_array[i],add_ps_mask=True)
            else:
                for key in b1.fixed_template_dict_nested.keys():
                    b1.fixed_template_dict_nested[key] = b1.fixed_template_dict_nested[key]*A_array[i]/A_array[i-1]
            ll_val = b1.ll(norun,4,4)
            # Make triangle

            # Append to arrays
            LL_array = np.append(LL_array,ll_val)
            A_array_short = np.append(A_array_short,A_array[i])
            spect_array = self.spectrum*np.array(A_array_short)

        # Save output
        np.save(work_dir+'ScanOutput/'+self.tag+'/En_array-'+str(self.flux_array_ebin)+'.npy',self.En_center)
        np.save(work_dir+'ScanOutput/'+self.tag+'/LL_array-'+str(self.flux_array_ebin)+'.npy',LL_array)
        np.save(work_dir+'ScanOutput/'+self.tag+'/Flux_array-'+str(self.flux_array_ebin)+'.npy',spect_array)
