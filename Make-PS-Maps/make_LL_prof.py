import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import sys, os

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt

import corner
import copy

from matplotlib import rc
###import pymultinest, triangle

import healpy as hp
import pandas as pd

import fermi.fermi_plugin as fp

mpl.rcParams.update({'font.size': 18, 'font.family': 'serif'})


###Make sure you append the git directory
sys.path.append('/group/hepheno/bsafdi/NPTF-working/NPTF-ID-Catalog/mkDMMaps/')
###New modules to load in
#import NFW
#import mkDMMaps

from NPTFit import nptfit # module for performing scan
from NPTFit import create_mask as cm # module for creating the mask
from NPTFit import dnds_analysis # module for analysing the output


######################################################################################################
######################################################################################################
######################################################################################################
######################################################################################################
#####################################     Catalog   ##################################################

class catalog:
    def __init__(self,catalog_path,nside=128):
        self.nside = nside
        self.tully_top_30 = pd.read_csv(catalog_path)

        self.get_variables()

    def get_variables(self):
        tully_top_30 = self.tully_top_30
        self.z_array = tully_top_30.z.values #[0.01]
        self.r_s_array = tully_top_30.rvir.values*1e-3 #Convert to Mpc#[1.0]
        self.J_array = 10**tully_top_30.logJ.values #[1e9]
        self.ell_array = tully_top_30.l.values*2*np.pi/360 #[0.0]
        self.b_array = tully_top_30.b.values*2*np.pi/360 #[0.0]
        

    def return_map(self,i):
        mk = mkDMMaps.mkDMMaps(z = self.z_array[i],r_s = self.r_s_array[i], J_0 = self.J_array[i],ell = self.ell_array[i],b = self.b_array[i],nside=self.nside)
        return mk.map #Here, we add to our final map the specific map for the i^th halo.

######################################################################################################
#####################################     Load Data   ################################################


class load_data:
    def __init__(self,Emin,eventclass=5,eventtype=3,fermi_data_dir = '/mnt/hepheno/FermiData/', maps_dir = '/mnt/hepheno/CTBCORE/',nside=128,mask_type='top300',force_energy=False,newstyle=1,diff='p8',data_July16=True):
        self.f = fp.fermi_plugin(maps_dir,fermi_data_dir=fermi_data_dir,CTB_en_min=Emin,CTB_en_max=Emin+1,nside=nside,eventclass=eventclass,eventtype=eventtype,newstyle=newstyle,data_July16=data_July16)

        self._setup_f(mask_type,Emin,force_energy,eventclass,eventtype,diff)
        self.extract()


    def _setup_f(self,mask_type,Emin,force_energy,eventclass,eventtype,diff):
        self.f.make_ps_mask(mask_type = mask_type,energy_bin = Emin,force_energy=force_energy)
        self.f.add_diffuse_newstyle(comp = diff,eventclass = eventclass, eventtype = eventtype) #diffuse
        self.f.add_bubbles() #bubbles
        self.f.add_iso()  #iso
        self.f.add_ps_model()

        self.f.load_psf()

    def extract(self):
        self.counts = self.f.CTB_count_maps[0]
        self.exposure = self.f.CTB_exposure_maps[0]
        self.ps_mask = self.f.ps_mask_array[0]
        self.template_dict = self.f.template_dict
        self.psf_rad = self.f.average_PSF_dict['iso']*np.pi/180.0


######################################################################################################
#####################################     Norm Scan   ################################################

class norm_scan:
    def __init__(self,counts,exposure,template_dict,ps_mask,ell,b,rad=20,tag='tmp',nlive=500): 
        #rad in deg
        self.ell = ell
        self.b = b
        self.rad = rad
        self.make_mask(ps_mask)

        self.n = nptfit.NPTF(tag=tag)
        self.n.load_data(counts, exposure)
        self.n.load_mask(self.total_mask)

        active_pixels = np.sum(np.logical_not(self.total_mask))
        if active_pixels > 0:
            self.configure_for_scan(template_dict)
            self.perform_scan(500)
            self.make_new_template()
        else:
            print "No data!"
            self.new_template_dict = None

    def make_mask(self,ps_mask):
        mask_0 = cm.make_mask_total(mask_ring = True, inner = 0, outer = self.rad, ring_b = self.b*180/np.pi, ring_l = self.ell*180/np.pi)
        self.total_mask = np.vectorize(bool)(ps_mask+mask_0)

    def configure_for_scan(self,template_dict):
        self.keys = template_dict.keys()

        for key in self.keys:
            self.n.add_template(template_dict[key][0],key)

        for key in self.keys:
            self.n.add_poiss_model(key, key[0:2], [0,20], False)

        self.n.configure_for_scan()

    def perform_scan(self,nlive):
        self.n.perform_scan(nlive=nlive)
        self.n.load_scan()
        self.medians = np.median(self.n.samples,axis=0)
        print "The medians from the norm are ", self.medians

    def make_new_template(self):
        self.new_template_dict = {}
        for i in range(len(self.keys)):
            key = self.keys[i]
            self.new_template_dict[key] = self.n.templates_dict[key]*self.medians[i]

######################################################################################################
#####################################     LL Profile   ###############################################

class LL_scan:
    def __init__(self,A_array,DM_template,counts,exposure,template_dict,ps_mask,ell,b,rad=5,tag='tmp',nlive=500): 
        #rad in deg
        self.DM_template = DM_template
        self.A_array = A_array

        self.ell = ell
        self.b = b
        self.rad = rad
        self.make_mask(ps_mask)

        self.compute_DM_intensity_base(exposure)

        self.n = nptfit.NPTF(tag=tag)
        self.n.load_data(counts, exposure)
        self.n.load_mask(self.total_mask)

        self.setup_base_n(template_dict)
        self.perform_scan(500)
        #self.make_new_template()

    def make_mask(self,ps_mask):
        mask_0 = cm.make_mask_total(mask_ring = True, inner = 0, outer = self.rad, ring_b = self.b*180/np.pi, ring_l = self.ell*180/np.pi)
        self.total_mask = np.vectorize(bool)(ps_mask+mask_0)
        self.active_pixels = np.sum(np.logical_not(self.total_mask))

    def compute_DM_intensity_base(self,exposure):
        self.DM_intensity_base = np.sum(self.DM_template/exposure)/(4*np.pi)

    def setup_base_n(self,template_dict):
        keys_full = template_dict.keys()
        self.keys = []
        for key in keys_full:
            if key != 'ps_model':
                self.keys += [key]

        for key in keys_full:
            self.n.add_template(template_dict[key],key)

        for key in self.keys:
            self.n.add_poiss_model(key, key[0:2], False,fixed=True, fixed_norm=1.0)

        self.n.add_poiss_model('ps_model','$ps_{model}$',[0,2],False)

    def perform_n_instance(self,n,A,nlive=500):
        self.new_n = copy.deepcopy(n)
        self.new_n.add_template(A*self.DM_template,'DM')
        self.new_n.add_poiss_model('DM','DM',False,fixed=True,fixed_norm=1.0)
        
        
        if self.active_pixels > 0:
            self.new_n.configure_for_scan()

            self.new_n.perform_scan(nlive=nlive)
            self.new_n.load_scan()
            max_LL = self.new_n.log_like_ptf(np.median(self.new_n.samples,axis=0))
        else:
            print "No active pixels!"
            max_LL = 0.0

        return max_LL, self.DM_intensity_base*A

        #self.n.configure_for_scan()

    def perform_scan(self,nlive):
        self.LL_array = []
        self.intens_array = []
        for i in range(len(self.A_array)):
            max_LL, intens = self.perform_n_instance(self.n,self.A_array[i],nlive=nlive)
            self.LL_array += [max_LL]
            self.intens_array += [intens]

        self.LL_array = np.array(self.LL_array)
        self.intens_array = np.array(self.intens_array)



        # self.n.perform_scan(nlive=nlive)
        # self.n.load_scan()
        # self.medians = np.median(self.n.samples,axis=0)


