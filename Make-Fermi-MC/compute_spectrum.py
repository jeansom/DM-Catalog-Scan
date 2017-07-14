#################
###relevant modules
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import healpy as hp
import copy

import pulsars.masks as masks

#################
###relevant modules from NPTF package
import fermi.fermi_plugin as fp  #module for loading templates needed for scan
import config.bayesian_scan_models as bsm #module for performing scan
import analysis_classes.fermi_analysis as fa  #module for analysing san

#################
###common directories
from global_variables import *

import argparse

### Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-r",
                  action="store_true", dest="scan", default=False)
parser.add_argument("-a",
                  action="store_true", dest="analysis", default=False)
results = parser.parse_args()
scan=results.scan
analysis=results.analysis


run_tag_base='getnorm'
band_mask=30
nlive=800
eventclass=5
eventtype=0
newstyle=1
data_July16=True
tag = 'spct'
data_type='p8'  #also p8
CTB_start_bin=0
CTB_end_bin=39
nside=128

band_mask_range =  [-band_mask,band_mask]  #measured from the Galactic plane
mask_ring = False

mask_type = 'top300' #can also be '0.99', 'top300', or 'False'
force_energy=True #we will force the PS mask from a specific energy bin
energy_bin_mask = 10

norm_file_path='/tigress/nrodd/2mass2furious/MakeMC/P8UCVA_norm'
spect_file_path='/tigress/nrodd/2mass2furious/MakeMC/P8UCVA_spec'

f_total = fp.fermi_plugin(maps_dir,fermi_data_dir=fermi_data_dir,CTB_en_min=CTB_start_bin,CTB_en_max=CTB_end_bin+1,nside=nside,eventclass=eventclass,eventtype=eventtype,newstyle=newstyle,data_July16=data_July16)
#f_total.make_ps_mask(mask_type = mask_type,energy_bin = energy_bin_mask,force_energy=force_energy)
f_total.make_ps_mask(mask_type = mask_type,energy_bin = energy_bin_mask,force_energy=force_energy)

f_total.add_diffuse_newstyle(comp = 'p7',eventclass = eventclass, eventtype = eventtype) #diffuse
f_total.add_bubbles() #bubbles
f_total.add_iso()  #iso
f_total.add_ps_model()

def setup_b_instance(ebin_min):
    global b, f, run_tag
    
    run_tag=run_tag_base + '-'+str(ebin_min) #set the run tag using ebin_min
    f = copy.deepcopy(f_total) #copy f_total and reduce to energy subset
    f.reduce_to_energy_subset(ebin_min-CTB_start_bin,ebin_min+1-CTB_start_bin)
    
    b = bsm.bayesian_scan_NPTF(tag=tag,nside=nside,work_dir=work_dir,psf_dir=psf_dir,nlive=nlive)
    b.load_external_data(f.CTB_en_bins,f.CTB_count_maps,f.CTB_exposure_maps)
    ##b.make_mask_total(band_mask_range =  band_mask_range,mask_ring = mask_ring,ps_mask_array = f.ps_mask_array)

    b.make_mask_total(band_mask_range =  band_mask_range,mask_ring = mask_ring,ps_mask_array = f.ps_mask_array)
    
    b.add_new_template(f.template_dict)
    b.rebin_external_data(1)


    ##############
    temp_key_list = ['ps_model','p7','bubs','iso']
    triangle_name = ['$A_{ps}$','$A_{diff}$','$A_{bubs}$','$A_{iso}$']
    prior_list = [[0,3],[0,2],[0,3],[0,15]]
    prior_is_log = [False,False,False,False]


    for key in b.templates_dict.keys():
        b.templates_dict[key] = np.array([b.templates_dict[key][0]])

    for i in range(len(temp_key_list)):
        b.add_poiss_model(temp_key_list[i],triangle_name[i],prior_list[i],prior_is_log[i])

    b.initiate_poissonian_edep()
    
    
def perform_scan():
    for en in range(CTB_start_bin,CTB_end_bin+1):
        setup_b_instance(en)
        b.perform_scan(run_tag=run_tag) #this time we will use multinest for the scan

def do_triangle():
    mt=fa.make_triangle(b,run_tag,edep=True)
    mt.make_triangle(return_fig=False,save_fig=True)
    
def do_spectrum():
    global cs
    kwargs_specrum={ 'band_mask_range': band_mask_range,'mask_ring': mask_ring}
    cs=fa.compute_spectra(b,run_tag,b.CTB_en_min,b.CTB_en_max,edep=True,**kwargs_specrum)
    #cs.mask_total_dict['bubs-0']=np.logical_not(b.templates_dict['bubs'][0])
    
    if f.CTB_en_min==0:
        over_write=True
    else:
        over_write=False
    
    cs.make_norm_dict()
    cs.save_norm_dict(norm_file_path,b.CTB_en_min,b.CTB_en_max,over_write= over_write)
    
    cs.make_spectra_dict()
    cs.save_spectra_dict(spect_file_path,b.CTB_en_min,b.CTB_en_max,over_write= over_write)

def perform_analysis():
    global triangle_array
    triangle_array = []
    for en in range(CTB_start_bin,CTB_end_bin+1):
        setup_b_instance(en)
        do_triangle()
        do_spectrum()

if scan:
    perform_scan()

if analysis:
    perform_analysis()
