###############################################################################
# simple_scan.py
###############################################################################
#
# Simple scan in an object ROI to get background norms and then LL profile for 
# a DM halo.
#
# Here we do this just to get the spectra of one object
#
###############################################################################

import os,sys
import argparse
import copy

import numpy as np
from iminuit import Minuit
import pandas as pd
from scipy import interpolate, integrate
from astropy import units as u
from astropy.coordinates import SkyCoord, Distance
from astropy.cosmology import Planck15
import healpy as hp
from tqdm import *
from astropy.io import fits

from local_dirs import *
from minuit_functions import call_ll

# Additional modules
sys.path.append(nptf_old_dir)
sys.path.append(work_dir + '/Smooth-Maps')
sys.path.append(work_dir + '/Make-DM-Maps')
import fermi.fermi_plugin as fp
import mkDMMaps
import king_smooth as ks
sys.path.append(work_dir + '/Scan-Small-ROI')
import LL_inten_to_xsec as Litx

# NPTFit modules
from NPTFit import nptfit # module for performing scan
from NPTFit import create_mask as cm # module for creating the mask


class Scan():
    def __init__(self, save_dir='/tigress/nrodd/DM-Catalog-Scan/Scan-One-Object/Virgo/', iobj=1, emin=4, emax=30, nside=128, eventclass=5, eventtype=0, diff='p8', catalog_file='2MRSLocalTully_ALL_DATAPAPER_Planck15_v7.csv', Burkert=0, use_boost=0, boost=1, float_ps_together=1, floatDM=1, verbose=1):
        
        self.catalog = pd.read_csv(work_dir + '/DataFiles/Catalogs/' + catalog_file) # Halo catalog

        self.iobj = iobj # Objects index to scan
        self.emin = emin # Minimum energy bin
        self.emax = emax # Maximum energy bin
        self.nside = nside # Healpix nside
        self.eventclass = eventclass # Fermi eventclass -- 5 for UCV
        self.eventtype = eventtype # Fermi eventtype -- 0 (All) or 3 (Q4)
        self.diff = diff # Diffuse model -- p6v11, p7, p8
        self.Burkert = Burkert # Whether to use a Burkert (True) or NFW (False)
        self.boost = boost # Whether to use boosted or unboosted J
        self.use_boost = use_boost # Whether to put down a boosted profile
        self.float_ps_together = float_ps_together # Whether to float the whole PS map
        self.floatDM = floatDM # Whether to float the DM in the initial scan
        self.verbose = verbose # Whether to print tqdm and Minuit output
        self.save_dir = save_dir # Directory to save output files

        if self.save_dir != "":
            if not os.path.exists(self.save_dir):
                try:
                    os.mkdir(self.save_dir)
                except OSError as e:
                    if e.errno != 17:
                        raise   
                self.save_dir += "/"

        # If floating sources individually, find nearby 3FGL PSs
        if not self.float_ps_together:
            source_3fg_df = pd.read_csv(work_dir + '/DataFiles/Catalogs/3fgl.dat', sep='|', comment='#')
            source_3fg_df.rename(columns=lambda x: x.strip(), inplace=True) # Strip whitespace
            for col in source_3fg_df.columns.values:
                try:
                    source_3fg_df[col] = source_3fg_df[col].map(str.strip)
                except TypeError:
                    continue
            source_3fg_df = source_3fg_df.convert_objects(convert_numeric=True)
            # Coordinates of nearby 3FGL
            self.c3 = SkyCoord("galactic", l=source_3fg_df['_Lii']*u.deg, b=source_3fg_df['_Bii']*u.deg)

        self.ebins = 2*np.logspace(-1,3,41)[self.emin:self.emax+2]
        self.de = self.ebins[1:] - self.ebins[:-1]
        self.emid = 10**((np.log10(self.ebins[1:]) + np.log10(self.ebins[:-1]))/2.)

        self.mc_tag = '_data'

        self.scan()

    def scan(self):

        ################
        # Fermi plugin #
        ################

        # Load the Fermi plugin - always load all energy bins, extract what is needed
        f_global = fp.fermi_plugin(maps_dir,fermi_data_dir=fermi_data_dir,work_dir=work_dir,CTB_en_min=0,CTB_en_max=40,nside=self.nside,eventclass=self.eventclass,eventtype=self.eventtype,newstyle=1,data_July16=True)

        # Load necessary templates
        f_global.add_diffuse_newstyle(comp = self.diff,eventclass = self.eventclass, eventtype = self.eventtype) 
        f_global.add_iso()  
        ps_temp = np.load(work_dir + '/DataFiles/PS-Maps/ps_map.npy')
        f_global.add_template_by_hand(comp='ps_model',template=ps_temp)

        ###################
        # Get DM halo map #
        ###################

        l = self.catalog.l.values[self.iobj]
        b = self.catalog.b.values[self.iobj]

        rs = self.catalog.rs.values[self.iobj]*1e-3
        if self.boost:
            J0 = 10**self.catalog.mulog10J_inf.values[self.iobj]
        else:
            J0 = 10**self.catalog.mulog10Jnb_inf.values[self.iobj]
        mk = mkDMMaps.mkDMMaps(z = self.catalog.z[self.iobj], r_s = rs , J_0 = J0, ell = l*np.pi/180, b = b*np.pi/180, nside=self.nside, use_boost=self.use_boost, Burkert=self.Burkert)
        DM_template_base = mk.map

        #########################################
        # Loop over energy bins to get spectrum #
        #########################################

        # 10 deg mask for the analysis
        analysis_mask_base = cm.make_mask_total(mask_ring = True, inner = 0, outer = 10, ring_b = b, ring_l = l)

        # ROI where we will normalise our templates
        ROI_mask = cm.make_mask_total(mask_ring = True, inner = 0, outer = 2, ring_b = b, ring_l = l)
        ROI = np.where(ROI_mask == 0)[0]

        # Setup output
        output_norms = np.zeros((self.emax+1-self.emin,4,2))

        for iebin, ebin in tqdm(enumerate(np.arange(self.emin,self.emax+1)), disable = 1 - self.verbose):
            
            ######################
            # Templates and maps #
            ######################

            if self.verbose:
                print "At bin", ebin

            data = f_global.CTB_count_maps[ebin].astype(np.float64)
            # Add large scale mask to analysis mask
            els_str = ['0.20000000','0.25178508','0.31697864','0.39905246','0.50237729','0.63245553','0.79621434','1.0023745','1.2619147','1.5886565','2.0000000','2.5178508','3.1697864','3.9905246','5.0237729','6.3245553','7.9621434','10.023745','12.619147','15.886565','20.000000','25.178508','31.697864','39.905246','50.237729','63.245553','79.621434','100.23745','126.19147','158.86565','200.00000','251.78508','316.97864','399.05246','502.37729','632.45553','796.21434','1002.3745','1261.9147','1588.6565']
            ls_mask_load = fits.open('/tigress/nrodd/LargeObjMask/Allpscmask_3FGL-energy'+els_str[ebin]+'large-obj.fits')
            ls_mask = np.array([np.round(val) for val in hp.ud_grade(ls_mask_load[0].data,self.nside,power=0)])
            analysis_mask = np.vectorize(bool)(analysis_mask_base+ls_mask)

            fermi_exposure = f_global.CTB_exposure_maps[ebin]

            DM_template = DM_template_base*fermi_exposure/np.sum(DM_template_base*fermi_exposure)
            ksi = ks.king_smooth(maps_dir, ebin, self.eventclass, self.eventtype, threads=1)
            DM_template_smoothed = ksi.smooth_the_map(DM_template)
            DM_intensity_base = np.sum(DM_template_smoothed/fermi_exposure)
            
            dif = f_global.template_dict[self.diff][ebin]
            iso = f_global.template_dict['iso'][ebin]
            psc = f_global.template_dict['ps_model'][ebin]

            # Get mean values in ROI
            dif_mu = np.mean(dif[ROI])
            iso_mu = np.mean(iso[ROI])
            psc_mu = np.mean(psc[ROI])
            DM_mu = np.mean(DM_template_smoothed[ROI])
            exp_mu = np.mean(fermi_exposure[ROI])

            ####################
            # NPTFit norm scan #
            ####################
            
            n = nptfit.NPTF(tag='norm_o'+str(self.iobj)+'_E'+str(ebin)+self.mc_tag)
            n.load_data(data, fermi_exposure)

            n.load_mask(analysis_mask)

            n.add_template(dif, self.diff)
            n.add_template(iso, 'iso')
            n.add_template(psc, 'psc')
            n.add_template(DM_template_smoothed, 'DM')

            n.add_poiss_model(self.diff, '$A_\mathrm{dif}$', [0,10], False)
            n.add_poiss_model('iso', '$A_\mathrm{iso}$', [0,20], False)
            n.add_poiss_model('psc', '$A_\mathrm{psc}$', [0,10], False)
            n.add_poiss_model('DM', '$A_\mathrm{DM}$', [0,1000], False)
            
            n.configure_for_scan()

            ##########
            # Minuit #
            ##########

            keys = n.poiss_model_keys
            limit_dict = {}
            init_val_dict = {}
            step_size_dict = {}
            for key in keys:
                if key == 'DM':
                    limit_dict['limit_'+key] = (0,1000)
                else:
                    limit_dict['limit_'+key] = (0,50)
                init_val_dict[key] = 0.0
                step_size_dict['error_'+key] = 1.0
            other_kwargs = {'print_level': self.verbose, 'errordef': 1}
            z = limit_dict.copy()
            z.update(other_kwargs)
            z.update(limit_dict)
            z.update(init_val_dict)
            z.update(step_size_dict)
            f = call_ll(len(keys),n.ll,keys)
            m = Minuit(f,**z)
            m.migrad(ncall=30000, precision=1e-14)
            

            # Output spectra in E^2 dN/dE, in units [GeV/cm^2/s/sr] as mean in 2 degrees
            output_norms[iebin,0,0] = m.values['p8']*dif_mu/exp_mu*self.emid[iebin]**2/self.de[iebin]
            output_norms[iebin,0,1] = m.errors['p8']*dif_mu/exp_mu*self.emid[iebin]**2/self.de[iebin]

            output_norms[iebin,1,0] = m.values['iso']*iso_mu/exp_mu*self.emid[iebin]**2/self.de[iebin]
            output_norms[iebin,1,1] = m.errors['iso']*iso_mu/exp_mu*self.emid[iebin]**2/self.de[iebin]

            output_norms[iebin,2,0] = m.values['psc']*psc_mu/exp_mu*self.emid[iebin]**2/self.de[iebin]
            output_norms[iebin,2,1] = m.errors['psc']*psc_mu/exp_mu*self.emid[iebin]**2/self.de[iebin]

            output_norms[iebin,3,0] = m.values['DM']*DM_mu/exp_mu*self.emid[iebin]**2/self.de[iebin]
            output_norms[iebin,3,1] = m.errors['DM']*DM_mu/exp_mu*self.emid[iebin]**2/self.de[iebin]

            ###################################
            # NPTFit fixed DM and bkg profile #
            ###################################
            
            # Make background sum and initiate second scan
            # If was no data leave bkg_sum as 0
            bkg_sum = np.zeros(len(data))
            if np.sum(data*np.logical_not(analysis_mask)) != 0:
                for key in keys:
                    if key != 'DM': # Don't add DM in here
                        if m.values[key] != 0:
                            bkg_sum += n.templates_dict[key]*m.values[key]
                        else: # If zero, use ~parabolic error
                            bkg_sum += n.templates_dict[key]*m.errors[key]/2.
            
            
            nDM = nptfit.NPTF(tag='dm_o'+str(self.iobj)+'_E'+str(ebin)+self.mc_tag)
            nDM.load_data(data, fermi_exposure)
            nDM.add_template(bkg_sum, 'bkg_sum')
            
            # If there is no data, only go over pixels where DM is non-zero
            if np.sum(data*np.logical_not(analysis_mask)) != 0:
                nDM.load_mask(analysis_mask)
            else:
                nodata_mask = DM_template_smoothed == 0
                nDM.load_mask(nodata_mask)
            nDM.add_poiss_model('bkg_sum', '$A_\mathrm{bkg}$', fixed=True, fixed_norm=1.0)
            
        np.save(self.save_dir + 'spec_o'+str(self.iobj)+self.mc_tag, output_norms)
