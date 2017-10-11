###############################################################################
# simple_scan.py
###############################################################################
#
# Code to output data cubes for Christoph
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
    def __init__(self, save_dir='/tigress/nrodd/DM-Catalog-Scan/Scan-One-Object/Cubes/', iobj=1, emin=0, emax=39, nside=128, eventclass=5, eventtype=0, diff='p8', catalog_file='2MRSLocalTully_ALL_DATAPAPER_Planck15_v7.csv', Burkert=0, use_boost=0, boost=1, float_ps_together=1, floatDM=1, verbose=1):
        
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

        # ROI where we will normalise our templates
        ROI_mask = cm.make_mask_total(mask_ring = True, inner = 0, outer = 10, ring_b = b, ring_l = l)
        ROI = np.where(ROI_mask == 0)[0]

        # Setup output
        output_cube = np.zeros(self.emax+2-self.emin)
        output_cube[0] = self.catalog.mulog10J_inf.values[self.iobj]

        for iebin, ebin in tqdm(enumerate(np.arange(self.emin,self.emax+1)), disable = 1 - self.verbose):
            
            ######################
            # Templates and maps #
            ######################

            if self.verbose:
                print "At bin", ebin

            fermi_exposure = f_global.CTB_exposure_maps[ebin]

            DM_template = DM_template_base*fermi_exposure/np.sum(DM_template_base*fermi_exposure)
            ksi = ks.king_smooth(maps_dir, ebin, self.eventclass, self.eventtype, threads=1)
            DM_template_smoothed = ksi.smooth_the_map(DM_template)
            DM_intensity_base = np.sum(DM_template_smoothed/fermi_exposure)
            
            output_cube[iebin+1] = DM_intensity_base

        np.save(self.save_dir + 'J_intensity_o'+str(self.iobj), output_cube)
