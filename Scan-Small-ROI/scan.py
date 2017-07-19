###############################################################################
# simple_scan.py
###############################################################################
#
# Simple scan in an object ROI to get background norms and then LL profile for 
# a DM halo.
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
import healpy as hp
from tqdm import *

from local_dirs import *
from minuit_functions import call_ll

# Additional modules
sys.path.append(nptf_old_dir)
sys.path.append(work_dir + '/Smooth-Maps')
sys.path.append(work_dir + '/Make-DM-Maps')
import fermi.fermi_plugin as fp
import mkDMMaps
import king_smooth as ks
import LL_inten_to_xsec as Litx

# NPTFit modules
from NPTFit import nptfit # module for performing scan
from NPTFit import create_mask as cm # module for creating the mask


class Scan():
    def __init__(self, perform_scan=0, perform_postprocessing=0, save_dir="", load_dir=None,imc=0, iobj=0, emin=0, emax=39, channel='b', nside=128, eventclass=5, eventtype=0, diff='p7', catalog_file='DarkSky_ALL_200,200,200_v3.csv', Burkert=0, boost=1, float_ps_together=1, Asimov=0, floatDM=1, verbose=0, noJprof=0, mc_dm=-1):
        
        self.catalog = pd.read_csv(work_dir + '/DataFiles/Catalogs/' + catalog_file) # Halo catalog

        self.iobj = iobj # Objects index to scan
        self.imc = imc # MC index
        self.emin = emin # Minimum energy bin
        self.emax = emax # Maximum energy bin
        self.channel = channel # Annihilation channel (see PPPC4DMID)
        self.nside = nside # Healpix nside
        self.eventclass = eventclass # Fermi eventclass -- 5 for UCV
        self.eventtype = eventtype # Fermi eventtype -- 0 (All) or 3 (Q4)
        self.diff = diff # Diffuse model -- p6v11, p7, p8
        self.Burkert = Burkert # Whether to use a Burkert (True) or NFW (False)
        self.boost = boost # Whether to use boosted or unboosted J
        self.float_ps_together = float_ps_together # Whether to float the whole PS map
        self.Asimov = Asimov # Whether to use the Asimov expectation
        self.floatDM = floatDM # Whether to float the DM in the initial scan
        self.verbose = verbose # Whether to print tqdm and Minuit output
        self.noJprof = noJprof # Whether to not do a profile over the J uncertainty
        self.save_dir = save_dir # Directory to save output files
        self.load_dir = load_dir # Directory to load intensity LLs from

        if mc_dm == -1:
            self.dm_string = "nodm"
        else:
            self.dm_string = "dm" + str(mc_dm)

        if self.save_dir != "":
            if not os.path.exists(self.save_dir):
                try:
                    os.mkdir(self.save_dir)
                except OSError as e:
                    if e.errno != 17:
                        raise   
                self.save_dir += "/"

        if self.load_dir is None:
            self.load_dir = self.save_dir

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

        if self.Asimov:
            self.mc_tag = '_Asimov'
        else:
            if self.imc != -1:
                self.mc_tag = '_mc' + str(self.imc)
            else:
                self.mc_tag = '_data'

        if perform_scan:
            self.scan()
        if perform_postprocessing:
            self.postprocess()

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
        f_global.add_bubbles()

        # If Asimov normalize the templates and create a summed map
        if self.Asimov:
            norm_file = work_dir + '/DataFiles/Misc/P8UCVA_norm.npy' 
            f_global.use_template_normalization_file(norm_file,key_suffix='-0')
            Asimov_data = np.zeros((40,hp.nside2npix(self.nside)))
            for key in f_global.template_dict.keys():
                Asimov_data += np.array(f_global.template_dict[key]) 

        ###################
        # Get DM halo map #
        ###################

        l = self.catalog.l.values[self.iobj]
        b = self.catalog.b.values[self.iobj]
        rs = self.catalog.rvir_inf.values[self.iobj]/self.catalog.cvir_inf.values[self.iobj]*1e-3
        if self.boost:
            J0 = 10**self.catalog.mulog10J_inf.values[self.iobj]
        else:
            J0 = 10**self.catalog.mulog10Jnb_inf.values[self.iobj]
        mk = mkDMMaps.mkDMMaps(z = self.catalog.z[self.iobj], r_s = rs , J_0 = J0, ell = l*np.pi/180, b = b*np.pi/180, nside=self.nside, Burkert=self.Burkert)
        DM_template_base = mk.map

        #########################################
        # Loop over energy bins to get xsec LLs #
        #########################################

        A_ary = 10**np.linspace(-6,6,200)
        LL_inten_ary = np.zeros((len(self.ebins)-1,len(A_ary)))
        inten_ary = np.zeros((len(self.ebins)-1,len(A_ary)))

        # 10 deg mask for the analysis
        analysis_mask = cm.make_mask_total(mask_ring = True, inner = 0, outer = 10, ring_b = b, ring_l = l)

        for iebin, ebin in tqdm(enumerate(np.arange(self.emin,self.emax+1)), disable = 1 - self.verbose):
            
            ######################
            # Templates and maps #
            ######################

            if self.verbose:
                print "At bin", ebin

            if self.imc is not None:
                data = np.load(mc_dir + 'MC_allhalos_p7_' + self.dm_string + '_v' + str(self.imc)+'.npy')[ebin].astype(np.float64)
            else:
                data = f_global.CTB_count_maps[ebin].astype(np.float64)

            fermi_exposure = f_global.CTB_exposure_maps[ebin]

            DM_template = DM_template_base*fermi_exposure/np.sum(DM_template_base*fermi_exposure)
            ksi = ks.king_smooth(maps_dir, ebin, self.eventclass, self.eventtype, threads=1)
            DM_template_smoothed = ksi.smooth_the_map(DM_template)

            DM_intensity_base = np.sum(DM_template_smoothed/fermi_exposure)
            
            dif = f_global.template_dict[self.diff][ebin]
            iso = f_global.template_dict['iso'][ebin]
            psc = f_global.template_dict['ps_model'][ebin]
            bub = f_global.template_dict['bubs'][ebin]

            # If doing Asimov this first scan is irrelevant, but takes no time so run
            
            ####################
            # NPTFit norm scan #
            ####################
            
            n = nptfit.NPTF(tag='norm_o'+str(self.iobj)+'_E'+str(ebin)+self.mc_tag)
            n.load_data(data, fermi_exposure)

            n.load_mask(analysis_mask)

            n.add_template(dif, self.diff)
            n.add_template(iso, 'iso')
            n.add_template(psc, 'psc')
            n.add_template(bub, 'bub')

            n.add_poiss_model(self.diff, '$A_\mathrm{dif}$', [0,10], False)
            n.add_poiss_model('iso', '$A_\mathrm{iso}$', [0,20], False)
            
            if (np.sum(bub*np.logical_not(analysis_mask)) != 0):
                n.add_poiss_model('bub', '$A_\mathrm{bub}$', [0,10], False)

            if self.floatDM:
                if ebin >= 7: 
                    # Don't float DM in initial scan for < 1 GeV. Below here
                    # Fermi PSF is so large that we find the DM often picks up
                    # spurious excesses in MC.
                    n.add_template(DM_template_smoothed, 'DM')
                    n.add_poiss_model('DM', '$A_\mathrm{DM}$', [0,1000], False)

            if self.float_ps_together:
                n.add_poiss_model('psc', '$A_\mathrm{psc}$', [0,10], False)
            else:
                # Astropy-formatted coordinates of cluster
                c2 = SkyCoord("galactic", l=[l]*u.deg, b=[b]*u.deg)
                idx3fgl_10, _, _, _ = c2.search_around_sky(self.c3, 10*u.deg)
                idx3fgl_18, _, _, _ = c2.search_around_sky(self.c3, 18*u.deg)
                
                ps_map_outer = np.zeros(hp.nside2npix(self.nside))
                for i3fgl in idx3fgl_18:
                    ps_file = np.load(ps_indiv_dir + '/ps_temp_128_5_'+str(self.eventtype)+'_'+str(i3fgl)+'.npy')
                    ps_map = np.zeros(hp.nside2npix(self.nside))
                    ps_map[np.vectorize(int)(ps_file[::,ebin,0])] = ps_file[::,ebin,1]
                    if i3fgl in idx3fgl_10: # If within 10 degrees, float individually
                        n.add_template(ps_map, 'ps_'+str(i3fgl))
                        n.add_poiss_model('ps_'+str(i3fgl), '$A_\mathrm{ps'+str(i3fgl)+'}$', [0,10], False)
                    else: # Otherwise, add to be floated together
                        ps_map_outer += ps_map

                if np.sum(ps_map_outer) != 0:
                    n.add_template(ps_map_outer, 'ps_outer')
                    n.add_poiss_model('ps_outer', '$A_\mathrm{ps_outer}$', [0,10], False)
                
            n.configure_for_scan()

            ##########
            # Minuit #
            ##########

            # Skip this step if there is 0 data (higher energy bins)
            if np.sum(data*np.logical_not(analysis_mask)) != 0: 
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
                m.migrad(ncall=10000, precision=1e-14)
                
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
            if self.Asimov: # Use background expectation for the data
                nDM.load_data(Asimov_data[ebin], fermi_exposure)
                nDM.add_template(Asimov_data[ebin], 'bkg_sum')
            else:
                nDM.load_data(data, fermi_exposure)
                nDM.add_template(bkg_sum, 'bkg_sum')
            
            # If there is no data, only go over pixels where DM is non-zero
            if np.sum(data*np.logical_not(analysis_mask)) != 0:
                nDM.load_mask(analysis_mask)
            else:
                nodata_mask = DM_template_smoothed == 0
                nDM.load_mask(nodata_mask)
            nDM.add_poiss_model('bkg_sum', '$A_\mathrm{bkg}$', fixed=True, fixed_norm=1.0)
            
            #####################
            # Get intensity LLs #
            #####################
                               
            for iA, A in enumerate(A_ary):
                new_n2 = copy.deepcopy(nDM)
                new_n2.add_template(A*DM_template_smoothed,'DM')
                new_n2.add_poiss_model('DM','DM',False,fixed=True,fixed_norm=1.0)
                new_n2.configure_for_scan()
                max_LL = new_n2.ll([])
                
                LL_inten_ary[iebin, iA] = max_LL
                inten_ary[iebin, iA] = DM_intensity_base*A

        np.savez(self.save_dir + 'LL_inten_o'+str(self.iobj)+self.mc_tag, LL=LL_inten_ary, intens=inten_ary)

    def postprocess(self):

        ##############################
        # Get intensity without xsec #
        ##############################

        m_ary = np.array([1.00000000e+01,1.50000000e+01,2.00000000e+01,2.50000000e+01,3.00000000e+01,4.00000000e+01,5.00000000e+01,6.00000000e+01,7.00000000e+01,8.00000000e+01,9.00000000e+01,1.00000000e+02,1.10000000e+02,1.20000000e+02,1.30000000e+02,1.40000000e+02,1.50000000e+02,1.60000000e+02,1.80000000e+02,2.00000000e+02,2.20000000e+02,2.40000000e+02,2.60000000e+02,2.80000000e+02,3.00000000e+02,3.30000000e+02,3.60000000e+02,4.00000000e+02,4.50000000e+02,5.00000000e+02,5.50000000e+02,6.00000000e+02,6.50000000e+02,7.00000000e+02,7.50000000e+02,8.00000000e+02,9.00000000e+02,1.00000000e+03,1.10000000e+03,1.20000000e+03,1.30000000e+03,1.50000000e+03,1.70000000e+03,2.00000000e+03,2.50000000e+03,3.00000000e+03,4.00000000e+03,5.00000000e+03,6.00000000e+03,7.00000000e+03,8.00000000e+03,9.00000000e+03,1.00000000e+04])

        # If b use the precomputed value
        if self.channel == 'b':
            PPnoxsec_ary = np.load(work_dir + '/DataFiles//PP-Factor/PPnoxsec_b_ary.npy')
        else:
            dNdLogx_df = pd.read_csv(work_dir + '/DataFiles//PP-Factor/AtProduction_gammas.dat', delim_whitespace=True)
            
            PPnoxsec_ary = np.zeros(shape=(len(m_ary),len(self.ebins)-1))
            for mi in range(len(m_ary)):
                dNdLogx_ann_df = dNdLogx_df.query('mDM == ' + (str(np.int(float(m_ary[mi])))))[['Log[10,x]',self.channel]]
                Egamma = np.array(m_ary[mi]*(10**dNdLogx_ann_df['Log[10,x]']))
                dNdEgamma = np.array(dNdLogx_ann_df[self.channel]/(Egamma*np.log(10)))
                dNdE_interp = interpolate.interp1d(Egamma, dNdEgamma)
                for ei in range(len(self.ebins)-1): # -1 because self.ebins-1 bins, self.ebins edges
                    if self.ebins[ei] < m_ary[mi]: # Only have flux if m > Ebin
                        if self.ebins[ei+1] < m_ary[mi]: # Whole bin is inside
                            PPnoxsec_ary[mi,ei] = 1.0/(8*np.pi*m_ary[mi]**2)*integrate.quad(lambda x: dNdE_interp(x), self.ebins[ei], self.ebins[ei+1])[0]
                        else: # Bin only partially contained
                            PPnoxsec_ary[mi,ei] = 1.0/(8*np.pi*m_ary[mi]**2)*integrate.quad(lambda x: dNdE_interp(x), self.ebins[ei], m_ary[mi])[0]

        ########################################
        # Load appropriate J-factor and errors #
        ########################################

        if self.Burkert:
            if self.boost:
                mulog10J = self.catalog[u'mulog10JB_inf'].values[self.iobj]
                siglog10J = self.catalog[u'siglog10JB_inf'].values[self.iobj]
            else:
                mulog10J = self.catalog[u'mulog10JBnb_inf'].values[self.iobj]
                siglog10J = self.catalog[u'siglog10JBnb_inf'].values[self.iobj]
        else:
            if self.boost:
                mulog10J = self.catalog[u'mulog10J_inf'].values[self.iobj]
                siglog10J = self.catalog[u'siglog10J_inf'].values[self.iobj]
            else:
                mulog10J = self.catalog[u'mulog10Jnb_inf'].values[self.iobj]
                siglog10J = self.catalog[u'siglog10Jnb_inf'].values[self.iobj]


        #############################################
        # Interpolate intensity LLs to get xsec LLs #
        #############################################

        LL_inten_file  =  np.load(self.load_dir + 'LL_inten_o'+str(self.iobj)+self.mc_tag+'.npz')
        LL_inten_ary, inten_ary = LL_inten_file['LL'], LL_inten_file['intens']
        xsecs = np.logspace(-33,-18,301)
        LL2_xsec_m_ary = np.zeros((len(m_ary),len(xsecs))) # 2 x LL, ready for TS

        # Interpolate to the limit without profiling over the J uncertainty if specified
        if self.noJprof:
            for im in tqdm(range(len(m_ary)), disable = 1 - self.verbose):
                for ixsec, xsec in enumerate(xsecs):
                    for iebin in range(len(self.ebins)-1):
                        intval = PPnoxsec_ary[im][iebin]*10**mulog10J*xsec
                        LL2_xsec_m_ary[im,ixsec] += 2*np.interp(intval,inten_ary[iebin], LL_inten_ary[iebin])
        # Otherwise profile over the error in J
        else:
            for im in tqdm(range(len(m_ary)), disable = 1 - self.verbose):
                LL2_xsec_m_ary[im] = Litx.construct_xsec_LL(xsecs,self.ebins,PPnoxsec_ary[im],LL_inten_ary,inten_ary,mulog10J,siglog10J)

        ####################################################
        # Calculate val, loc and xsec of max TS, and limit #
        ####################################################

        TS_m_xsec = np.zeros(3)
        TS_m_xsec[2] = xsecs[0]
        lim_ary = np.zeros(len(m_ary))
        for im in range(len(m_ary)): 
            TS_xsec_ary = LL2_xsec_m_ary[im] - LL2_xsec_m_ary[im][0]
            
            # Find value, location and xsec at the max TS (as a fn of mass)
            max_loc = np.argmax(TS_xsec_ary)
            max_TS = TS_xsec_ary[max_loc]
            if max_TS > TS_m_xsec[0]:
                TS_m_xsec[0] = max_TS
                TS_m_xsec[1] = im
                TS_m_xsec[2] = xsecs[max_loc]
            
            # Calculate limit
            for xi in range(max_loc,len(xsecs)):
                val = TS_xsec_ary[xi] - max_TS
                if val < -2.71:
                    scale = (TS_xsec_ary[xi-1]-max_TS+2.71)/(TS_xsec_ary[xi-1]-TS_xsec_ary[xi])
                    lim_ary[im] = 10**(np.log10(xsecs[xi-1])+scale*(np.log10(xsecs[xi])-np.log10(xsecs[xi-1])))
                    break

        #####################################
        # Setup save string and output data #
        #####################################

        save_LLx_str = 'LL2_TSmx_lim_'+self.channel
        if not self.boost:
            save_LLx_str += '_nb'
        if self.Burkert:
            save_LLx_str += '_Burk'
        if self.noJprof:
            save_LLx_str += '_noJprof'
        save_LLx_str += '_o'+str(self.iobj) 
        save_LLx_str += self.mc_tag

        np.savez(self.save_dir + save_LLx_str, LL2=LL2_xsec_m_ary, TSmx=TS_m_xsec, lim=lim_ary)
