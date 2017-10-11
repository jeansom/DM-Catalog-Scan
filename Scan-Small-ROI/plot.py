###############################################################################
# plot.py
###############################################################################
#
# Code to make limits as a function of mass and number of halos
#
###############################################################################

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import *

class LimitPlot():
    def __init__(self, 
        data_dir='/tigress/bsafdi/github/NPTF-working//NPTF-ID-Catalog/SimpleScan/data/FloatPS_indiv_floatDM/', 
        nmc=20,
        catalog_file='/tigress/bsafdi/github/NPTF-working/NPTF-ID-Catalog/data/Catalogs/DarkSky_ALL_200,200,200_v3.csv',
        elephantm=[0, 11, 52],
        halos_ran=100,
        halos_to_keep=100,
        bcut=20,
        cut_0p5=False,
        nonoverlap=False,
        nonoverlapradius=2,
        TS100=5,
        TS1000=9,
        TSabove=16,
        xsecslim=10,
        elephant=True,
        data_type="mc"):

        self.data_dir = data_dir # Directory where files are located
        self.halos_to_keep = halos_to_keep # Maximum number of halos to keep after all cuts
        self.nmc = nmc # Number of MCs
        self.elephant = elephant # Whether want elephant (True) or limit (False)
        self.elephantm = elephantm
        self.TS100 = TS100 # Max TS of halos with Nh < 100 in number, used with xsec cut
        self.TS1000 = TS1000 # Max TS of halos with 100 < Nh < 1000, used with xsec cut
        self.TSabove = TSabove # Max TS of halos with Nh > 1000, used with xsec cut
        self.xsecslim = xsecslim # Cut halos with best-fit xsec > this many times strongest limit
        self.nonoverlap = nonoverlap # Whether to only use halos with non-overlapping regions
        self.nonoverlapradius = nonoverlapradius # Radius of non-overlapping region
        self.data_type = data_type # "data", "Asimov" or "mc"
        self.halos_ran = halos_ran # Number of halos that actually exist as LLs
        # cut_0p5 # Whether to exclude halos with 3FGL PSs within 0.5 deg

        if self.data_type in ["data", "Asimov"]:
            self.nmc = 1

        if self.data_type != "skylocs":
            self.data_dir += '/LL2_TSmx_lim_b_o'

        # Load the catalog and extract the b values for the latitude cut, l for the ROI cut (if specified),
        # Cut at number of halos run
        catalog = pd.read_csv(catalog_file)[:self.halos_ran]
        self.b_array = catalog.b.values
        self.ell_array = catalog.l.values
        
        # Apply 0p5 cut if required
        near_3FGL = np.array([ len(catalog[u'3FGL 0.5'].values[i]) for i in range(len(self.b_array))])
        if cut_0p5:
            self.good_vals = np.where((np.abs(self.b_array) > bcut) & (near_3FGL == 2))[0]
        else:
            self.good_vals = np.where(np.abs(self.b_array) > bcut)[0]

        self.n_good_halos = len(self.good_vals)

    def return_limits(self):
        """
        Returns an array of limits of shape (halos, masses, MC)
        that can be processed to make elephant or limit plots
        """

        # Define the xsec and mass array, these should be the same as used during the runs
        xsecs = np.logspace(-33,-18,301)
        self.marr = np.array([1.00000000e+01,1.50000000e+01,2.00000000e+01,2.50000000e+01,3.00000000e+01,
                         4.00000000e+01,5.00000000e+01,6.00000000e+01,7.00000000e+01,8.00000000e+01,
                         9.00000000e+01,1.00000000e+02,1.10000000e+02,1.20000000e+02,1.30000000e+02,
                         1.40000000e+02,1.50000000e+02,1.60000000e+02,1.80000000e+02,2.00000000e+02,
                         2.20000000e+02,2.40000000e+02,2.60000000e+02,2.80000000e+02,3.00000000e+02,
                         3.30000000e+02,3.60000000e+02,4.00000000e+02,4.50000000e+02,5.00000000e+02,
                         5.50000000e+02,6.00000000e+02,6.50000000e+02,7.00000000e+02,7.50000000e+02,
                         8.00000000e+02,9.00000000e+02,1.00000000e+03,1.10000000e+03,1.20000000e+03,
                         1.30000000e+03,1.50000000e+03,1.70000000e+03,2.00000000e+03,2.50000000e+03,
                         3.00000000e+03,4.00000000e+03,5.00000000e+03,6.00000000e+03,7.00000000e+03,
                         8.00000000e+03,9.00000000e+03,1.00000000e+04])

        MC_limit_arr = np.zeros((self.halos_to_keep,len(self.marr),self.nmc))
        xsec_at_maxTS_ary = np.zeros((self.halos_to_keep,len(self.marr),self.nmc))
        halos_passed = np.zeros(self.nmc)

        for imc in tqdm_notebook(range(self.nmc)):

            if self.data_type in ["data", "Asimov"]:
                self.data_str = "_" + self.data_type
            elif self.data_type == "mc":
                self.data_str = "_mc" + str(imc)
            elif self.data_type == "skylocs":
                self.data_str = "_data"

            ################
            # Top 10 Limit #
            ################

            # As a reference point determine the best limits amongst the top 10 objects that pass our cuts

            existing_theta = np.array([])
            existing_phi = np.array([])
            best_lim = np.zeros(len(self.marr))+1e-18
            top10_count = 0

            #for iobj in tqdm_notebook(range(self.n_good_halos)):
            for iobj in range(self.n_good_halos):

                # Skip if not a good object
                if iobj not in self.good_vals: continue

                # If specified check if it overlaps with previous halos
                if self.nonoverlap:
                    obj_theta = np.pi/2-self.b_array[iobj]*np.pi/180.
                    obj_phi = self.ell_array[iobj]*np.pi/180.

                    if len(existing_theta) == 0: # At object 0
                        existing_theta = np.append(existing_theta,obj_theta)
                        existing_phi = np.append(existing_phi,obj_phi)
                    else:
                        rvals = np.arccos(np.cos(existing_theta)*np.cos(obj_theta)
                                          +np.sin(existing_theta)*np.sin(obj_theta)
                                          *np.cos(existing_phi-obj_phi))

                        if (np.min(rvals) < self.nonoverlapradius*np.pi/180.): continue # Skip, overlaps!

                        # Otherwise passed, keep its location for future overlap check
                        existing_theta = np.append(existing_theta,obj_theta)
                        existing_phi = np.append(existing_phi,obj_phi)

                # Passed, add to counter
                top10_count += 1

                # Load limit and append where stronger
                if self.data_type == 'skylocs':
                    pp_file = np.load(self.data_dir + str(imc) + "/LL2_TSmx_lim_b_o"  + str(iobj) + self.data_str + '.npz')
                else:
                    pp_file = np.load(self.data_dir +  str(iobj) + self.data_str + '.npz')
                lim = pp_file['lim']
                for im in range(len(self.marr)):
                    if lim[im] < best_lim[im]:
                        best_lim[im] = lim[im]

                # If have 10 stop, otherwise continue
                if top10_count == 10: 
                    break

            # If doing elephant, use specified masses
            # otherwise, do all masses
            if self.elephant: 
                masses = self.elephantm
            else: 
                masses = np.arange(len(self.marr))

            #for im in tqdm_notebook(masses):
            for im in masses:

                ##################
                # Combined Limit #
                ##################

                # Using the top10 limit as a baseline for cutting, compute the combined limit

                existing_theta = np.array([])
                existing_phi = np.array([])
                halos_kept = 0
                LL2_cumulative = np.zeros(len(xsecs))

                #for iobj in tqdm_notebook(self.good_vals):
                for iobj in self.good_vals:

                    # Load halo max TS, mloc and xsec values and see if passes cut
                    if self.data_type == 'skylocs':
                        pp_file = np.load(self.data_dir + str(imc) + "/LL2_TSmx_lim_b_o" + str(iobj) + self.data_str + '.npz')
                    else:
                        pp_file = np.load(self.data_dir +  str(iobj) + self.data_str + '.npz')
                    TSmx = pp_file['TSmx']

                    if halos_kept < 100: TSlim = self.TS100
                    elif halos_kept < 1000: TSlim = self.TS100
                    elif halos_kept < 10000: TSlim = self.TS100

                    if ((TSmx[0] > TSlim) & (TSmx[2] > self.xsecslim*best_lim[int(TSmx[1])])): 
                        continue    

                    # If specified check if it overlaps with previous halos
                    if self.nonoverlap:
                        obj_theta = np.pi/2-self.b_array[iobj]*np.pi/180.
                        obj_phi = self.ell_array[iobj]*np.pi/180.

                        if len(existing_theta) == 0: # At object 0
                            existing_theta = np.append(existing_theta,obj_theta)
                            existing_phi = np.append(existing_phi,obj_phi)
                        else:
                            rvals = np.arccos(np.cos(existing_theta)*np.cos(obj_theta)
                                              +np.sin(existing_theta)*np.sin(obj_theta)
                                              *np.cos(existing_phi-obj_phi))

                            if (np.min(rvals) < self.nonoverlapradius*np.pi/180.): 
                                continue # Skip, overlaps!

                            # Otherwise passed, keep its location for future overlap check
                            existing_theta = np.append(existing_theta,obj_theta)
                            existing_phi = np.append(existing_phi,obj_phi)


                    # Add this objects LL at the relevant mass to the cumulative LL 
                    # and then calculate the limit for this number of halos
                    LL2_cumulative += pp_file['LL2'][im]

                    TS_xsec_ary = LL2_cumulative - LL2_cumulative[0]

                    max_loc = np.argmax(TS_xsec_ary)
                    max_TS = TS_xsec_ary[max_loc]
                    
                    for xi in range(max_loc,len(xsecs)):
                        val = TS_xsec_ary[xi] - max_TS
                        if val < -2.71:
                            # Save log10 of the limit
                            scale = (TS_xsec_ary[xi-1]-max_TS+2.71)/(TS_xsec_ary[xi-1]-TS_xsec_ary[xi])
                            MC_limit_arr[halos_kept, im, imc] = np.log10(xsecs[xi-1])+scale*(np.log10(xsecs[xi])-np.log10(xsecs[xi-1]))
                            break
                    
                    xsec_at_maxTS_ary[halos_kept, im, imc] = xsecs[max_loc]

                    # Passed, add to counter
                    halos_kept += 1
                    
                    # If have enough halos stop, otherwise continue
                    if halos_kept == self.halos_to_keep: 
                        break

                halos_passed[imc] = halos_kept
            
            # Going to either specified halos to keep or minimum of passed halos
            halos_to_plot = int(min([min(halos_passed), self.halos_to_keep]))

        #################
        # Return arrays #
        #################

        # Return limit and maxTS arrays

        self.halos_that_passed = halos_passed
        return MC_limit_arr[:halos_to_plot], xsec_at_maxTS_ary[:halos_to_plot]

    def files_exist(self):
        """
        Check for missing files.
        Returns two arrays:
            imc_missing: missing MC files
            iobj_missing corresponding missing object files
        """
        iobj_missing = []
        imc_missing = []
        for iobj in tqdm_notebook(range(self.halos_ran)):
            for imc in range(self.nmc):
                if self.data_type in ["data", "Asimov"]:
                    self.data_str = "_" + self.data_type
                elif self.data_type == "mc":
                    self.data_str = "_mc" + str(imc)

                if not os.path.isfile(self.data_dir + str(iobj) + self.data_str + '.npz'):
                    iobj_missing.append(iobj)
                    imc_missing.append(imc)
        print len(iobj_missing), "files missing!"
        return imc_missing, iobj_missing
