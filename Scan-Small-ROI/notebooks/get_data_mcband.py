import sys
import shutil
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import *
from plot_skylocs import LimitPlot
from astropy.cosmology import Planck15
import corner

catalog = pd.read_csv("/tigress/nrodd/DM-Catalog-Scan/DataFiles/Catalogs/2MRSLocalTully_ALL_DATAPAPER_Planck15_v7.csv")

plot_data_skylocs = LimitPlot(data_dir='/tigress/nrodd/DM-Catalog-Scan/Scan-Small-ROI/data/Tully_skylocs_no0p5mask_emin4/',
                        elephant=False, 
                        nmc=200,
                        bcut=20,
                        nonoverlap=True,
                        nonoverlapradius=2.,
                        xsecslim=10,
                        TS100=9,
                        TS1000=9,
                        halos_ran=1000, 
                        halos_to_keep=1000,
                        file_prefix='LL2_TSmx_lim_b_emin4_o',
                        data_type="skylocs",
                        skip_halos=[0],
                        custom_good_vals=np.load("included_halos.npy"))

data_skylocs_ary, _, _ = plot_data_skylocs.return_limits()
np.save("data_skylocs_ary_ecut4_200",data_skylocs_ary)
#data_skylocs_ary = np.load("data_skylocs_ary_ecut4.npy")