print("Starting import...")

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

sys.path.append('/tigress/nrodd/DM-Catalog-Scan/Scan-Small-ROI-Della')

from local_dirs import *
from minuit_functions import call_ll

# Additional modules
sys.path.append(nptf_old_dir)
sys.path.append('/tigress/nrodd/DM-Catalog-Scan/Scan-Small-ROI-Della/Smooth-Maps') # Different dir because recompile smooth king
sys.path.append(work_dir + '/Make-DM-Maps')
import fermi.fermi_plugin as fp
import mkDMMaps
import king_smooth as ks
import LL_inten_to_xsec as Litx

# NPTFit modules
from NPTFit import nptfit # module for performing scan
from NPTFit import create_mask as cm # module for creating the mask


print("...done!")
