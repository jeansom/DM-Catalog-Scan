import sys
import shutil
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import *
from plot_skylocs import LimitPlot
from astropy.cosmology import Planck15
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--imc",
                  action="store", dest="imc", default=0,type=int)

results = parser.parse_args()
imc = results.imc

catalog = pd.read_csv("/tigress/nrodd/DM-Catalog-Scan/DataFiles/Catalogs/2MRSTully_ALL_DATAPAPER_Planck15_v4.csv")

nobj = 1000

lb_cat = []

for iobj in tqdm(range(nobj)):
    rep_angext = np.array([0.02785567,0.12069876,0.21354185,0.30638494,0.39922802,0.49207111,0.5849142,0.67775728,0.77060037,0.86344346,0.95628654,1.04912963,1.14197272,1.2348158,1.32765889,1.42050198,1.51334507,1.60618815,1.69903124,1.79187433])
    obj_angext = 2*catalog[u'rs'].values[iobj] / \
                 (Planck15.angular_diameter_distance(catalog[u'z'].values[iobj]).value*1000) \
                 * 180./np.pi
    rep_index = (np.abs(rep_angext-obj_angext)).argmin()

    lb_cat.append(np.loadtxt("../data/Tully_randlocs"+str(int(np.loadtxt("../data/Tully_skylocs"+str(imc)+"/skyloc_obj"+str(iobj)+".txt")))+"/lb_obj"+str(rep_index)+".dat"))

np.save("../data/Tully_skylocs/lb_cat_mc"+str(imc)+".npy",lb_cat)