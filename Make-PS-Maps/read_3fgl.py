# Code to read in Sids J-factors for the various halos

import numpy as np
import pandas as pd

csv_file = '/tigress/nrodd/NPTF-ID-Catalog/LL_prof_indiv/3FGL/fluxes_3fgl_binned.csv'
csv_file = '/tigress/smsharma/public/fluxes_3fgl_binned.csv'
FGL = pd.read_csv(csv_file)

coords = np.zeros(shape=(3034,2))
fluxvals = np.zeros(shape=(3034,40))

for i in range(3034):
    lval = FGL['l'].values[i]
    coords[i,0] = (((lval + 180) % 360) - 180)
    coords[i,1] = FGL['b'].values[i]
    for j in range(40):
        fluxvals[i,j] = FGL[str(j)].values[i]

np.save('3FGL_coords.npy',coords)
np.save('3FGL_fluxes.npy',fluxvals)
