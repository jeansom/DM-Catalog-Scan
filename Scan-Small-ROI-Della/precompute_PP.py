###############################################################################
# simple_scan.py
###############################################################################
#
# Code to precompute the particle physics factor
#
###############################################################################

import numpy as np
import pandas as pd
from scipy import interpolate, integrate
from local_dirs import *

channel = 'b'

ebins = 2*np.logspace(-1,3,41)

dNdLogx_df = pd.read_csv(additional_data_dir + 'AtProduction_gammas.dat', delim_whitespace=True)

m_ary = np.array([1.00000000e+01,1.50000000e+01,2.00000000e+01,2.50000000e+01,3.00000000e+01,4.00000000e+01,5.00000000e+01,6.00000000e+01,7.00000000e+01,8.00000000e+01,9.00000000e+01,1.00000000e+02,1.10000000e+02,1.20000000e+02,1.30000000e+02,1.40000000e+02,1.50000000e+02,1.60000000e+02,1.80000000e+02,2.00000000e+02,2.20000000e+02,2.40000000e+02,2.60000000e+02,2.80000000e+02,3.00000000e+02,3.30000000e+02,3.60000000e+02,4.00000000e+02,4.50000000e+02,5.00000000e+02,5.50000000e+02,6.00000000e+02,6.50000000e+02,7.00000000e+02,7.50000000e+02,8.00000000e+02,9.00000000e+02,1.00000000e+03,1.10000000e+03,1.20000000e+03,1.30000000e+03,1.50000000e+03,1.70000000e+03,2.00000000e+03,2.50000000e+03,3.00000000e+03,4.00000000e+03,5.00000000e+03,6.00000000e+03,7.00000000e+03,8.00000000e+03,9.00000000e+03,1.00000000e+04])

PPnoxsec_ary = np.zeros(shape=(len(m_ary),len(ebins)-1))
for mi in range(len(m_ary)):
    dNdLogx_ann_df = dNdLogx_df.query('mDM == ' + (str(np.int(float(m_ary[mi])))))[['Log[10,x]',channel]]
    Egamma = np.array(m_ary[mi]*(10**dNdLogx_ann_df['Log[10,x]']))
    dNdEgamma = np.array(dNdLogx_ann_df[channel]/(Egamma*np.log(10)))
    dNdE_interp = interpolate.interp1d(Egamma, dNdEgamma)
    for ei in range(len(ebins)-1): # -1 because ebins-1 bins, ebins edges
        # Only have flux if m > Ebin
        if ebins[ei] < m_ary[mi]:
            if ebins[ei+1] < m_ary[mi]:
                # Whole bin is inside
                PPnoxsec_ary[mi,ei] = 1.0/(8*np.pi*m_ary[mi]**2)*integrate.quad(lambda x: dNdE_interp(x), ebins[ei], ebins[ei+1])[0]
            else:
                # Bin only partially contained
                PPnoxsec_ary[mi,ei] = 1.0/(8*np.pi*m_ary[mi]**2)*integrate.quad(lambda x: dNdE_interp(x), ebins[ei], m_ary[mi])[0]

np.save(additional_data_dir + 'PPnoxsec_' + channel + '_ary', PPnoxsec_ary)
