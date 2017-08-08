# Find max TS in the data

import numpy as np

TS_vals = np.zeros(100)

for i in range(len(TS_vals)):
    TS_vals[i] = np.load('/tigress/nrodd/DM-Catalog-Scan/Scan-Small-ROI/data/Tully/LL2_TSmx_lim_b_o'+str(i)+'_data.npz')['TSmx'][0]

print np.max(TS_vals)
print np.where(TS_vals == np.max(TS_vals))[0]
