# This is code to take a large set of limits and determine quartiles from them

import numpy as np

mcfiles = 100

output16 = np.zeros(shape=(53,2))
output84 = np.zeros(shape=(53,2))

for mi in range(53):
    getmass = np.loadtxt('./Output/all_v0-lim.dat')
    output16[mi,0] = np.log10(getmass[mi,0])
    output84[mi,0] = np.log10(getmass[mi,0])
    lim_ary = np.zeros(mcfiles)
    for mci in range(mcfiles):
        getlim = np.loadtxt('./Output/all_v'+str(mci)+'-lim.dat')
        lim_ary[mci] = getlim[mi,1]
    
    output16[mi,1] = np.log10(np.percentile(lim_ary, 16))
    output84[mi,1] = np.log10(np.percentile(lim_ary, 84))

np.savetxt('./Output/all_q16-lim.dat',output16)
np.savetxt('./Output/all_q84-lim.dat',output84)
