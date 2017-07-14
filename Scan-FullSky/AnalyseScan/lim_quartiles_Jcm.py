# This is code to take a large set of limits and determine quartiles from them

import numpy as np

mcfiles = 100

output16 = np.zeros(shape=(53,2))
output84 = np.zeros(shape=(53,2))

for mi in range(53):
    getmass = np.loadtxt('./Output/Jcm1_v0-lim.dat')
    output16[mi,0] = np.log10(getmass[mi,0])
    output84[mi,0] = np.log10(getmass[mi,0])
    move_ary = np.zeros(9)
    for i in range(9):
        lim_ary = np.zeros(mcfiles)
        for mci in range(mcfiles):
            getlim = np.loadtxt('./Output/Jcm'+str(i+1)+'_v'+str(mci)+'-lim.dat')
            lim_ary[mci] = getlim[mi,1]
        move_ary[i] = np.log10(np.percentile(lim_ary, 50))
    
    output16[mi,1] = np.min(move_ary)
    output84[mi,1] = np.max(move_ary)

np.savetxt('./Output/Jcm_min-lim.dat',output16)
np.savetxt('./Output/Jcm_max-lim.dat',output84)
