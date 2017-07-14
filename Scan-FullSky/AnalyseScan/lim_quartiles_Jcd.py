# This is code to take a large set of limits and determine quartiles from them

import numpy as np

mcfiles = 100

cutvals = np.linspace(13,18.5,15)[::-1]
cutvals = cutvals[0:14]
output16 = np.zeros(shape=(14,2))
output84 = np.zeros(shape=(14,2))

for ci in range(14):
    output16[ci,0] = cutvals[ci]
    output84[ci,0] = cutvals[ci]
    lim_ary = np.zeros(mcfiles)
    for mci in range(mcfiles):
        getlim = np.loadtxt('./Output/Jcd'+str(ci+1)+'_v'+str(mci)+'-lim.dat')
        lim_ary[mci] = getlim[1] # Pull out limit at 100 GeV
    
    output16[ci,1] = np.log10(np.percentile(lim_ary, 16))
    output84[ci,1] = np.log10(np.percentile(lim_ary, 84))

np.savetxt('./Output/Jcd_q16-lim.dat',output16)
np.savetxt('./Output/Jcd_q84-lim.dat',output84)
