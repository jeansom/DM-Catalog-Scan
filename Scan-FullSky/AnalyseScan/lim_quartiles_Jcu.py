# This is code to take a large set of limits and determine quartiles from them

import numpy as np

mcfiles = 100

cutvals = np.linspace(13,18.5,15)[1:13]
cutvals = np.append(cutvals,[18.0,18.1105897103])
output16 = np.zeros(shape=(14,2))
output84 = np.zeros(shape=(14,2))

for ci in range(14):
    output16[ci,0] = cutvals[ci]
    output84[ci,0] = cutvals[ci]
    lim_ary = np.zeros(mcfiles)
    for mci in range(mcfiles):
        if ci < 12:
            getlim = np.loadtxt('./Output/Jcu'+str(ci+1)+'_v'+str(mci)+'-lim.dat')
        else:
            getlim = np.loadtxt('./Output/Jcu'+str(ci+2)+'_v'+str(mci)+'-lim.dat')
        lim_ary[mci] = getlim[1] # Pull out limit at 100 GeV
    
    output16[ci,1] = np.log10(np.percentile(lim_ary, 16))
    output84[ci,1] = np.log10(np.percentile(lim_ary, 84))

np.savetxt('./Output/Jcu_q16-lim.dat',output16)
np.savetxt('./Output/Jcu_q84-lim.dat',output84)
