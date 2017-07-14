# This is code to take a large set of lixits and deterxine quartiles from them

import numpy as np

mcfiles = 100

xvals = np.linspace(-21,-28,15)

output16 = np.zeros(shape=(15,2))
output84 = np.zeros(shape=(15,2))

for xi in range(15):
    output16[xi,0] = xvals[xi]
    output84[xi,0] = xvals[xi]
    lim_ary = np.zeros(mcfiles)
    for mci in range(mcfiles):
        getlim = np.loadtxt('./Output/Jcis'+str(xi+1)+'_v'+str(mci)+'-lim.dat')
        lim_ary[mci] = getlim[1]
    
    output16[xi,1] = np.log10(np.percentile(lim_ary, 16))
    output84[xi,1] = np.log10(np.percentile(lim_ary, 84))

np.savetxt('./Output/Jcis_q16-lim.dat',output16)
np.savetxt('./Output/Jcis_q84-lim.dat',output84)
