# This is code to take a large set of lixits and deterxine quartiles from them

import numpy as np

mcfiles = 100

xvals = np.linspace(-21,-28,15)

output = np.zeros(shape=(15,5))

for xi in range(15):
    output[xi,0] = xvals[xi]
    
    lim_ary = np.zeros(mcfiles)
    for mci in range(mcfiles):
        getlim = np.loadtxt('./Output/Jcis'+str(xi+1)+'_v'+str(mci)+'_xs.dat')
        lim_ary[mci] = getlim[1] 
    output[xi,1] = np.percentile(lim_ary, 50)

    lim_ary = np.zeros(mcfiles)
    for mci in range(mcfiles):
        getlim = np.loadtxt('./Output/Jcis'+str(xi+1)+'_v'+str(mci)+'_xp.dat')
        lim_ary[mci] = getlim[1]
    output[xi,2] = np.percentile(lim_ary, 50)

    lim_ary = np.zeros(mcfiles)
    for mci in range(mcfiles):
        getlim = np.loadtxt('./Output/Jcis'+str(xi+1)+'_v'+str(mci)+'_xm.dat')
        lim_ary[mci] = getlim[1]
    output[xi,3] = np.percentile(lim_ary, 50)

    lim_ary = np.zeros(mcfiles)
    for mci in range(mcfiles):
        getlim = np.loadtxt('./Output/Jcis'+str(xi+1)+'_v'+str(mci)+'_xts.dat')
        lim_ary[mci] = getlim[1]
    output[xi,4] = np.percentile(lim_ary, 50)

np.savetxt('./Output/Jcis_x.dat',output)
