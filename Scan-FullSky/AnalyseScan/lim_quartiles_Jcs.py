# This is code to take a large set of limits and determine quartiles from them

import numpy as np

mcfiles = 100

output50 = np.zeros(shape=(53,2))

for mi in range(53):
    getmass = np.loadtxt('./Output/Jcm1_v0-lim.dat')
    output50[mi,0] = np.log10(getmass[mi,0])
    lim_ary = np.zeros(mcfiles)
    for mci in range(mcfiles):
        getlim = np.loadtxt('./Output/Jcm1_v'+str(mci)+'-lim.dat')
        lim_ary[mci] = getlim[mi,1]
    
    output50[mi,1] = np.log10(np.percentile(lim_ary, 50))

np.savetxt('./Output/Jcm1_q50-lim.dat',output50)

output50 = np.zeros(shape=(53,2))

for mi in range(53):
    getmass = np.loadtxt('./Output/Jcs1_v0-lim.dat')
    output50[mi,0] = np.log10(getmass[mi,0])
    lim_ary = np.zeros(mcfiles)
    for mci in range(mcfiles):
        getlim = np.loadtxt('./Output/Jcs1_v'+str(mci)+'-lim.dat')
        lim_ary[mci] = getlim[mi,1]

    output50[mi,1] = np.log10(np.percentile(lim_ary, 50))

np.savetxt('./Output/Jcs1_q50-lim.dat',output50)

output50 = np.zeros(shape=(53,2))

for mi in range(53):
    getmass = np.loadtxt('./Output/Jcs2_v0-lim.dat')
    output50[mi,0] = np.log10(getmass[mi,0])
    lim_ary = np.zeros(mcfiles)
    for mci in range(mcfiles):
        getlim = np.loadtxt('./Output/Jcs2_v'+str(mci)+'-lim.dat')
        lim_ary[mci] = getlim[mi,1]

    output50[mi,1] = np.log10(np.percentile(lim_ary, 50))

np.savetxt('./Output/Jcs2_q50-lim.dat',output50)

output50 = np.zeros(shape=(53,2))

for mi in range(53):
    getmass = np.loadtxt('./Output/Jcs3_v0-lim.dat')
    output50[mi,0] = np.log10(getmass[mi,0])
    lim_ary = np.zeros(mcfiles)
    for mci in range(mcfiles):
        getlim = np.loadtxt('./Output/Jcs3_v'+str(mci)+'-lim.dat')
        lim_ary[mci] = getlim[mi,1]

    output50[mi,1] = np.log10(np.percentile(lim_ary, 50))

np.savetxt('./Output/Jcs3_q50-lim.dat',output50)

output50 = np.zeros(shape=(53,2))

for mi in range(53):
    getmass = np.loadtxt('./Output/Jcs4_v0-lim.dat')
    output50[mi,0] = np.log10(getmass[mi,0])
    lim_ary = np.zeros(mcfiles)
    for mci in range(mcfiles):
        getlim = np.loadtxt('./Output/Jcs4_v'+str(mci)+'-lim.dat')
        lim_ary[mci] = getlim[mi,1]

    output50[mi,1] = np.log10(np.percentile(lim_ary, 50))

np.savetxt('./Output/Jcs4_q50-lim.dat',output50)

output50 = np.zeros(shape=(53,2))

for mi in range(53):
    getmass = np.loadtxt('./Output/Jcs5_v0-lim.dat')
    output50[mi,0] = np.log10(getmass[mi,0])
    lim_ary = np.zeros(mcfiles)
    for mci in range(mcfiles):
        getlim = np.loadtxt('./Output/Jcs5_v'+str(mci)+'-lim.dat')
        lim_ary[mci] = getlim[mi,1]

    output50[mi,1] = np.log10(np.percentile(lim_ary, 50))

np.savetxt('./Output/Jcs5_q50-lim.dat',output50)
