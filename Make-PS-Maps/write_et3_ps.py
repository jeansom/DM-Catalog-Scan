# Code to write out all point sources as a 2D array, where 1st value is the index in an nside=128 where it is, and the second is the value of the PS

import numpy as np
import healpy as hp
import make_LL_prof as mlp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

eventtype=3
nside = 128
rad_norm = 20.
ell = 0.
b = 0.
mask_type = 'top300'
maps_dir='/tigress/smsharma/public/CTBCORE/'
fermi_data_dir='/tigress/smsharma/public/FermiData/'

# Import the PS template
ps_temp = np.load('/tigress/nrodd/NPTF-ID-Catalog/LL_prof_indiv/3FGL/ps_temp_5_'+str(eventtype)+'.npy')

# Determine which PSs need to be added for the norm and scan runs
coord_3FGL = np.load('/tigress/nrodd/NPTF-ID-Catalog/LL_prof_indiv/3FGL/3FGL_coords.npy')
flux_3FGL = np.load('/tigress/nrodd/NPTF-ID-Catalog/LL_prof_indiv/3FGL/3FGL_fluxes.npy')

npix = hp.nside2npix(nside)
exp_maps = np.zeros(shape=(40,npix))
for E in range(40):
    data = mlp.load_data(E,eventclass=5,eventtype=3,fermi_data_dir=fermi_data_dir,maps_dir=maps_dir,nside=nside,mask_type=mask_type)
    exp_maps[E] = data.exposure

# Set up theta/phi array for later
npix = hp.nside2npix(nside)
theta_arr, phi_arr = hp.pix2ang(nside,range(npix))

for psi in range(0,3034):
    print "At point source",psi,"out of",len(flux_3FGL)
    psl = coord_3FGL[psi,0]
    psb = coord_3FGL[psi,1]
    npixupbin = hp.nside2npix(1024)
    pix_num = hp.ang2pix(1024,np.pi/2.,0.)
    thetarot, phirot = hp.pix2ang(1024, np.arange(npixupbin))
    r = hp.Rotator(rot = [psl*np.pi/180,psb*np.pi/180], coord=None, inv=False, deg=False, eulertype='ZYX')
    thetarot, phirot = r(thetarot, phirot)
    pixrot = hp.ang2pix(1024, thetarot, phirot, nest=False)
    
    # Determine pixels within 10 degrees
    theta = np.pi/2. - psb*np.pi/180.
    phi = (psl % 360)*np.pi/180.
    rval = np.arccos(np.cos(theta)*np.cos(theta_arr)+np.sin(theta)*np.sin(theta_arr)*np.cos(phi-phi_arr))
    roi = np.where(rval <= 10.0*np.pi/180.)[0]
    
    # Now create maps
    ps_out = np.zeros(shape=(len(roi),40,2))
    for E in range(40):
        ps_temp_E = ps_temp[E]
        ps_rot_temp = ps_temp_E[..., pixrot]
        ps_rot_temp = hp.ud_grade(ps_rot_temp,nside,power=-2)
        ps_rot_temp = ps_rot_temp/np.sum(ps_rot_temp)

        # Adjust for flux
        ps_position = hp.ang2pix(nside, theta, phi)
        ps_rot_temp *= flux_3FGL[psi][E]*exp_maps[E][ps_position]
        
        for i in range(len(roi)):
            ps_out[i,E,0] = roi[i]
            ps_out[i,E,1] = ps_rot_temp[roi[i]]

    np.save('/tigress/bsafdi/github/NPTF-working/NPTF-ID-Catalog/data/ps_data/ps_temp_'+str(nside)+'_5_3_'+str(psi)+'.npy',ps_out)
