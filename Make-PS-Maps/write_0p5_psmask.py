# Make a mask of pixels within 0.5 degrees of a PS

import numpy as np
import healpy as hp
import pandas as pd

mdist = 0.5*np.pi/180.

def dist_phi(phi1,phi2):
    # Determine the shortest distance in phi accounting for the periodicity
    return np.minimum(np.abs(phi1-phi2), 2*np.pi-np.abs((phi1-phi2)))

def dist_sphere(theta1,theta2,phi1,phi2):
    # This is a function to calculate the angular distance on a sphere, assuming that
    # the two points are close
    # For details (and the full formula): https://en.wikipedia.org/wiki/Great-circle_distance
    return np.sqrt( (theta1 - theta2)**2 + dist_phi(phi1,phi2)**2*(np.sin((theta1+theta2)/2.))**2 )


# Get the 3FGL phi and theta
csv_file = '/tigress/smsharma/public/fluxes_3fgl_binned.csv'
FGL = pd.read_csv(csv_file)

coords = np.zeros(shape=(3034,2))

phivals = np.zeros(3034)
thetavals = np.zeros(3034)

for i in range(3034):
    phivals[i] = FGL['l'].values[i]*np.pi/180. 
    thetavals[i] = np.pi/2-FGL['b'].values[i]*np.pi/180.

# Now fill up a map
nside = 128
npix = hp.nside2npix(nside)

mask = np.zeros(npix, dtype=np.int32)

for i in range(npix):
    if ((i+1) % 1000 == 0):
        print "At pixel:",str(i+1) + "/" + str(npix)
    maskpix = False
    ptheta, pphi = hp.pix2ang(nside, i)
    dist_3FGL = np.zeros(3034)
    for j in range(3034):
        dist_3FGL[j] = dist_sphere(ptheta, thetavals[j], pphi, phivals[j])

    if (np.min(dist_3FGL) <= mdist):
        mask[i] = 1

np.save('./mask0p5_3FGL.npy',mask)
