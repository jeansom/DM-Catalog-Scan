# This is code to test if there's a bug in the PSF smoothing

import numpy as np
import healpy as hp

#the_map = np.load('/tigress/smsharma/public/Jfactor_bartels1_truth_map_onlyobj0.npy')
nside=128
the_map = hp.ud_grade(np.load('/tigress/bsafdi/github/NPTF-working/NPTF-ID-Catalog/data/Jfactor_bartels1_truth_map.npy'),nside,power=-2) 

sigma = 1.0
print "Sigma =",sigma
the_smooth_map = hp.smoothing(the_map, sigma= sigma*2*np.pi/360)
where_vec = np.where(the_smooth_map < 0)[0]
the_smooth_map[where_vec] = 0.0
print "New/Old mean:",np.mean(the_smooth_map)/np.mean(the_map)

sigma = 0.5
print "Sigma =",sigma
the_smooth_map = hp.smoothing(the_map, sigma= sigma*2*np.pi/360)
where_vec = np.where(the_smooth_map < 0)[0]
the_smooth_map[where_vec] = 0.0
print "New/Old mean:",np.mean(the_smooth_map)/np.mean(the_map)

sigma = 0.4
print "Sigma =",sigma
the_smooth_map = hp.smoothing(the_map, sigma= sigma*2*np.pi/360)
where_vec = np.where(the_smooth_map < 0)[0]
the_smooth_map[where_vec] = 0.0
print "New/Old mean:",np.mean(the_smooth_map)/np.mean(the_map)

sigma = 0.3
print "Sigma =",sigma
the_smooth_map = hp.smoothing(the_map, sigma= sigma*2*np.pi/360)
where_vec = np.where(the_smooth_map < 0)[0]
the_smooth_map[where_vec] = 0.0
print "New/Old mean:",np.mean(the_smooth_map)/np.mean(the_map)

sigma = 0.2
print "Sigma =",sigma
the_smooth_map = hp.smoothing(the_map, sigma= sigma*2*np.pi/360)
where_vec = np.where(the_smooth_map < 0)[0]
the_smooth_map[where_vec] = 0.0
print "New/Old mean:",np.mean(the_smooth_map)/np.mean(the_map)

sigma = 0.19
print "Sigma =",sigma
the_smooth_map = hp.smoothing(the_map, sigma= sigma*2*np.pi/360)
where_vec = np.where(the_smooth_map < 0)[0]
the_smooth_map[where_vec] = 0.0
print "New/Old mean:",np.mean(the_smooth_map)/np.mean(the_map)

sigma = 0.18
print "Sigma =",sigma
the_smooth_map = hp.smoothing(the_map, sigma= sigma*2*np.pi/360)
where_vec = np.where(the_smooth_map < 0)[0]
the_smooth_map[where_vec] = 0.0
print "New/Old mean:",np.mean(the_smooth_map)/np.mean(the_map)

sigma = 0.17
print "Sigma =",sigma
the_smooth_map = hp.smoothing(the_map, sigma= sigma*2*np.pi/360)
where_vec = np.where(the_smooth_map < 0)[0]
the_smooth_map[where_vec] = 0.0
print "New/Old mean:",np.mean(the_smooth_map)/np.mean(the_map)

sigma = 0.16
print "Sigma =",sigma
the_smooth_map = hp.smoothing(the_map, sigma= sigma*2*np.pi/360)
where_vec = np.where(the_smooth_map < 0)[0]
the_smooth_map[where_vec] = 0.0
print "New/Old mean:",np.mean(the_smooth_map)/np.mean(the_map)

sigma = 0.15
print "Sigma =",sigma
the_smooth_map = hp.smoothing(the_map, sigma= sigma*2*np.pi/360)
where_vec = np.where(the_smooth_map < 0)[0]
the_smooth_map[where_vec] = 0.0
print "New/Old mean:",np.mean(the_smooth_map)/np.mean(the_map)

sigma = 0.14
print "Sigma =",sigma
the_smooth_map = hp.smoothing(the_map, sigma= sigma*2*np.pi/360)
where_vec = np.where(the_smooth_map < 0)[0]
the_smooth_map[where_vec] = 0.0
print "New/Old mean:",np.mean(the_smooth_map)/np.mean(the_map)

sigma = 0.13
print "Sigma =",sigma
the_smooth_map = hp.smoothing(the_map, sigma= sigma*2*np.pi/360)
where_vec = np.where(the_smooth_map < 0)[0]
the_smooth_map[where_vec] = 0.0
print "New/Old mean:",np.mean(the_smooth_map)/np.mean(the_map)

sigma = 0.12
print "Sigma =",sigma
the_smooth_map = hp.smoothing(the_map, sigma= sigma*2*np.pi/360)
where_vec = np.where(the_smooth_map < 0)[0]
the_smooth_map[where_vec] = 0.0
print "New/Old mean:",np.mean(the_smooth_map)/np.mean(the_map)

sigma = 0.11
print "Sigma =",sigma
the_smooth_map = hp.smoothing(the_map, sigma= sigma*2*np.pi/360)
where_vec = np.where(the_smooth_map < 0)[0]
the_smooth_map[where_vec] = 0.0
print "New/Old mean:",np.mean(the_smooth_map)/np.mean(the_map)

sigma = 0.1
print "Sigma =",sigma
the_smooth_map = hp.smoothing(the_map, sigma= sigma*2*np.pi/360)
where_vec = np.where(the_smooth_map < 0)[0]
the_smooth_map[where_vec] = 0.0
print "New/Old mean:",np.mean(the_smooth_map)/np.mean(the_map)

sigma = 0.05
print "Sigma =",sigma
the_smooth_map = hp.smoothing(the_map, sigma= sigma*2*np.pi/360)
where_vec = np.where(the_smooth_map < 0)[0]
the_smooth_map[where_vec] = 0.0
print "New/Old mean:",np.mean(the_smooth_map)/np.mean(the_map)

sigma = 0.01
print "Sigma =",sigma
the_smooth_map = hp.smoothing(the_map, sigma= sigma*2*np.pi/360)
where_vec = np.where(the_smooth_map < 0)[0]
the_smooth_map[where_vec] = 0.0
print "New/Old mean:",np.mean(the_smooth_map)/np.mean(the_map)
