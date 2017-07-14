# Code to make a full sky PS map from our individually smoothed point sources

import numpy as np

ps_map = np.zeros(shape=(40,196608))

for psi in range(3034):
    print "psi:",psi
    ps_file = np.load('/tigress/bsafdi/github/NPTF-working/NPTF-ID-Catalog/data/ps_data/ps_temp_128_5_0_'+str(psi)+'.npy')
    for E in range(40):
        ps_map[E][np.vectorize(int)(ps_file[::,E,0])] += ps_file[::,E,1] 

np.save('./ps_map.npy',ps_map)
