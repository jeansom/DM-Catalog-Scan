# This is code to smooth the DS maps
# Note we need to exposure correct the map before smoothing, and then divide out

from global_variables import *
from global_var import *
import numpy as np
import healpy as hp
import fermi.fermi_plugin as fp
import king_smooth as ks

threads = 1 # TRY CHANGING THIS!

# Setup Fermi plugin
nside=128
eventclass=5 # 2 (Source) or 5 (UltracleanVeto)
eventtype=0 # 0 (all), 3 (bestpsf) or 5 (top3 quartiles)
<<<<<<< HEAD
emin_bin=0
emax_bin=40
=======
>>>>>>> 36349550a7726d617bf18f693d95dffe8dcccf9b

f_global = fp.fermi_plugin(maps_dir,fermi_data_dir=fermi_data_dir,work_dir=work_dir,CTB_en_min=emin_bin,CTB_en_max=emax_bin,nside=nside,eventclass=eventclass,eventtype=eventtype,newstyle=1,data_July16=True)

# Load the J-factor map and convert to human units

J_map = hp.ud_grade(np.load('/zfs/nrodd/NPTF-ID-Catalog/AccurateSmoothing/Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a1e+13_final.npy'),nside,power=-2)
J_map /= GeV**2*Centimeter**-5

# Exposure correct then smooth J_map energy bin by energy bin
<<<<<<< HEAD
J_map_arr_smoothed = np.zeros(shape=(emax_bin-emin_bin,len(J_map)))
for en in range(emax_bin-emin_bin):
    print "At energy bin " + str(en+1) + "/" + str(emax_bin-emin_bin)
    # Correct for exposure
    J_map_tmp_ps = J_map*f_global.CTB_exposure_maps[en]
    # Load an instance of the smoothing class
    ksi = ks.king_smooth(maps_dir,en,eventclass,eventtype,threads)
    # Smooth
    J_map_tmp_s = ksi.smooth_the_map(J_map_tmp_ps)
    # Remove the exposure map and save
    J_map_arr_smoothed[en] = J_map_tmp_s/f_global.CTB_exposure_maps[en]


np.save('/zfs/nrodd/NPTF-ID-Catalog/AccurateSmoothing/Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a1e+13_final_smoothed.npy',J_map_arr_smoothed)
=======
J_map_arr_smoothed = np.zeros(len(J_map))

# Correct for exposure
J_map_tmp_ps = J_map*f_global.CTB_exposure_maps[0]
# Load an instance of the smoothing class
ksi = ks.king_smooth(maps_dir,ebin,eventclass,eventtype,threads)
# Smooth
J_map_tmp_s = ksi.smooth_the_map(J_map_tmp_ps)
# Remove the exposure map and save
J_map_arr_smoothed = J_map_tmp_s/f_global.CTB_exposure_maps[0]
np.save('SmoothedMapsLocsSystematicsBinned/'+J_file+"_"+str(ebin)+'.npy',J_map_arr_smoothed)
>>>>>>> 36349550a7726d617bf18f693d95dffe8dcccf9b
