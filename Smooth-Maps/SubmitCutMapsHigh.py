"""
Code to generate large amount of parton-level
information in MadGraph and pipe it through a
Pythia script.
"""

import sys, os
import random
from math import log10, floor
import numpy as np

batch='''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 2:00:00
#SBATCH --mem=4GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
#SBATCH --mail-user=smsharma@princeton.edu
#SBATCH -p hepheno

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/group/hepheno/heptools/MultiNest/lib/

cd /group/hepheno/smsharma/Fermi-LSS/AccurateSmoothing
source activate venv_py27

python smooth_DS_maps.py --ebin '''

rtn = lambda x, n: round(x, -int(floor(log10(x))) + (n - 1))

for jcutdown in np.logspace(18,18.5,10):
	for ebin in range(40):
		J_file = "Jfactor_DS_true_map_200.0_200.0_200.0b1e+20_a"+str(rtn(jcutdown,3))+"_final"
		batchn = batch + str(ebin) + " --J_file " + J_file
		fname = "batch/batch_" + J_file + "_" + str(ebin) + ".batch" # 
		f=open(fname, "w")
		f.write(batchn)
		f.close()
		os.system("sbatch " + fname);

# for jcutup in np.logspace(13,18.5,15):
# 	for ebin in range(40):
# 		J_file = "Jfactor_DS_true_map_200.0_200.0_200.0b"+str(rtn(jcutup,3))+"_a1e+13_final"
# 		batchn = batch + str(ebin) + " --J_file " + J_file
# 		fname = "batch/batch_" + J_file + "_" + str(ebin) + ".batch" # 
# 		f=open(fname, "w")
# 		f.write(batchn)
# 		f.close()
# 		os.system("sbatch " + fname);
