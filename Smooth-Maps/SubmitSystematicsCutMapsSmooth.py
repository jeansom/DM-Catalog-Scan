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

for sys_str in ["ltom_cvir_correa","alpha2_ltom_cvir_dsmedian","mmin1e4_ltom_cvir_dsmedian","ltom_cvir_dsmedian","ltom_cvir_dsmedian","cvir_dsmedian"]:
	for ebin in range(40):
		J_file = "Jfactor_DS_inf_map_200.0_200.0_200.0b1e+20_a2e+17_" + sys_str
		batchn = batch + str(ebin) + " --J_file " + J_file
		fname = "batch/batch_" + J_file + "_" + str(ebin) + ".batch" # 
		f=open(fname, "w")
		f.write(batchn)
		f.close()
		os.system("sbatch " + fname);