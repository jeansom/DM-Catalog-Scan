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

for loc in ["100,100,100","100,100,300","100,300,100","300,100,100","300,300,100","100,300,300","300,100,300", "300,300,300", "200,200,200"]:
	box_center = [float(coord) for coord in loc.split(',')]
	for ebin in range(40):
		J_file = "Jfactor_DS_true_map_"+str(box_center[0])+"_"+str(box_center[1])+"_"+str(box_center[2])+"b1e+20_a1e+13_final"
		batchn = batch + str(ebin) + " --J_file " + J_file
		fname = "batch/batch_" + J_file + "_" + str(ebin) + ".batch" # 
		f=open(fname, "w")
		f.write(batchn)
		f.close()
		os.system("sbatch " + fname);