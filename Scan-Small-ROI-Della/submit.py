import sys, os
import random
import numpy as np

batch='''#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 1:04:00
#SBATCH --mem=4GB
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=smsharma@princeton.edu
##SBATCH -C ivy

export PATH="/tigress/smsharma/anaconda2/bin:$PATH"
source activate venv_py27

cd  /tigress/lnecib/MultiNest
export LD_LIBRARY_PATH=$(pwd)/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

cd /tigress/bsafdi/github/NPTF-working/NPTF-ID-Catalog/SimpleScan/
'''

obj_missing = np.load("obj_missing.npy")
imc_missing = np.load("imc_missing.npy")

# for iobj in obj_missing:
# 	for imc in imc_missing:
for iobj,imc in zip(obj_missing,imc_missing):
		batchn = batch  + "\n"
		batchn += "python scan_interface.py --imc " + str(imc) + " --iobj " + str(iobj) + " --save_dir FloatPS_together_noDM --float_ps_together 1 --Asimov 0 --floatDM 0"
		fname = "batch/" + str(imc) + "_" + str(iobj) + ".batch" 
		f=open(fname, "w")
		f.write(batchn)
		f.close()
		os.system("chmod +x " + fname);
		os.system("sbatch " + fname);