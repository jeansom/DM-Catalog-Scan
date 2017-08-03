import os, sys
import numpy as np

nmc = 200

for imc in range(200):


    batch1='''#!/bin/bash
#SBATCH -n 1   # node count
#SBATCH -t 1:34:00
#SBATCH --mem-per-cpu=4gb
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=smsharma@princeton.edu
##SBATCH -C ivy

export PATH="/tigress/smsharma/anaconda2_della5/bin:$PATH"
source activate venv_py27
module load intel-mpi/intel/5.1.3/64

cd  /tigress/lnecib/MultiNest
export LD_LIBRARY_PATH=$(pwd)/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

cd /tigress/nrodd/DM-Catalog-Scan/Scan-Small-ROI/run_scripts
python make_cats.py --imc '''
    batch2 =str(imc)+'\n'
    batchn = batch1+batch2
    fname = "./batch/run-DS-della5"+str(imc)+"-v"+str(imc)+str(imc)+".batch"
    f=open(fname, "w")
    f.write(batchn)
    f.close()
    os.system("sbatch "+fname);
