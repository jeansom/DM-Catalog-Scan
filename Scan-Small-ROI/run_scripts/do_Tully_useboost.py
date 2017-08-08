import os, sys
import numpy as np

mcstart = 0
nmc = 1

halo_step=50
i_start=0
n_steps=20

for mci in [-1]:

    halo_start=i_start*halo_step

    for it in range(i_start,n_steps):

        batch1='''#!/bin/bash
#SBATCH -n 50   # node count
#SBATCH -t 2:04:00
#SBATCH --mem-per-cpu=4gb
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=smsharma@princeton.edu
#SBATCH -C ivy

export PATH="/tigress/smsharma/anaconda2/bin:$PATH"
source activate # venv_py27
# module load openmpi/gcc/1.6.5/64
module load rh/devtoolset/4
module load intel-mpi/intel/2017.2/64

cd  /tigress/lnecib/MultiNest
export LD_LIBRARY_PATH=$(pwd)/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

cd /tigress/nrodd/DM-Catalog-Scan/Scan-Small-ROI/
'''
        batch2 ='start_idx='+str(halo_start)+'\n'+'catalog_file=2MRSLocalTully_ALL_DATAPAPER_Planck15_v7.csv'+'\n'
        batch3 = '''
echo "#!/bin/bash \necho i = \$1 \npython scan_interface.py --catalog_file $catalog_file --diff p8 --start_idx $start_idx --perform_scan 0 --imc ''' + str(mci) + ''' --use_boost 1 --iobj \$1 --save_dir Tully_useboost --float_ps_together 0 --Asimov 0 --floatDM 1 --restrict_pp 1 --emin 4" > ./run_scripts/conf/run-Data-ecut4.g-indiv-useboost-'''+str(it)+'''-v'''+str(mci)+'''.sh
chmod u+x ./run_scripts/conf/run-Data-ecut4.g-indiv-useboost-'''+str(it)+'''-v'''+str(mci)+'''.sh
'''
        runpart='echo   0-49  ./run_scripts/conf/run-Data-ecut4.g-indiv-useboost-'+str(it)+'-v'+str(mci)+'.sh %t  > ./run_scripts/conf/run-Data-indiv-useboost-'+str(it)+'-v'+str(mci)+'.conf'+'\n'+'\n'+'srun --multi-prog --no-kill --wait=0 ./run_scripts/conf/run-Data-indiv-useboost-'+str(it)+'-v'+str(mci)+'.conf'+'\n'+'\n'

        batchn = batch1+batch2 + batch3 + runpart
        fname = "./batch/run-Data-ecut4.g-indiv-useboost-"+str(it)+"-v"+str(mci)+".batch"
        f=open(fname, "w")
        f.write(batchn)
        f.close()
        os.system("sbatch "+fname);

        halo_start += halo_step
