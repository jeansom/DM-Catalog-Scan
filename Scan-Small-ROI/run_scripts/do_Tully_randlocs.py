import os, sys
import numpy as np

mcstart = 0
nmc = 100

halo_step=0
i_start=20
n_steps=1

loc=99

for mci in range(mcstart,mcstart+nmc):

    halo_start=i_start*halo_step

    for it in range(i_start,i_start+n_steps):

        batch1='''#!/bin/bash
#SBATCH -n 20   # node count
#SBATCH -t 1:34:00
#SBATCH --mem-per-cpu=4gb
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=smsharma@princeton.edu
##SBATCH -C ivy

export PATH="/tigress/smsharma/anaconda2/bin:$PATH"
source activate venv_py27
module load rh/devtoolset/4
module load intel-mpi/gcc/2017.2/64

cd  /tigress/lnecib/MultiNest
export LD_LIBRARY_PATH=$(pwd)/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

cd /tigress/nrodd/DM-Catalog-Scan/Scan-Small-ROI/
'''
        batch2 ='start_idx='+str(halo_start)+'\n'+'catalog_file=fake_cat_angext_v1.csv'+'\n'
        batch3 = '''
echo "#!/bin/bash \necho i = \$1 \npython scan_interface.py --catalog_file $catalog_file --start_idx $start_idx --randlocs 1 --perform_scan 1 --perform_postprocessing 0 --imc -1 --iobj \$1 --save_dir Tully_debug_randlocs''' + str(mci) + ''' --float_ps_together 0 --Asimov 0 --floatDM 1" > run_scripts/conf/run-Tully'''+str(it)+'''-v'''+str(mci)+str(loc)+'''.sh
chmod u+x run_scripts/conf/run-Tully'''+str(it)+'''-v'''+str(mci)+str(loc)+'''.sh
'''
        runpart='echo   0-19  ./run_scripts/conf/run-Tully'+str(it)+'-v'+str(mci)+str(loc)+'.sh %t  > run_scripts/conf/run-Tully'+str(it)+'-v'+str(mci)+str(loc)+'.conf'+'\n'+'\n'+'srun --multi-prog --no-kill --wait=0 run_scripts/conf/run-Tully'+str(it)+'-v'+str(mci)+str(loc)+'.conf'+'\n'+'\n'

        batchn = batch1+batch2 + batch3 + runpart
        fname = "./batch/run-Tully"+str(it)+"-v"+str(mci)+str(loc)+".batch"
        f=open(fname, "w")
        f.write(batchn)
        f.close()
        os.system("sbatch "+fname);

        halo_start += halo_step
