import os, sys
import numpy as np

mcstart = 0
nmc = 5

halo_step=50
i_start=0
n_steps=20

for mci in range(mcstart,mcstart+nmc):
    for imc_dm in [1,3,5,7,9]:

        halo_start=i_start*halo_step

        for it in range(i_start,i_start+n_steps):

            batch1='''#!/bin/bash
#SBATCH -n 50   # node count
#SBATCH -t 1:34:00
#SBATCH --mem-per-cpu=4gb
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=smsharma@princeton.edu
#SBATCH -C ivy

export PATH="/tigress/smsharma/anaconda2/bin:$PATH"
source activate # venv_py27
module load openmpi/gcc/1.6.5/64
module load rh/devtoolset/4
# module load intel-mpi/intel/2017.2/64

cd  /tigress/lnecib/MultiNest
export LD_LIBRARY_PATH=$(pwd)/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

cd /tigress/nrodd/DM-Catalog-Scan/Scan-Small-ROI/
'''
            batch2 ='start_idx='+str(halo_start)+'\n'+'catalog_file=2MRSLocalTully_ALL_DATAPAPER_Planck15_v7.csv'+'\n'
            batch3 = '''
echo "#!/bin/bash \necho i = \$1 \npython scan_interface.py --catalog_file $catalog_file --mc_string 10000dFm --mc_dm '''+str(imc_dm)+''' --start_idx $start_idx --perform_scan 1 --imc ''' + str(mci) + ''' --iobj \$1 --save_dir Tully_injsig10000'''+str(imc_dm)+''' --float_ps_together 0 --Asimov 0 --floatDM 1" > run_scripts/conf/run-Tully-inj10000-'''+str(it)+'''-v'''+str(mci)+'''-dm'''+str(imc_dm)+ '''.sh
chmod u+x run_scripts/conf/run-Tully-inj10000-'''+str(it)+'''-v'''+str(mci)+'-dm'+str(imc_dm)+'''.sh
'''
            runpart='echo   0-49  ./run_scripts/conf/run-Tully-inj10000-'+str(it)+'-v'+str(mci)+'-dm'+str(imc_dm)+ '.sh %t  > run_scripts/conf/run-Tully-inj10000-'+str(it)+'-v'+str(mci)+'-dm'+str(imc_dm)+ '.conf'+'\n'+'\n'+'srun --multi-prog --no-kill --wait=0 run_scripts/conf/run-Tully-inj10000-'+str(it)+'-v'+str(mci)+'-dm'+str(imc_dm)+ '.conf'+'\n'+'\n'

            batchn = batch1+batch2 + batch3 + runpart
            fname = "./batch/run-Tully-inj10000-"+str(it)+"-v"+str(mci)+'-dm'+str(imc_dm)+ ".batch"
            f=open(fname, "w")
            f.write(batchn)
            f.close()
            os.system("sbatch "+fname);

            halo_start += halo_step
