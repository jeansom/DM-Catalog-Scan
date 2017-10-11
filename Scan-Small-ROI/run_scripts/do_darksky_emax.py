import os, sys
import numpy as np

mcstart = 0
nmc = 50

halo_step=50
i_start=0
n_steps=10

# locs = ['200,200,200','100,100,100']
locs = ['200,200,200']

# for emax in [30,31,32,33,34,35]:
for emax in [30]:
    for iloc, loc in enumerate(locs):
        # if iloc == 0: continue
        for mci in range(mcstart,mcstart+nmc):
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
module load rh/devtoolset/4
module load intel-mpi/gcc/2017.2/64

cd  /tigress/lnecib/MultiNest
export LD_LIBRARY_PATH=$(pwd)/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

cd /tigress/nrodd/DM-Catalog-Scan/Scan-Small-ROI/
'''
                batch2 ='start_idx='+str(halo_start)+'\n'+'catalog_file=DarkSky_ALL_'+str(loc)+'_v3.csv'+'\n'
                batch3 = '''
echo "#!/bin/bash \necho i = \$1 \npython scan_interface.py --catalog_file $catalog_file --start_idx $start_idx --perform_scan 0 --imc ''' + str(mci)+ ''' --iobj \$1 --save_dir DarkSky_sid_loc''' + str(iloc) + ''' --float_ps_together 0 --Asimov 0 --restrict_pp 1 --floatDM 1 --emin 4 --emax ''' + str(emax) + '''" > run_scripts/conf/run-loc0-DS'''+str(it)+'''-v'''+str(mci)+str(loc)+'''-emax'''+str(emax)+'''.sh
chmod u+x run_scripts/conf/run-loc0-DS'''+str(it)+'''-v'''+str(mci)+str(loc)+'''-emax'''+str(emax)+'''.sh
'''
                runpart='echo   0-49  ./run_scripts/conf/run-loc0-DS'+str(it)+'-v'+str(mci)+str(loc)+'-emax'+str(emax)+'.sh %t  > run_scripts/conf/run-loc0-DS'+str(it)+'-v'+str(mci)+str(loc)+'-emax'+str(emax)+'.conf'+'\n'+'\n'+'srun --multi-prog --no-kill --wait=0 run_scripts/conf/run-loc0-DS'+str(it)+'-v'+str(mci)+str(loc)+'-emax'+str(emax)+'.conf'+'\n'+'\n'

                batchn = batch1+batch2 + batch3 + runpart
                fname = "./batch/run-loc0-DS"+str(it)+"-v"+str(mci)+str(loc)+"-emax"+str(emax)+".batch"
                f=open(fname, "w")
                f.write(batchn)
                f.close()
                os.system("sbatch "+fname);

                halo_start += halo_step
