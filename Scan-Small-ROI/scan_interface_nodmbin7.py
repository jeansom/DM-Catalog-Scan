import argparse

from scan_nodmbin7 import Scan
from local_dirs import *

parser = argparse.ArgumentParser()
parser.add_argument("--perform_scan",
                  action="store", dest="perform_scan", default=1,type=int)
parser.add_argument("--perform_postprocessing",
                  action="store", dest="perform_postprocessing", default=1,type=int)
parser.add_argument("--imc",
                  action="store", dest="imc", default=0,type=int)
parser.add_argument("--iobj",
                  action="store", dest="iobj", default=0,type=int)
parser.add_argument("--Asimov",
                  action="store", dest="Asimov", default=0,type=int)
parser.add_argument("--save_dir",
                  action="store", dest="save_dir", default="",type=str)
parser.add_argument("--load_dir",
                  action="store", dest="load_dir", default="",type=str)
parser.add_argument("--float_ps_together",
                  action="store", dest="float_ps_together", default=1,type=int)
parser.add_argument("--noJprof",
                  action="store", dest="noJprof", default=0,type=int)
parser.add_argument("--start_idx",
                  action="store", dest="start_idx", default=0,type=int)
parser.add_argument("--floatDM",
                  action="store", dest="floatDM", default=0,type=int)
parser.add_argument("--mc_dm",
                  action="store", dest="mc_dm", default=-1,type=int)
parser.add_argument("--catalog_file",
                  action="store", dest="catalog_file", default="DarkSky_ALL_200,200,200_v3.csv",type=str)

results = parser.parse_args()
iobj=results.iobj
imc=results.imc
Asimov=results.Asimov
save_dir=results.save_dir
load_dir=results.load_dir
float_ps_together=results.float_ps_together
noJprof=results.noJprof
start_idx=results.start_idx
floatDM=results.floatDM
perform_scan=results.perform_scan
perform_postprocessing=results.perform_postprocessing
mc_dm=results.mc_dm
catalog_file=results.catalog_file

if load_dir != "":
  load_dir = work_dir + '/Scan-Small-ROI/data/' + str(load_dir) + "/"
else:
  load_dir = None

Scan(perform_scan=perform_scan, 
  perform_postprocessing=perform_postprocessing, 
  imc=imc, 
  iobj=start_idx+iobj, 
  Asimov=Asimov, 
  float_ps_together=float_ps_together,
  noJprof=noJprof,
  floatDM=floatDM,
  mc_dm=mc_dm,
  load_dir=load_dir,
  verbose=True,
  catalog_file=catalog_file, 
  save_dir=work_dir + '/Scan-Small-ROI/data/' + str(save_dir) + "/")
