#!/usr/bin/env python
import os 
import glob 

indir = '../video/toanalyse'
outdir = '../flows'

#Get files to run
files = glob.glob(indir + "/*.avi")

for fn in files:
	fnbase = os.path.split(fn)
	print fnbase
	cmd = './optical_flow_video ' + fn + ' ' + outdir + '/brox_' + fnbase[1]
	os.system(cmd)
