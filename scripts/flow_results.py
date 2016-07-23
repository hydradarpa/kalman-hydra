import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt

import os.path 
import sys
import numpy as np

import numpy.random as rd 

import seaborn as sns
import pandas as pd 

"""plot_flow.py <path>

Plot RMS of all methods on all datasets
"""

path = './synthetictests/flowresults/'
img_out = './synthetictests/flowresults/flow_RMSs'

np.set_printoptions(threshold = 'nan', linewidth = 150, precision = 3)

rms = {}

flowerrs = {}

for root, dirs, files in os.walk(path):
	for f in files:
		if f.endswith(".pkl"):
			p = os.path.join(root, f)
			print(p)
			#./synthetictests/flowresults/square1/rotate_testflow_solver_it_20/flow_errors.pkl
			s = p.split('/')
			geom = s[3]
			aa = s[4].split('_testflow')
			warp = aa[0]
			if warp.endswith(".pkl"):
				continue 
			if len(aa) > 1:
				notes = aa[1]
			else:
				notes = ''
			flowerr =  pd.read_pickle(p)
			if warp not in flowerrs:
				flowerrs[warp] = {}
			if notes == '_alpha_0.4':
				flowerrs[warp][geom] = flowerr

		if f.endswith(".npy"):
			p = os.path.join(root, f)
			print(p)
			#./synthetictests/flowresults/square1/rotate_testflow_solver_it_20/rms_flow.npz.npy
			s = p.split('/')
			geom = s[3]
			aa = s[4].split('_testflow')
			warp = aa[0]

			if warp not in rms:
				rms[warp] = {}
			if geom not in rms[warp]:
				rms[warp][geom] = {}

			if len(aa) > 1:
				notes = aa[1]
			else:
				notes = ''
			rms_flow = np.load(p)
			averms = np.mean(rms_flow[rms_flow > 0])
			rms[warp][geom][notes] = averms

#Make a matrix out of you
nW = len(rms)
nG = len(rms[warp])
nN = len(rms[warp][geom])

warps = rms.keys()
geoms = rms[warp].keys()
notes = rms[warp][geom].keys()

rms_mat = np.zeros((nW, nN, nG))

for i,warp in enumerate(warps):
	for j,geom in enumerate(geoms):
		for k,note in enumerate(notes):
			rms_mat[i,k,j] = rms[warp][geom][note]

#fn_out = img_out + '.csv'
#np.savetxt(fn_out, rms_mat)

nS = 50000

#Combine flow errs 
flowerrs_comb = {}
for i,warp in enumerate(warps):
	for j,geom in enumerate(geoms):
		if not flowerrs[warp][geom].isnull().values.any():
			print 'adding', geom
			if warp not in flowerrs_comb:
				flowerrs_comb[warp] = flowerrs[warp][geom]
			else:
				flowerrs_comb[warp] = pd.concat([flowerrs_comb[warp], flowerrs[warp][geom]])
	nT = len(flowerrs_comb[warp])
	samp = rd.random_integers(0,nT-1,nS)
	df = flowerrs_comb[warp].iloc[samp]
	g1 = sns.jointplot("video_sample", "abs_res_sample", data = df, kind="kde", color="b")
	g1.savefig(img_out + '_' + warp + '_intensity_abs_res.eps')
	g2 = sns.jointplot("abs_flow_sample", "abs_res_sample", data = df, kind="kde", color="b")
	g2.savefig(img_out + '_' + warp + '_flow_abs_res.eps')
	#Wait for user input...
	raw_input("Press ENTER to continue")
