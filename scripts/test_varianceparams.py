#!/usr/bin/env python
import sys
import pdb 
import time 
import cv2 
import os

import numpy as np 
import matplotlib.pyplot as plt 

from multiprocessing import Pool 
from distmesh_dyn import DistMesh
from kalman import KalmanFilter, IteratedKalmanFilter, KalmanFilterMorph
from renderer import VideoStream, FlowStream

#name = 'square3_gradient_texture'
#name = 'square2_gradient'
#name = 'square1'
name = 'hydra1'
#ff = 'translate_leftup_stretch'
ff = 'translate_leftup'
cuda = True
sparse = True

#Input
m_in = './synthetictests/' + name + '/' + ff + '_mesh.txt'
dm_in = './synthetictests/' + name + '/' + ff + '_initmesh.pkl'
v_in = './synthetictests/' + name + '/' + ff + '/' + ff + '.avi'
flow_in = './synthetictests/' + name + '/' + ff + '/' + ff + '_flow'

mask_flow = True 
nI = 2
eps_Zs = np.power(10, np.linspace(-5,-1,5))		#Image variance/weight
eps_Js = np.power(10, np.linspace(-5,-1,5))		#Flow variance/weight
eps_Ms = np.power(10, np.linspace(-5,-1,5))		#Mask variance/weight

#Trial run...
eps_Zs = [1e-3]		#Image variance/weight
eps_Js = [1e-3]		#Flow variance/weight
eps_Ms = [1e-7]		#Mask variance/weight
eps_Z = 1
eps_J = 1
eps_M = 1e-5
eps_F = 1

gridsize = 18
threshold = 8
max_frames = 30

def runIEKF(eps_Z, eps_J, eps_M, eps_F):	
	print 'Running with:'
	print '- eps_Z:', eps_Z
	print '- eps_J:', eps_J
	print '- eps_M:', eps_M
	print '- eps_F:', eps_F

	if mask_flow:
		notes = 'masked_iekf'
	else:
		notes = 'iekf'
	notes += '_eps_Z_%f'%eps_Z
	notes += '_eps_J_%f'%eps_J
	notes += '_eps_M_%f'%eps_M
	notes += '_eps_F_%f'%eps_F

	img_out = './synthetictests/' + name + '/' + ff + '_' + notes + '_pred/'
	if not os.path.isdir(img_out):
		os.makedirs(img_out)

	print 'Loading synthetic data streams'
	capture = VideoStream(v_in, threshold)
	frame = capture.current_frame()
	mask, ctrs, fd = capture.backsub()
	distmesh = DistMesh(frame, h0 = gridsize)
	distmesh.load(dm_in)
	
	#Load true data
	f_mesh = open(m_in, 'r')
	lines = f_mesh.readlines()
	nF = len(lines)-1
	nX = int(lines[0].split(',')[1])
	truestates = np.zeros((nF, nX*4), dtype = np.float32)
	predstates = np.zeros((nF, nX*4), dtype = np.float32)
	for i in range(1,nF+1):
		line = lines[i]
		truestates[i-1,:] = [float(x) for x in line.split(',')[1:]]
	
	predstates[0,:] = truestates[0,:]
	
	rms_vel = np.zeros(nF)
	rms_pos = np.zeros(nF)
	
	flowstream = FlowStream(flow_in)
	ret_flow, flowframe = flowstream.read()
	kf = IteratedKalmanFilter(distmesh, frame, flowframe, cuda = cuda, sparse = sparse,\
	 eps_F = eps_F, eps_Z = eps_Z, eps_J = eps_J, eps_M = eps_M, nI = nI)
	
	count = 0
	print 'Tracking with Kalman filter'
	while(capture.isOpened() and count < max_frames):
	#for idx in range(1):
		count += 1
		ret, frame, grayframe, mask = capture.read()
		ret_flow, flowframe = flowstream.read()
		if ret is False or ret_flow is False:
			break
	
		print 'Frame %d' % count 
		kf.compute(grayframe, flowframe, mask, maskflow = mask_flow, imageoutput = img_out+'solution_frame_%03d'%count)
		#kf.compute(grayframe, flowframe, mask, maskflow = mask_flow)
	
		predstates[count,:] = np.squeeze(kf.state.X)
		r_pos = truestates[count,0:(2*nX)]-predstates[count,0:(2*nX)]
		r_vel = truestates[count,(2*nX):]-predstates[count,(2*nX):]
		rms_pos[count] = np.sqrt(np.mean(np.multiply(r_pos, r_pos)))
		rms_vel[count] = np.sqrt(np.mean(np.multiply(r_vel, r_vel)))
		print 'RMS_pos:', rms_pos[count], 'RMS_vel:', rms_vel[count]
	
	print 'Saving'
	np.savez('./synthetictests/' + name + '/' + ff + '_' + notes + '_pred.npz', predstates, truestates, rms_pos, rms_vel)
	
	print 'Done... how\'d we do?'
	
	#Make plot of tracking error
	plt.plot(range(nF), rms_pos, label='RMS position')
	plt.plot(range(nF), rms_vel, label='RMS velocity')
	plt.legend(loc='upper left')
	plt.ylabel('RMS')
	plt.xlabel('Frame')
	plt.savefig('./synthetictests/' + name + '/' + ff + '_' + notes + '_pred_rms.eps')

#def work(f):
	

#Loop over variances
pool = Pool(processes = 5)
for eps_Z in eps_Zs:
	for eps_J in eps_Js:
		#Use a multiprocessing Pool to do this instead??
		#runkf = lambda m: runIEKF(eps_Z, eps_J, m)
		#pool.map(runkf, eps_Ms)

		#Or the serial way...
		for eps_M in eps_Ms:
			runIEKF(eps_Z, eps_J, eps_M, eps_F)