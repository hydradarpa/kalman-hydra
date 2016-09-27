#!/usr/bin/env python
import sys
from distmesh_dyn import DistMesh
from imgproc import findObjectThreshold 
from synth import test_data, test_data_texture, test_data_image
from kalman import KalmanFilter, IteratedKalmanFilter,\
		IteratedMSKalmanFilter, stats
from renderer import VideoStream

import cProfile
import re
#cProfile.run('re.compile("foo|bar")')

import pdb 
import time 
import cv2 

import numpy as np 
#import matplotlib.image as mpimg
import matplotlib.pyplot as plot 

cuda = True
gridsize = 80
threshold = 9
name = 'test_data'

#Grab test data
video, flow = test_data(680, 680)
#video, flow = test_data_texture(680, 680)
#video, flow = test_data_image()
flowframe = flow[:,:,:,0]
frame = video[:,:,0]

#Make mesh
distmesh = DistMesh(frame, h0 = gridsize)
mask, ctrs, h = findObjectThreshold(frame, threshold = threshold)
distmesh.createMesh(ctrs, h, frame, plot=False)

nx = 680
start = nx//3
end = 2*nx//3

#kf = KalmanFilter(distmesh, frame, flowframe, cuda)
#kf = IteratedKalmanFilter(distmesh, frame, flowframe, cuda)
#kf = IteratedKalmanFilter(distmesh, frame, flowframe, cuda, sparse = False)
#kf = KalmanFilterMorph(distmesh, frame, cuda)

kf = IteratedMSKalmanFilter(distmesh, frame, flowframe, cuda, multi = True)
#kf = MSKalmanFilter(distmesh, frame, flowframe, cuda)

rend = kf.state.renderer
cuda = kf.state.renderer.cudagl 
kf.state.render()
state = kf.state 
nx = nx 
deltaX = -2

#Test creation of multi-perturbations

nF = video.shape[2]
nI = 10
count = 0

#for i in range(nF):
for i in range(5):
	count += 1
	print 'Frame %d' % count 
	frame = video[:,:,i]
	mask = (frame > 0).astype(np.uint8)
	flowframe = flow[:,:,:,i]
	time.sleep(0.3)

	#kf._newton
	#J = kf._jacobian()

	#(Hz, HTH, Hz_components) = kf.state.update(frame, flowframe, mask)

	#(e_im, e_fx, e_fy, e_m, fx, fy) = kf.compute(frame, flowframe, mask, imageoutput = 'screenshots/' + name + '_frame_' + str(i))

	#z_gpu = cuda.initjacobian(frame, flowframe, test = True)
	#z_cpu = cuda.initjacobian_CPU(frame, flowframe, test = True)
	#print 'Test initjacobian'
	#print 'CPU:', z_cpu
	#print 'GPU:', z_gpu
	#
	##Perturb vertices a bit and rerender
	#idx = 10
	#eps = 1e-9
	#print 'Test jz'
	#for idx in range(36):
	#	kf.state.X[idx,0] += deltaX
	#	kf.state.refresh()
	#	kf.state.render()
	#	kf.state.X[idx,0] -= deltaX
	#
	#	jz_gpu = cuda.jz(state)
	#	jz_cpu = cuda.jz_CPU(state)
	#	print idx, 'CPU:', jz_cpu, 'GPU:', jz_gpu, '% diff:', 100*abs(jz_gpu-jz_cpu)/(jz_cpu+eps)
	#
	#print 'Test j'
	#for i in range(20):
	#	for j in range(20):
	#		j_gpu = cuda.j(state, deltaX, i, j)
	#		j_cpu = cuda.j_CPU(state, deltaX, i, j)
	#		print i, j, 'CPU:', j_cpu, 'GPU:', j_gpu, '% diff:', 100*abs(j_gpu-j_cpu)/(j_cpu+eps)

	(e_im, e_fx, e_fy, e_m, fx, fy) = kf.compute(frame, flowframe, mask, imageoutput = './test_forces')
	print 'Error image:', e_im
	print 'Error flow x:', e_fx
	print 'Error flow y:', e_fy
	print 'Error mask:', e_m
