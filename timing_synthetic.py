#!/usr/bin/env python
import sys
from distmesh_dyn import DistMesh
from imgproc import findObjectThreshold 
from synth import test_data, test_data_texture, test_data_image
from kalman import MSKalmanFilter, IteratedMSKalmanFilter, stats, IteratedKalmanFilter
from renderer import VideoStream

import pdb 
import time 
import cv2 

import numpy as np 
import matplotlib.pyplot as plot 

cuda = True
threshold = 9
name = 'test_data'
video, flow = test_data(680, 680)

nx = 680
start = nx//3
end = 2*nx//3
nI = 5

gridsizes = [80, 60, 40, 20, 10, 8, 6, 4]
#gridsizes = [80]
nF = video.shape[2]

#In data, for each grid size, we store:
#- mesh pts 
#- ave update time, 
#- ave pred time,
#- no. partitions jacobian
#- no. partitions hessian
#- ave per update iteration renders
#- ave per update iteration jacobain renders
#- ave per update iteration hessian renders
#- ave per update iteration renders theoretical (without multi pert rendering)
#- ave per update jacobain time
#- ave per update hessian time

data = np.zeros((len(gridsizes), 11))

for idx, gridsize in enumerate(gridsizes):
	idx = 1; gridsize = 30
	print 'Running KF for gridsize =', gridsize
	flowframe = flow[:,:,:,0]
	frame = video[:,:,0]
	
	distmesh = DistMesh(frame, h0 = gridsize)
	mask, ctrs, h = findObjectThreshold(frame, threshold = threshold)
	distmesh.createMesh(ctrs, h, frame, plot=False)
	
	kf = IteratedKalmanFilter(distmesh, frame, flowframe, cuda, multi = False)
	kf.state.render()
	
	count = 0
	for i in range(3):
		count += 1
		print 'Frame %d' % count 
		frame = video[:,:,i]
		mask = (frame > 0).astype(np.uint8)
		flowframe = flow[:,:,:,i]
		time.sleep(1)
		kf.compute(frame, flowframe, mask)

	kf.state.renderer.cudagl._destroy_PBOs()
	kf.__del__()

	#Extract stats
	meshpts = stats.meshpts 
	ave_update_time = stats.stateupdatetc[0]/stats.stateupdatetc[1]
	ave_pred_time = stats.statepredtime[0]/stats.stateupdatetc[1]
	jacpartitions = stats.jacobianpartitions
	hesspartitions = stats.hessianpartitions
	ave_nrenders = stats.renders[0]/stats.stateupdatetc[1]
	ave_jacrenders = stats.jacobianrenderstc[1]/stats.stateupdatetc[1]
	ave_hessrenders = stats.hessianrenderstc[1]/stats.stateupdatetc[1]
	ave_theoryrenders = ave_jacrenders + ave_hessrenders
	ave_jac_time = stats.jacobianrenderstc[0]/stats.stateupdatetc[1]
	ave_hess_time = stats.hessianrenderstc[0]/stats.stateupdatetc[1]

	data[idx, :] = [meshpts, ave_update_time, ave_pred_time, jacpartitions,\
					hesspartitions, ave_nrenders, ave_jacrenders, ave_hessrenders,\
					ave_theoryrenders, ave_jac_time, ave_hess_time]

	stats.reset()

#Plot data