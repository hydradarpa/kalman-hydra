#!/usr/bin/env python
import sys
from distmesh_dyn import DistMesh
from imgproc import findObjectThreshold 
from synth import test_data, test_data_texture, test_data_image
from kalman import MSKalmanFilter, IteratedMSKalmanFilter, stats 
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

#gridsizes = [80, 60, 40, 20, 10, 8, 6, 4]
gridsizes = [80]
nF = video.shape[2]

for gridsize in gridsizes:
	print 'Running KF for gridsize =', gridsize
	flowframe = flow[:,:,:,0]
	frame = video[:,:,0]
	
	distmesh = DistMesh(frame, h0 = gridsize)
	mask, ctrs, h = findObjectThreshold(frame, threshold = threshold)
	distmesh.createMesh(ctrs, h, frame, plot=False)
	
	kf = IteratedMSKalmanFilter(distmesh, frame, flowframe, cuda)
	kf.state.render()
	
	count = 0
	for i in range(1):
		count += 1
		print 'Frame %d' % count 
		frame = video[:,:,i]
		mask = (frame > 0).astype(np.uint8)
		flowframe = flow[:,:,:,i]
		time.sleep(0.3)
		kf.compute(frame, flowframe, mask)

	kf.__del__()
	stats.reset()