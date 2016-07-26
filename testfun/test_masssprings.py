#!/usr/bin/env python
import sys
from distmesh_dyn import DistMesh
from imgproc import findObjectThreshold 
from synth import test_data, test_data_texture, test_data_image
from kalman import KalmanFilter, IteratedKalmanFilter, KalmanFilterMorph,\
		MSKalmanFilter, IteratedMSKalmanFilter
from renderer import VideoStream

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

kf = IteratedMSKalmanFilter(distmesh, frame, flowframe, cuda)

#Now perturb the positions a bit...
kf.state.X = kf.state.X*1.2
kf.state.refresh()

rend = kf.state.renderer
cuda = kf.state.renderer.cudagl 
kf.state.render()
state = kf.state 
nx = nx 
deltaX = -2

nF = video.shape[2]
nI = 10
count = 0

for i in range(500):
	count += 1
	print 'Frame %d' % count 
	#frame = video[:,:,i]
	#mask = (frame > 0).astype(np.uint8)
	#flowframe = flow[:,:,:,i]
	kf.predict()
	kf.state.refresh()
	kf.state.render()
	#Wait for user input:
	raw_input("Press enter to continue, ctrl-d to quit")
