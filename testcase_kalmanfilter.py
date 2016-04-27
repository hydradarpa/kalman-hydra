#!/usr/bin/env python
import sys
from distmesh_dyn import DistMesh
from imgproc import findObjectThreshold 
from kalman import test_data, test_data_texture, test_data_image
from kalman2 import KalmanFilter, IteratedKalmanFilter, KalmanFilterMorph
from renderer import VideoStream

import pdb 
import time 
import cv2 

import numpy as np 
#import matplotlib.image as mpimg
import matplotlib.pyplot as plot 

cuda = False 
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
distmesh.createMesh(ctrs, h, frame, plot=True)

nx = 680
start = nx//3
end = 2*nx//3

kf = KalmanFilter(distmesh, frame, flowframe, cuda)
#kf = IteratedKalmanFilter(distmesh, frame, cuda)
#kf = KalmanFilterMorph(distmesh, frame, cuda)

nF = video.shape[2]
nI = 10
count = 0
for i in range(nF):
	count += 1
	print 'Frame %d' % count 
	frame = video[:,:,i]
	flowframe = flow[:,:,:,i]
	time.sleep(0.3)
	(e_im, e_fx, e_fy, fx, fy) = kf.compute(frame, flowframe, imageoutput = 'screenshots/' + name + '_frame_' + str(i))
	print 'Error image:', e_im
	print 'Error flow x:', e_fx
	print 'Error flow y:', e_fy