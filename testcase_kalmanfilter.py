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

video, flow = test_data(680, 680)
#video, flow = test_data_texture(680, 680)
#video, flow = test_data_image()
flowframe = flow[:,:,:,0]
frame = video[:,:,0]
#Make contours
##mask, ctrs, fd = capture.backsub()
distmesh = DistMesh(frame, h0 = gridsize)
mask, ctrs, h = findObjectThreshold(frame, threshold = threshold)
distmesh.createMesh(ctrs, h, frame)

kf = KalmanFilter(distmesh, frame, flowframe, cuda)
#(flowx, flowy) = kf.state.get_flow()
#print np.max(flowx[:,:,0]), np.max(flowx[:,:,1]), np.max(flowx[:,:,2])
#print np.min(flowx[:,:,0]), np.min(flowx[:,:,1]), np.min(flowx[:,:,2])

#kf = IteratedKalmanFilter(distmesh, frame, cuda)
#kf = KalmanFilterMorph(distmesh, frame, cuda)
kf.compute(frame, flowframe)
nF = video.shape[2]
nI = 10
count = 0
for i in range(nF):
	count += 1
	print 'Frame %d' % count 
	frame = video[:,:,i]
	flowframe = flow[:,:,:,i]
	#for j in range(nI):
	time.sleep(0.3)
	cv2.waitKey(0)
		#raw_input("--Iteration %d Finished. Press Enter to continue" % j)
	kf.compute(frame, flowframe)
	#kf.compute(grayframe, flowframe)