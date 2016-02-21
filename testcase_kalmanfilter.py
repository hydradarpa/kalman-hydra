#!/usr/bin/env python
import sys
from kalman import KalmanFilter, IteratedKalmanFilter, test_data, test_data_texture
from renderer import VideoStream
from distmesh_dyn import DistMesh
from imgproc import findObjectThreshold 

import pdb 
import time 
import cv2 

cuda = True 
gridsize = 100
threshold = 9

video = test_data(680, 680)
#video = test_data_texture(680, 680)
frame = video[:,:,0]
#Make contours
##mask, ctrs, fd = capture.backsub()
distmesh = DistMesh(frame, h0 = gridsize)
mask, ctrs, h = findObjectThreshold(frame, threshold = threshold)
distmesh.createMesh(ctrs, h, frame)

flowframe = None
#kf = KalmanFilter(distmesh, frame, cuda)
kf = IteratedKalmanFilter(distmesh, frame, cuda)
kf.compute(frame, flowframe)
nF = video.shape[2]
nI = 10
count = 0
for i in range(nF):
	count += 1
	print 'Frame %d' % count 
	frame = video[:,:,i]
	#for j in range(nI):
	time.sleep(0.3)
	cv2.waitKey(0)
		#raw_input("--Iteration %d Finished. Press Enter to continue" % j)
	kf.compute(frame, flowframe)
	#kf.compute(grayframe, flowframe)