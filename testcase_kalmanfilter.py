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
video, flow = test_data(680, 680)
#video, flow = test_data_texture(680, 680)
#video, flow = test_data_image()

flowframe = flow[:,:,:,0]
frame = video[:,:,0]

#Flip everything to check if there's an error in the data or in the the processing
#frame = frame.transpose()
#flowframe = flowframe.transpose((1,0,2))
#flowframe = flowframe[:,:,::-1]

#Make contours
##mask, ctrs, fd = capture.backsub()
distmesh = DistMesh(frame, h0 = gridsize)

mask, ctrs, h = findObjectThreshold(frame, threshold = threshold)
distmesh.createMesh(ctrs, h, frame, plot=True)

#Simplify things further... just one triangle
#distmesh.p = np.array([[ 226., 453.],
#       [ 453., 226.],
#       [ 453., 453.]])
#distmesh.t = np.array([[0, 1, 2]])
#distmesh.N = 3

#Correct the coordinates, save the data for unit test code purposes...
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

	#frame = frame.transpose()
	#flowframe = flowframe.transpose((1,0,2))
	#flowframe = flowframe[:,:,::-1]

	#for j in range(nI):
	time.sleep(0.3)
	#cv2.waitKey(0)
	#raw_input("--Iteration %d Finished. Press Enter to continue" % j)
	(e_im, e_fx, e_fy, fx, fy) = kf.compute(frame, flowframe, imageoutput = 'screenshots/' + name + '_frame_' + str(i))
	print 'Error im:', e_im
	print 'Error flow x:', e_fx
	print 'Error flow y:', e_fy
	#kf.compute(grayframe, flowframe)