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
import dill

import numpy as np 
#import matplotlib.image as mpimg
import matplotlib.pyplot as plot 

def save_object(objs, filename):
	with open(filename, 'wb') as f:
		dill.dump(objs, f)

cuda = False 
gridsize = 110
threshold = 9

name = 'test_data'
video, flow = test_data(680, 680)
#video, flow = test_data_texture(680, 680)
#video, flow = test_data_image()
flowframe = flow[:,:,:,0]
frame = video[:,:,0]

#Make contours
##mask, ctrs, fd = capture.backsub()
distmesh = DistMesh(frame, h0 = gridsize)
mask, ctrs, h = findObjectThreshold(frame, threshold = threshold)
distmesh.createMesh(ctrs, h, frame, plot=True)

#Correct the coordinates, save the data for unit test code purposes...
nx = 680
start = nx//3
end = 2*nx//3
distmesh.p = np.array([[start, start], [end, start], [start, end], [end, end]], np.float32)
distmesh.plot()
vel = np.zeros(distmesh.p.shape, np.float32)

#Save images of all ones and all zeros, for unit tests 
save_object((distmesh, vel, flowframe, nx, frame), './test/testdata_texture.pkl')
frame = np.zeros(frame.shape, np.uint8)
flowframe = np.zeros(flowframe.shape, np.float32)
save_object((distmesh, vel, flowframe, nx, frame), './test/testdata_zeros.pkl')
frame = 128*np.ones(frame.shape, np.uint8)
flowframe = 1.534*np.ones(flowframe.shape, np.float32)
vel = 1.534*np.ones(distmesh.p.shape, np.float32)
save_object((distmesh, vel, flowframe, nx, frame), './test/testdata_ones.pkl')