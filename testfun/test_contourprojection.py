#!/usr/bin/env python
import sys
from distmesh_dyn import DistMesh
from imgproc import findObjectThreshold, pointPolygonGrid
from synth import test_data, test_data_texture, test_data_image
from kalman import IteratedMSKalmanFilter, IteratedKalmanFilter
from renderer import VideoStream
from matplotlib import pyplot as plt

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
nF = video.shape[2]

gridsize = 60
print 'Running KF for gridsize =', gridsize
flowframe = flow[:,:,:,0]
frame = video[:,:,0]

distmesh = DistMesh(frame, h0 = gridsize)
mask, ctrs, h = findObjectThreshold(frame, threshold = threshold)
distmesh.createMesh(ctrs, h, frame, plot=False)

kf = IteratedMSKalmanFilter(distmesh, frame, flowframe, cuda, multi = True)
#kf = IteratedKalmanFilter(distmesh, frame, flowframe, cuda)

self = kf.state.renderer 
rend_mask = self.rendermask()

#plt.imshow(rend_mask[:,:,2]); plt.show()

#Perturb positions
kf.state.X[0:2*kf.state.N] += 10
kf.state.refresh()
kf.state.render()

#Then see if the projection correctly moves the borders back in the contour
kf.projectmask(mask)
kf.state.refresh()
kf.state.render()

#(mask2, ctrs, fd) = findObjectThreshold(mask, threshold = 0.5)
#grid = pointPolygonGrid(fd, 680, 680)
#plt.imshow(grid); plt.colorbar(); plt.show()