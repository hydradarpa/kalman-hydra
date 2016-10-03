#!/usr/bin/env python
import sys, os
from renderer import VideoStream, FlowStream
from kalman import IteratedMSKalmanFilter
from distmesh_dyn import DistMesh
from scipy.io import loadmat 

import cv2 
import numpy as np 

fn_in='./video/20160412/stk_0001.avi'
fn_in2='./video/20160412/stk_0001a.avi'
flow_in='./video/20160412/stk_0001_flow/flow'
name='stack0001'
threshold = 42
cuda = True
gridsize = 25

mfsf_in = './mfsf_results/stk_0001_mfsf_nref100/'
dm_out = 'init_mesh.pkl'

imageoutput = mfsf_in + '/mesh/'
#Make directory if needed...
if not os.path.exists(imageoutput):
    os.makedirs(imageoutput)

#Load MFSF data
a = loadmat(mfsf_in + '/result.mat')

params = a['parmsOF']
u = a['u']
v = a['v']

#Find reference frame 
nref = params['nref'][0,0][0,0]

#Skip to this frame and create mesh 
capture = VideoStream(fn_in, threshold)

nx = int(capture.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
nF = int(capture.cap.get(cv2.CAP_PROP_FRAME_COUNT))

vid = np.zeros((nx, nx, nF))
masks = np.zeros((nx, nx, nF))
for idx in range(nF):
	print 'Loading frame', idx
	ret, frame, grayframe, mask = capture.read()
	if idx == nref:
		mask, ctrs, fd = capture.backsub()
		refframe = frame 
	vid[:,:,idx] = grayframe 
	masks[:,:,idx] = mask

#distmesh = DistMesh(frame, h0 = gridsize)
#distmesh.createMesh(ctrs, fd, frame, plot = True)
#Save this distmesh and reload it for quicker testing
#distmesh.save(mfsf_in + dm_out)

distmesh = DistMesh(refframe, h0 = gridsize)
distmesh.load(mfsf_in + dm_out)

refpositions = distmesh.p

#Create dummy input for flow frame
flowframe = np.zeros((nx, nx, 2))

#Create Kalman Filter object to store mesh and make use of plotting functions
kf = IteratedMSKalmanFilter(distmesh, refframe, flowframe, cuda = cuda)

#Perturb mesh points according to MFSF flow field and save screenshot output
nF = u.shape[2]
N = kf.size()/4

for idx in range(nF):
	#capture.seek(idx)
	#Update positions based on reference positions and flow field
	print("Visualizing frame %d" % idx)
	#ret, frame, y_im, y_m = capture2.read()
	y_im = vid[:,:,idx].astype(np.dtype('uint8'))
	y_m = masks[:,:,idx].astype(np.dtype('uint8'))
	#if not ret:
	#	print("End of stream encountered")
	#	break 
	kf.state.renderer.update_frame(y_im, flowframe, y_m)
	dx = u[refpositions[:,1].astype(int), refpositions[:,0].astype(int), idx]
	dy = v[refpositions[:,1].astype(int), refpositions[:,0].astype(int), idx]
	X = refpositions.copy()
	X[:,0] += dx
	X[:,1] += dy
	kf.state.X[0:2*N] = np.reshape(X, (-1,1))
	kf.state.refresh()
	kf.state.render()
	kf.state.renderer.screenshot(saveall=True, basename = imageoutput + '_frame_%03d' % idx)
