import numpy as np
import cv2
import scipy.spatial as spspatial
import distmesh as dm 
import distmesh.mlcompat as ml
import distmesh.utils as dmutils
import time 
import h5py
from matplotlib import pyplot as plt

from imgproc import * 
from distmesh_dyn import *

################################################################################
#Parameters
################################################################################

threshold = 7
#fn = "./video/local_prop_cb_with_bud.avi"
fn = './video/20150306_GCaMP_Chl_EC_mouth_open.avi'

#flow_in = "./flows/local_prop_cb_with_bud_dense_liu_sor.hdf"
flow_in = './flows/20150306_GCaMP_Chl_EC_mouth_open.hdf'

#fn_out = './video/local_pro_cb_with_bud_lui_mesh.avi'
fn_out = './video/20150306_GCaMP_Chl_EC_mouth_open_dense_lui.avi'

################################################################################
#Set up object
################################################################################

#Read in video
cap = cv2.VideoCapture(fn)
ret, frame = cap.read()
frame_orig = frame.copy()
nx,ny = frame.shape[0:2]
nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
(mask, ctrs, fd) = findObjectThreshold(frame, threshold = threshold)

#Show the outline
cv2.drawContours(frame, ctrs.contours, -1, (0,255,0), 1)
cv2.imshow('frame',frame)
k = cv2.waitKey(30) & 0xff

#Read in flow
f = h5py.File(flow_in, 'r')
flowliu = f['float']

#Output
fps = 20.0
framesize = np.shape(frame)[0:2]
framesize = framesize[::-1]
fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
output = cv2.VideoWriter(fn_out,fourcc, fps, framesize)

################################################################################
#Mesh creation
################################################################################

distmesh = DistMesh(frame, h0 = 25)
distmesh.createMesh(ctrs, fd, frame)

################################################################################
#Main loop
################################################################################

prvs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
count = 0 

while(cap.isOpened()):
	ret, frame2 = cap.read()
	if ret == False:
		break 
	next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

	#Get dense optic flow
	flowfr = flowliu[count,:,:,:]
	p = distmesh.p.astype(int)
	idx = np.logical_and(p[:,0]<ny,p[:,1]<nx)
	flow = flowfr[:,p[idx,1],p[idx,0]].T
	distmesh.p[idx,:] += flow
	count += 1

	#Update the other points
	###Recompute contours
	(mask, ctrs, fd) = findObjectThreshold(frame2, threshold = threshold)
	##-Reset bar lengths for points that moved from flow.
	##-For points that didn't move, move these points with their forces,
	## keeping the points that did move from flow fixed.
	##-Add a few iterations of motion for all points to stabilize noisy movements
	#distmesh.updateMesh(ctrs, fd, next)

	#Draw contour and lines
	cv2.drawContours(frame2, ctrs.contours, -1, (0,255,0), 1)
	drawGrid(frame2, distmesh.p, distmesh.bars, distmesh.L, distmesh.F)
	draw_str(frame2, (20, 20), 'frame count: %d/%d' % (count, nframes))
	cv2.imshow('frame',frame2)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
	output.write(frame2)

	prvs = next

cap.release()
output.release()
cv2.destroyAllWindows()