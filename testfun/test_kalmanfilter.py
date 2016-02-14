import numpy as np
import cv2
import scipy.spatial as spspatial
import distmesh as dm 
import distmesh.mlcompat as ml
import distmesh.utils as dmutils
import time 
from matplotlib import pyplot as plt

from kalman import *
from imgproc import * 
from distmesh_dyn import *

################################################################################
#Parameters
################################################################################

threshold = 9
fn = "./video/GCaMP_local_prop.avi"
fn_out = './video/GCaMP_local_prop_grid.avi'
lk_params = dict( winSize  = (19, 19),
				  maxLevel = 2,
				  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict( maxCorners = 1000,
					   qualityLevel = 0.01,
					   minDistance = 8,
					   blockSize = 19 )

################################################################################
#Set up object
################################################################################

cap = cv2.VideoCapture(fn)
ret, frame = cap.read()
frame_orig = frame.copy()
(mask, ctrs, fd) = findObjectThreshold(frame, threshold = threshold)
ofLK = OpticalFlowLK(cap, lk_params, feature_params)

nx,ny = frame.shape[0:2]

#Show the outline
cv2.drawContours(frame, ctrs.contours, -1, (0,255,0), 1)
cv2.imshow('frame',frame)
k = cv2.waitKey(30) & 0xff

fps = 20.0
#framesize = np.shape(frame)[0:2]
#framesize = framesize[::-1]
#fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
#output = cv2.VideoWriter(fn_out,fourcc, fps, framesize)

################################################################################
#Mesh creation
################################################################################

distmesh = DistMesh(frame)
distmesh.createMesh(ctrs, fd, frame_orig, True)

prvs = cv2.cvtColor(frame_orig,cv2.COLOR_BGR2GRAY)
kf = KalmanFilter(distmesh, prvs)

kf.compute()

################################################################################
#Main loop
################################################################################

while(cap.isOpened()):
	ret, frame2 = cap.read()
	if ret == False:
		break 
	next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

	#Interpolate from sparse optic flow, using findHomography example:
	#Get sparse optic flow of a set of points with each frame

	#oft = time.time() 
	flowimg = ofLK.run(next, prvs, frame2)
	p0LK = ofLK.p0 
	p1LK = ofLK.p1
	flow = p1LK - p0LK
	#opticflowtime += time.time() - oft 

	#fst = time.time() 
	##Use these points to shift around adjacent points:
	nmvp = np.zeros((np.shape(distmesh.p)[0],1))
	mvp = np.zeros((np.shape(distmesh.p)))
	#Only look at points that are within boundary
	fdp1 = fd(p1LK[:,0,:])
	inbdry = fdp1 < 0 

	#Find points adjacent to each tracked point
	for tpt, tflow in zip(p0LK[inbdry,0,:], flow[inbdry,0,:]):
		simpidx = distmesh.delaunay.find_simplex(tpt)
		simp = distmesh.delaunay.vertices[simpidx]
		nmvp[simp] += 1 
		mvp[simp] += tflow

	#For each point, take the average motion (if adjacent to more than one
	#tracked point)
	moved = nmvp > 0 
	mvp[moved[:,0]] = np.divide(mvp[moved[:,0]], np.hstack((nmvp[moved,np.newaxis], nmvp[moved,np.newaxis])))
	distmesh.p += mvp

	#findsimplextime += time.time() - fst 


	#umt = time.time() 
	#Update the other points
	###Recompute contours
	#fot = time.time()
	(mask, ctrs, fd) = findObjectThreshold(frame2, threshold = threshold)
	#findobjecttime += time.time() - fot 
	##-Reset bar lengths for points that moved from flow.
	##-For points that didn't move, move these points with their forces,
	## keeping the points that did move from flow fixed.
	##-Add a few iterations of motion for all points to stabilize noisy movements
	#umt = time.time() 
	distmesh.updateMesh(ctrs, fd, next)
	#updateMesh(distmesh, ctrs, fd, next)
	#updatemeshtime += time.time() - umt 

	#Draw contour and lines
	#dt = time.time()
	cv2.drawContours(frame2, ctrs.contours, -1, (0,255,0), 1)
	drawGrid(frame2, distmesh.p, distmesh.bars)
	cv2.imshow('frame',frame2)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
	output.write(frame2)
	#drawingtime += time.time() - dt

	prvs = next

	#totaltime = drawingtime + updatemeshtime + findsimplextime + opticflowtime
	#totaltime = updatemeshtime + findobjecttime
	#print 'Drawing', 100*drawingtime/totaltime, 'Update Mesh', 100*updatemeshtime/totaltime, 'Find simplex', 100*findsimplextime/totaltime, 'Optic flow', 100*opticflowtime/totaltime
	#print 'Update Mesh', 100*updatemeshtime/totaltime, 'Find object', 100*findobjecttime/totaltime

cap.release()
output.release()
cv2.destroyAllWindows()