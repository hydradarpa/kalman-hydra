import numpy as np
import cv2
import scipy.spatial as spspatial
import distmesh as dm 
import distmesh.mlcompat as ml
import distmesh.utils as dmutils
from matplotlib import pyplot as plt
import bob.ip.optflow.liu

from imgproc import * 
from distmesh_dyn import *

################################################################################
#Parameters
################################################################################

threshold = 7
fn = "./video/local_prop_cb_with_bud.avi"
fn_out = "./video/local_prop_cb_with_bud_dense_liu.avi"

################################################################################
#Set up object
################################################################################

cap = cv2.VideoCapture(fn)
ret, frame = cap.read()
frame_orig = frame.copy()
(mask, ctrs, fd) = findObjectThreshold(frame, threshold = threshold)

nx,ny = frame.shape[0:2]

#Show the outline
cv2.drawContours(frame, ctrs.contours, -1, (0,255,0), 3)
cv2.imshow('frame',frame)
k = cv2.waitKey(30) & 0xff

fps = 20.0
framesize = np.shape(frame)[0:2]
framesize = framesize[::-1]
fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
output = cv2.VideoWriter(fn_out,fourcc, fps, framesize)

################################################################################
#Mesh creation
################################################################################

#(p, t, bars, L, params) = distmesh(ctrs, fd, frame)

################################################################################
#Main loop
################################################################################

hsv = np.zeros_like(frame)
hsv[...,1] = 255
prvs = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
while(cap.isOpened()):
	ret, frame2 = cap.read()
	if ret == False:
		break 
	next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

	#Optic flow
	#Dense flow
	#flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

	(u, v, warped) = bob.ip.optflow.liu.sor.flow(prvs, next, n_outer_fp_iterations=1, n_sor_iterations = 2)
	flow = np.stack((u, v), 2)
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
	dst = cv2.addWeighted(frame2,0.7,bgr,0.3,0)

	#This doesn't work very well.... moving onto a sparse example
	#See testmeshgen_homography

	#Track the points specified...

	#Move points based on flow 
	#yp = p[:,0].astype(int)
	#xp = p[:,1].astype(int)
	#yp[yp>=ny] = ny-1
	#xp[xp>=nx] = nx-1
	#p += flow[xp, yp, :]

	#Re-compute contours
	(mask, ctrs, fd) = findObjectThreshold(frame2, threshold = threshold)

	#-Reset bar lengths for points that moved a significant amount from flow.
	#-For points that didn't move, move these points with their forces,
	# keeping the points that did move from flow fixed.
	#distmesh_dynamic(p, t, bars, L, ctrs, fd, params)

	#Draw contour and lines
	cv2.drawContours(dst, ctrs.contours, -1, (0,255,0), 1)
	#drawGrid(dst, p, bars)
	cv2.imshow('frame',dst)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
	output.write(dst)

	prvs = next

cap.release()
output.release()
cv2.destroyAllWindows()