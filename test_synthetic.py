from kalman2 import KalmanFilter
from renderer import VideoStream, FlowStream
from distmesh_dyn import DistMesh
import os.path 
import cv2
import numpy as np 

name = 'square2_gradient'
ff = 'translate_leftup'

m_in = './synthetictests/' + name + '/' + ff + '_mesh.txt'
v_in = './synthetictests/' + name + '/' + ff + '/' + ff + '.avi'
flow_in = './synthetictests/' + name + '/' + ff + '/' + ff + '_flow'

gridsize = 50
threshold = 8

#Create KF
print 'Loading synthetic data streams'
capture = VideoStream(v_in, threshold)
frame = capture.current_frame()
mask, ctrs, fd = capture.backsub()
distmesh = DistMesh(frame, h0 = gridsize)
distmesh.createMesh(ctrs, fd, frame, plot = False)

#Load true data
f_mesh = open(m_in, 'r')
lines = f_mesh.readlines()
nF = len(lines)-1
nX = int(lines[0].split(',')[1])
truestates = np.zeros((nF, nX*4), dtype = np.float32)
predstates = np.zeros((nF, nX*4), dtype = np.float32)
for i in range(1,nF+1):
	line = lines[i]
	truestates[i-1,:] = [float(x) for x in line.split(',')[1:]]

#Update distmesh.p to be exactly the first frame
distmesh.p = truestates[0,0:(2*nX)].reshape(nX,2)
predstates[0,:] = truestates[0,:]

flowstream = FlowStream(flow_in)
ret_flow, flowframe = flowstream.read()
kf = KalmanFilter(distmesh, frame, flowframe, cuda = False)

count = 0
print 'Tracking with Kalman filter'
while(capture.isOpened()):
	count += 1
	ret, frame, grayframe = capture.read()
	ret_flow, flowframe = flowstream.read()
	if ret is False or ret_flow is False:
		break
	print 'Frame %d' % count 
	kf.compute(grayframe, flowframe)
	predstates[count,:] = np.squeeze(kf.state.X)
np.save('./synthetictests/' + name + '/' + ff + '.pred.npz', predstates, truestates)	

print 'Done'


print 'How\'d we do?'

#Compare trajectory computed to actual trajectory (L2 error)


