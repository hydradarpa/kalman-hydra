from synthetic import TestMesh
import os.path 
import cv2

name = 'square2_gradient'
ff = 'translate_leftup'

m_in = './synthetictests/' + name + '/' + ff + '_mesh.txt'
v_in = './synthetictests/' + name + '/' + ff
flow_in = './synthetictests/' + name + '/' + ff + '_flow_'

gridsize = 50
threshold = 8

#Create KF
capture = VideoStream(v_in, threshold)
frame = capture.current_frame()
mask, ctrs, fd = capture.backsub()
distmesh = DistMesh(frame, h0 = gridsize)
distmesh.createMesh(ctrs, fd, frame, plot = True)

#Load flow data from directory
flowstream = FlowStream(flow_in)

kf = KalmanFilter(distmesh, frame, cuda = False)

#Run for niterations
kf.compute(capture.gray_frame(), flowframe)


#Compare trajectory computed to actual trajectory (L2 error)