from kalman import KalmanFilter, IteratedKalmanFilter, IteratedMSKalmanFilter
from renderer import VideoStream, FlowStream
from distmesh_dyn import DistMesh
import os.path 
import cv2
import numpy as np 

from vispy import gloo 

from matplotlib import pyplot as plt

name = 'hydra1'
ff = 'warp'
notes = 'masked_iekf'
cuda = True
sparse = True

#Input
m_in = './synthetictests/' + name + '/' + ff + '_mesh.txt'
dm_in = './synthetictests/' + name + '/' + ff + '_initmesh.pkl'
v_in = './synthetictests/' + name + '/' + ff + '/' + ff + '.avi'
flow_in = './synthetictests/' + name + '/' + ff + '/' + ff + '_flow'

#Output
img_out = './synthetictests/' + name + '/' + ff + '_' + notes + '_pred/'
if not os.path.isdir(img_out):
	os.makedirs(img_out)

gridsize = 18
threshold = 8

#Create KF
print 'Loading synthetic data streams'
capture = VideoStream(v_in, threshold)
frame = capture.current_frame()
mask, ctrs, fd = capture.backsub()
distmesh = DistMesh(frame, h0 = gridsize)
distmesh.load(dm_in)

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

predstates[0,:] = truestates[0,:]

rms_vel = np.zeros(nF)
rms_pos = np.zeros(nF)

flowstream = FlowStream(flow_in)
ret_flow, flowframe = flowstream.read()

kf = IteratedKalmanFilter(distmesh, frame, flowframe, cuda = cuda, sparse = sparse)
self = kf.state.renderer

with self._fbo4:
	m = gloo.read_pixels()
np.max(m[:,:,2])
np.sum(m[:,:,2] == 255)
np.unique(m[:,:,2])
kf.state.E[0]
kf.state.labels
np.unique(m[:,:,1])
