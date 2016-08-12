from kalman import KalmanFilter, IteratedKalmanFilter, IteratedMSKalmanFilter
from renderer import VideoStream, FlowStream
from distmesh_dyn import DistMesh
import os.path 
import cv2
import numpy as np 

from vispy import gloo 

from matplotlib import pyplot as plt

from useful import * 

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
self = kf.state

#Test labels
#with self._fbo4:
#	m = gloo.read_pixels()
#np.max(m[:,:,2])
#np.sum(m[:,:,2] == 255)
#np.unique(m[:,:,2])
#kf.state.E[0]
#kf.state.labels
#np.unique(m[:,:,1])

################################################################################
#### Test Jacobian calculation compared with CPU computation ###################
################################################################################

#self = kf.state
#Hz = np.zeros((self.size(),1))
#Hz_components = np.zeros((self.size(),4))
#self.refresh() 
#self.render()
#self.renderer.initjacobian(frame, flowframe, mask)
#idx = 0
#e = np.array(self.E[0])
#i = 0
#j = 0
#deltaX = 2
#
#offset = i+2*self.N*j 
#ee = offset + 2*e 
#self.X[ee,0] += deltaX
#self.refresh(idx)
#self.render()
#(hz, hzc) = self.renderer.jz(self)

#(hz, hzc) = self._jacobian(frame, flowframe, mask, deltaX = 2)
(hz_multi, hzc_multi) = self._jacobian_multi(frame, flowframe, mask, deltaX = 2)
#np.savez('./test_multipert_validation_single.npz', hz = hz, hzc = hzc)
a = np.load('./test_multipert_validation_single.npz')

#Shows five errors of around 2%, all others are small enough to be around round-off
#All errors occur in the mask component. Errors all of size 250 - might indicate something


################################################################################
##################Hessian computation###########################################
################################################################################

HTH = np.zeros((self.size(),self.size()))
HTH_c = np.zeros((4, self.size(), self.size()))
deltaX = 2
y_im = frame
y_flow = flowframe 
y_m = mask 
for idx, e in enumerate(self.E_hessian):
	self.refresh(idx, hess = True) 
	self.render()
	#Set reference image to unperturbed images
	self.renderer.initjacobian(y_im, y_flow, y_m)
	ee = e.copy()
	eeidx = self.E_hessian_idx[idx]
	#print e 
	for i1 in range(2):
		for j1 in range(2):
			for i2 in range(2):
				for j2 in range(2):
					offset1 = i1+2*self.N*j1 
					offset2 = i2+2*self.N*j2 
					ee[:,0] = 2*e[:,0] + offset1 
					ee[:,1] = 2*e[:,1] + offset2 
					#Do the render
					(h, h_hist, hcomp) = self.renderer.j_multi(self, deltaX, ee, idx, eeidx)
					#Unpack the answers into the hessian matrix
					h = h[h_hist > 0]
					hcomp = hcomp[np.squeeze(np.array(h_hist > 0)),:]
					qidx = self.Q[np.squeeze(np.array(h_hist)),:]
					for idx2 in range(len(qidx)):
						q = qidx[idx2]
						q1 = 2*q[0]+i1+2*self.N*j1
						q2 = 2*q[1]+i2+2*self.N*j2
						HTH[q1,q2] = h[0,idx2]/deltaX/deltaX
						HTH[q2,q1] = HTH[q1,q2]
						HTH_c[0,q1,q2] = hcomp[idx2,0]/deltaX/deltaX
						HTH_c[0,q2,q1] = HTH_c[0,q1,q2]
						HTH_c[1,q1,q2] = hcomp[idx2,1]/deltaX/deltaX
						HTH_c[1,q2,q1] = HTH_c[1,q1,q2]
						HTH_c[2,q1,q2] = hcomp[idx2,2]/deltaX/deltaX
						HTH_c[2,q2,q1] = HTH_c[2,q1,q2]
						HTH_c[3,q1,q2] = hcomp[idx2,3]/deltaX/deltaX
						HTH_c[3,q2,q1] = HTH_c[3,q1,q2]



(HTH_multi) = self._hessian_sparse_multi(frame, flowframe, mask, deltaX = 2)
#(HTH) = self._hessian_sparse(frame, flowframe, mask, deltaX = 2)
#np.savez('./test_multipert_validation_single_HTH.npz', HTH = HTH)
b = np.load('./test_multipert_validation_single_HTH.npz')

#Larger deviations from HTH. Some of magnitude 100% suggesting some components are
#reported as being zero in one case and non-zero in the other. 
