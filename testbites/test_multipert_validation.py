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

#HTH = np.zeros((self.size(),self.size()))
#HTH_c = np.zeros((4, self.size(), self.size()))
#deltaX = 2
#y_im = frame
#y_flow = flowframe 
#y_m = mask 
#for idx, e in enumerate(self.E_hessian):
#	self.refresh(idx, hess = True) 
#	self.render()
#	#Set reference image to unperturbed images
#	self.renderer.initjacobian(y_im, y_flow, y_m)
#	ee = e.copy()
#	eeidx = self.E_hessian_idx[idx]
#	#print e 
#	for i1 in range(2):
#		for j1 in range(2):
#			for i2 in range(2):
#				for j2 in range(2):
#					offset1 = i1+2*self.N*j1 
#					offset2 = i2+2*self.N*j2 
#					ee[:,0] = 2*e[:,0] + offset1 
#					ee[:,1] = 2*e[:,1] + offset2 
#					#Do the render
#					(h, h_hist, hcomp) = self.renderer.j_multi(self, deltaX, ee, idx, eeidx)
#					#Unpack the answers into the hessian matrix
#					h = h[h_hist > 0]
#					hcomp = hcomp[np.squeeze(np.array(h_hist > 0)),:]
#					qidx = self.Q[np.squeeze(np.array(h_hist)),:]
#					for idx2 in range(len(qidx)):
#						q = qidx[idx2]
#						q1 = 2*q[0]+i1+2*self.N*j1
#						q2 = 2*q[1]+i2+2*self.N*j2
#						HTH[q1,q2] = h[0,idx2]/deltaX/deltaX
#						HTH[q2,q1] = HTH[q1,q2]
#						HTH_c[0,q1,q2] = hcomp[idx2,0]/deltaX/deltaX
#						HTH_c[0,q2,q1] = HTH_c[0,q1,q2]
#						HTH_c[1,q1,q2] = hcomp[idx2,1]/deltaX/deltaX
#						HTH_c[1,q2,q1] = HTH_c[1,q1,q2]
#						HTH_c[2,q1,q2] = hcomp[idx2,2]/deltaX/deltaX
#						HTH_c[2,q2,q1] = HTH_c[2,q1,q2]
#						HTH_c[3,q1,q2] = hcomp[idx2,3]/deltaX/deltaX
#						HTH_c[3,q2,q1] = HTH_c[3,q1,q2]



(HTH_multi) = self._hessian_sparse_multi(frame, flowframe, mask, deltaX = 2)
#(HTH) = self._hessian_sparse(frame, flowframe, mask, deltaX = 2)
#np.savez('./test_multipert_validation_single_HTH.npz', HTH = HTH)
b = np.load('./test_multipert_validation_single_HTH.npz')

#Larger deviations from HTH. Some of magnitude 100% suggesting some components are
#reported as being zero in one case and non-zero in the other. 
HTH = b['HTH']

#Number of missing entries in HTH_multi 
np.sum((np.abs(HTH)>0.000001)&(np.abs(HTH_multi) < 0.000001))
#=180 out of 1122 are missing

#Number of additional entries in HTH_multi
np.sum((np.abs(HTH_multi)>0.000001)&(np.abs(HTH) < 0.000001))
#=0 

#Thus HTH_multi is missing some components...

#Of the one's that are there, what is the relative error?
HTH_diff = HTH-HTH_multi 
relHTHerr = 100*HTH_diff/HTH 
concurHTHrelerr = relHTHerr[(np.abs(HTH)>0.000001)&(np.abs(HTH_multi) > 0.000001)]
np.sum(np.abs(concurHTHrelerr)>2) #= 57 ... out of 942 are greater than 2% error

#Which ones are missing?

#Total number of non-zeros in HTH
#1122

#Need to investigate the Q listing and the E_hessian matrix....
np.transpose(np.nonzero((np.abs(HTH)>0.000001)&(np.abs(HTH_multi) < 0.000001)))

#For instance... these are some missing values:
#[[  0,   4],
#       [  0,   5],
#       [  0,   6],
#       [  0,   7],
#       [  1,   4],
#       [  1,   5],
#       [  1,   6],
#       [  1,   7],
#       [  2,  20],
#       [  2,  21],
#       [  3,  20],
#       [  3,  21],
#...

#Are these in Q?
#self.Q= array([[ 0,  0],
#       [ 0,  2],
#       [ 0,  3],
#       [ 1,  1],

#Yes... so they must go missing elsewhere... are they in self.E_hessian?
#In particular... where are [0,2] and [0,3] as these are missing according to above

#Both [0,2] and [0,3] are in E_hessian... so problem lies down the line...

#Is all of Q in E_hessian?
qlen = len(self.Q)
eqlen = np.sum(np.array([len(i) for i in self.E_hessian]))
#These lengths check out...

#Do all missing values from from one render?
#No, the ones marked -- below are missing. It's always i != j parts
#For some renders it is _all_ of the i!=j parts, but some renders are a mix
#[array([[ 0,  0],
#        [ 1,  1],
#        [ 4,  4],
#        [ 6,  6],
#        [ 9,  9],
#        [16, 16],
#        [17, 17],
#        [22, 22],
#        [26, 26],
#        [27, 27],
#        [31, 31],
#        [34, 34]]), 
#--	array([[ 0,  2],
#        [ 2,  2],
#--      [ 4,  5],
#        [ 5,  5],
#        [11, 11],
#        [13, 13],
#        [18, 18],
#        [19, 19],
#        [23, 23],
#--      [26, 29],
#        [29, 29],
#        [32, 32]]), 
#-- array([[ 0,  3],
#--      [ 1, 10],
#        [ 3,  3],
#--      [ 4,  8],
#        [ 8,  8],
#        [10, 10],
#        [12, 12],
#--      [13, 14],
#        [14, 14],
#--      [16, 20],
#        [18, 22],
#        [20, 20],
#--      [26, 30],
#        [28, 28],
#        [30, 30],
#        [32, 34]]), array([[ 1,  2],
#        [ 5,  8],
#        [11, 12],
#        [14, 19],
#--      [18, 25],
#        [23, 27],
#        [25, 25],
#        [29, 30]]), array([[ 2,  3],
#        [ 5,  9],
#        [11, 16],
#--      [22, 24],
#        [24, 24],
#        [30, 31]]), array([[ 2,  6],
#        [ 8,  9],
#        [16, 19],
#        [17, 18],
#        [24, 26],
#        [31, 32]]), 
#--     array([[ 2,  7],
#        [ 7,  7],
#        [ 9, 13],
#        [11, 17],
#        [19, 20],
#        [22, 25],
#        [28, 29],
#--      [32, 33],
#        [33, 33]]), array([[ 2, 10],
#        [12, 17],
#        [16, 23],
#        [25, 26],
#        [28, 31]]), array([[ 3,  7],
#        [ 5, 10],
#--      [16, 21],
#        [21, 21],
#        [24, 25],
#        [30, 33]]), array([[ 6,  7],
#        [ 9, 14],
#        [17, 21],
#        [20, 27],
#        [29, 31]]), array([[ 6, 10],
#        [18, 21],
#        [24, 27],
#        [31, 33]]), array([[ 6, 11],
#        [20, 23],
#        [33, 34]]), array([[ 6, 12],
#--      [ 9, 15],
#        [15, 15],
#        [21, 22],
#        [27, 28]]), array([[ 7, 12],
#        [ 9, 10],
#        [21, 23]]), array([[10, 11],
#        [22, 23]]), array([[10, 15],
#        [23, 24]]), array([[11, 15],
#        [24, 28]]), array([[11, 21],
#        [24, 29]]), array([[14, 15]]), array([[15, 16]]), array([[15, 19]])]

#Ah.. I see. No partition should have the same node referenced more than once,
#and yet some of these partitions do. In particular, all duplicates contain an 
#i == j pair. Thus the diagonal terms aren't being dealt with properly

