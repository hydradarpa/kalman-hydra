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
self = kf.state.renderer

################################################################################
################################################################################
################################################################################

#kf.state.__init__
#Setup of labels for multipert ij list:
im = frame 
flow = flowframe 
cuda = True
eps_F = 1
eps_Z = 1e-3
eps_J = 1e-3
eps_M = 1e-3
vel = None
sparse = True

#Set up initial geometry parameters and covariance matrices
_ver = np.array(distmesh.p, np.float32)
#_vel = np.zeros(_ver.shape, np.float32)
if vel is None:
	_vel = np.zeros(_ver.shape, np.float32)
	#For testing we'll give some initial velocity
	#_vel = -3*np.ones(_ver.shape, np.float32)
else:
	_vel = vel.reshape(_ver.shape)

#Set up initial guess for texture
tex = im
nx = im.shape[0]
ny = im.shape[1]
M = nx*ny

#Take advantage of sparsity in Hessian H (default to True)
sparse = sparse

#Number of observations
NZ = M
eps_F = eps_F
eps_Z = eps_Z
eps_J = eps_J
eps_M = eps_M

#Fixed quantities
#Coordinates relative to texture. Stays constant throughout video
N = distmesh.size()
u = _ver
#Orientation of simplices
tri = distmesh.t
NT = tri.shape[0]
#The SciPy documentation claims the edges are provided in an order that orients
#them counter-clockwise, though this doesn't appear to be the case...
#So we should keep the orientations here. Still not sure if the magnitude
#of the cross product is useful or not. Probably not
a = _ver[tri[:,1],:] - _ver[tri[:,0],:]
b = _ver[tri[:,2],:] - _ver[tri[:,0],:]
ori = np.sign(np.cross(a,b))

#Remove faces that are too thin: sin(\theta) is too small
sineface = np.zeros(len(a))
for i in range(len(a)):
	sineface[i] = np.cross(a[i,:],b[i,:])/(np.linalg.norm(a[i,:])*np.linalg.norm(b[i,:]))
nz = abs(sineface) > 0.06
sineface = sineface[nz]
ori = ori[nz]
tri = tri[nz]
distmesh.t = tri 

#Form state vector
X = np.vstack((_ver.reshape((-1,1)), _vel.reshape((-1,1))))
e = np.eye(2*N)
z = np.zeros((2*N,2*N))
F = np.bmat([[e, e], [z, e]])
Weps = eps_F * np.bmat([[e/4, e/2], [e/2, e]])
W = np.eye(_vel.shape[0]*4)

#Note connectivity of vertices for efficient computing of Hessian H
Jv = np.eye(N)
for t in tri:
	Jv[t[0],t[1]] = 1
	Jv[t[0],t[2]] = 1
	Jv[t[1],t[2]] = 1
	Jv[t[1],t[0]] = 1
	Jv[t[2],t[0]] = 1
	Jv[t[2],t[1]] = 1

Jv = Jv 

#Can actually save more space by not computing the vx and vy cross perturbations
#as these will also be orthogonal. But that shouldn't have too big an impact really...
e = np.ones((2,2))
J = np.kron(Jv, e)
#print J 
J = np.kron(e,J)
I = distmesh.L.shape[0]

#Compute incidence matrix
Kp = np.zeros((N, I))
for idx,[i1,i2] in enumerate(distmesh.bars):
	#An edge exists, add these pairs
	Kp[i1,idx] = 1
	Kp[i2,idx] = -1
K = np.kron(Kp, np.eye(2))

#Compute initial edge lengths...
l0 = distmesh.L.copy()
L = distmesh.L 

#Compute partitions for multiperturbation rendering
E = []
labels = []
Q = np.arange(N)
A = np.arange(N)
while len(Q) > 0:
	#print 'Outer loop'
	P = np.array([])
	e = []
	while len(Q) > 0:
		#print '  * Inner loop'
		#Current vertex
		q = Q[0]
		#print '  * q = ', q
		#Things connected to current vertex
		p = np.nonzero(Jv[q,:])[0]
		p = np.setdiff1d(p, q)
		#print '  * p = ', p
		#Add current vertex
		e += [q]
		#Add things connected to current vertex to the 'do later' list
		P = np.intersect1d(np.union1d(P, p), A)
		#print '  * P = ', P
		A = np.setdiff1d(A, q)
		#Remove q and p from Q
		Q = np.setdiff1d(Q, p)
		Q = np.setdiff1d(Q, q)
	Q = P
	#if type(e).__module__ == np.__name__:
	#	ee = e.tolist()
	#else:
	#	ee = e
	#E += [ee]
	E += [e]

#print E 

#For each element of the partitions we label the triangles and assign them colors
#for the mask 
#We check there're no conflicts in the labels
#Horribly inefficient... 
labels = -1*np.ones((len(tri), len(E)))
for k,e in enumerate(E):
	label = -1*np.ones(len(tri))
	#For each triangle, find it any of its vertices are mentioned in e,
	#give it a color...
	for i, node in enumerate(e):
		for j, t in enumerate(tri):
			if node in t:
				label[j] = node
	labels[:,k] = label

#Compute independent pairs of dependent pairs of vertices
E_hessian = []
E_hessian_idx = []
Q = []
for i in range(N):
	for j in range(i+1,N):
		if Jv[i,j]:
			Q = Q + [[i,j]]
Q = np.array(Q)
Qidx = np.arange(len(Q))
A = Q.copy()
Aidx = Qidx.copy()

while len(Q) > 0:
	#print 'Outer loop'
	P = np.array([])
	Pidx = np.array([])
	e = np.array([])
	eidx = np.array([])
	while len(Q) > 0:
		#print '  * Inner loop'
		#Current vertices
		q = Q[0]
		qidx = Qidx[0]
		#All things connected to current vertex
		p1 = np.nonzero(Jv[q[0],:])[0]
		p2 = np.nonzero(Jv[q[1],:])[0]
		p = np.union1d(p1,p2)
		p = np.setdiff1d(p, q)
		#All pairs that contain these vertices
		p_all1 = np.array([i in p for i in Q[:,0]])
		p_all2 = np.array([i in p for i in Q[:,1]])
		p_all_idx = p_all1 + p_all2 
		p_all = Q[p_all_idx,:]
		p_all_idx = Qidx[p_all_idx]

		#Add current vertex
		e = union2d(e, q)
		eidx = np.union1d(eidx,[qidx])

		#Add things connected to current vertex to the 'do later' list
		P = intersect2d(union2d(P, p_all), A)
		Pidx = np.intersect1d(np.union1d(Pidx, p_all_idx), Aidx)
		A = setdiff2d(A, q)
		Aidx = np.setdiff1d(Aidx, qidx)
		#Remove q and p from Q
		Q = setdiff2d(Q, p_all)
		Q = setdiff2d(Q, q)
		Qidx = np.setdiff1d(Qidx, p_all_idx)
		Qidx = np.setdiff1d(Qidx, qidx)
	Q = P
	Qidx = Pidx
	E_hessian += [e]	
	E_hessian_idx += [eidx]	

#Create labels for the hessian multiperturbation
Q = []
for i in range(N):
	for j in range(i+1,N):
		if Jv[i,j]:
			Q = Q + [[i,j]]
Q = np.array(Q)

################################################################################
################################################################################
################################################################################


kf = IteratedKalmanFilter(distmesh, frame, flowframe, cuda = cuda, sparse = sparse)
self = kf.state.renderer

#Test labels
with self._fbo4:
	m = gloo.read_pixels()
np.max(m[:,:,2])
np.sum(m[:,:,2] == 255)
np.unique(m[:,:,2])
kf.state.E[0]
kf.state.labels
np.unique(m[:,:,1])

self = kf.state
Hz = np.zeros((self.size(),1))
Hz_components = np.zeros((self.size(),4))
self.refresh() 
self.render()
self.renderer.initjacobian(frame, flowframe, mask)
idx = 0
e = np.array(self.E[0])
i = 0
j = 0

offset = i+2*self.N*j 
ee = offset + 2*e 
deltaX = 2
self.X[ee,0] += deltaX
self.refresh(idx)
self.render()
(hz, hzc) = self.renderer.jz(self)
Hz[ee,0] = hz/deltaX
Hz_components[ee,:] = hzc/deltaX

self.X[ee,0] -= 2*deltaX
self.refresh(idx)
self.render()
(hz, hzc) = self.renderer.jz(self)
Hz[ee,0] -= hz/deltaX
Hz_components[ee,:] -= hzc/deltaX
self.X[ee,0] += deltaX

Hz[ee,0] = Hz[ee,0]/2
Hz_components[ee,:] = Hz_components[ee,:]/2

