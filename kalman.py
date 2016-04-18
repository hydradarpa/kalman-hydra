#################################
#Extended iterated Kalman filter#
#################################

#lansdell. Feb 9th 2016

import numpy as np
import cv2 

from distmesh_dyn import DistMesh
from renderer import Renderer

import pdb

class KFState:
	def __init__(self, distmesh, im, cuda, eps_Q = 1, eps_R = 1e-3):
		#Set up initial geometry parameters and covariance matrices
		self._ver = np.array(distmesh.p, np.float32)

		self._vel = np.zeros(self._ver.shape, np.float32)
		#For testing we'll give some initial velocity
		#self._vel = np.ones(self._ver.shape, np.float32)

		#Set up initial guess for texture
		self.tex = im
		self.nx = im.shape[0]
		self.ny = im.shape[1]
		#self.M = self.nx*self.ny

		#Number of 'observations'
		self.NZ = 1
		self.eps_Q = eps_Q
		self.eps_R = eps_R

		#Fixed quantities
		#Coordinates relative to texture. Stays constant throughout video
		self.N = distmesh.size()
		self.u = self._ver
		#Orientation of simplices
		self.tri = distmesh.t
		self.NT = self.tri.shape[0]
		#The SciPy documentation claims the edges are provided in an order that orients
		#them counter-clockwise, though this doesn't appear to be the case...
		#So we should keep the orientations here. Still not sure if the magnitude
		#of the cross product is useful or not. Probably not
		a = self._ver[self.tri[:,1],:] - self._ver[self.tri[:,0],:]
		b = self._ver[self.tri[:,2],:] - self._ver[self.tri[:,0],:]
		self.ori = np.sign(np.cross(a,b))

		#Form state vector
		self.X = np.vstack((self._ver.reshape((-1,1)), self._vel.reshape((-1,1))))
		e = np.eye(2*self.N)
		z = np.zeros((2*self.N,2*self.N))
		self.F = np.bmat([[e, e], [z, e]])
		self.Q = eps_Q * np.bmat([[e/4, e/2], [e/2, e]])
		self.R = eps_R * np.ones((self.NZ,self.NZ))
		self.P = np.eye(self._vel.shape[0]*4)

		#Renderer
		self.renderer = Renderer(distmesh, self._vel, self.nx, im, self.eps_R, cuda)

	def size(self):
		return self.X.shape[0]

	def refresh(self):
		self.renderer.update_vertex_buffer(self.vertices(), self.velocities())

	def render(self):
		self.renderer.on_draw(None)

	def vertices(self):
		return self.X[0:(2*self.N)].reshape((-1,2))

	def velocities(self):
		return self.X[(2*self.N):].reshape((-1,2))

	def z(self, y):
		return self.renderer.z(y)

class KalmanFilter:
	def __init__(self, distmesh, im, cuda):
		self.distmesh = distmesh
		self.N = distmesh.size()
		print 'Creating filter with ' + str(self.N) + ' nodes'
		self.state = KFState(distmesh, im, cuda)
		self.state.M = self.observation(im)
		#print self.state.M 

	def compute(self, y_im, y_flow = None):
		self.state.renderer.update_frame(y_im)
		self.predict()
		self.update(y_im, y_flow = None)

	def predict(self):
		print 'Predicting'
		#import rpdb2 
		#rpdb2.start_embedded_debugger("asdf")

		X = self.state.X 
		F = self.state.F 
		Q = self.state.Q
		P = self.state.P 
		#Prediction equations 
		self.state.X = F*X
		self.state.P = F*P*F.T + Q 
		#print np.sum(self.state.velocities())

	def size(self):
		#State space size
		return self.N*4

	def update(self, y_im, y_flow = None):
		#import rpdb2 
		#rpdb2.start_embedded_debugger("asdf")
		z_tilde = self.observation(y_im)
		print 'z = %f' % z_tilde 
		print 'Updating'
		X = self.state.X
		P = self.state.P
		R = self.state.R
		H = self.linearize_obs(z_tilde, y_im)
		M = self.state.M
		I = np.eye(P.shape[0])

		##Update equations 
		S = H*P*H.T + R
		print S.shape 
		Sinv = np.linalg.inv(S)
		K = P*H.T*Sinv 
		self.state.P = (I - K*H)*P 
		self.state.X = X + K*(z_tilde - M)

	def observation(self, y_im):
		self.state.refresh()
		self.state.render()
		return self.state.z(y_im)

	def linearize_obs(self, z_tilde, y_im, deltaX = 2):
		H = np.zeros((1, self.size()))
		for idx in range(self.state.N*2):
			self.state.X[idx,0] += deltaX
			zp = self.observation(y_im)
			self.state.X[idx,0] -= deltaX
			H[0,idx] = (z_tilde - zp)/deltaX
		return H

class IteratedKalmanFilter(KalmanFilter):
	def __init__(self, distmesh, im, cuda):
		KalmanFilter.__init__(self, distmesh, im, cuda)
		self.nI = 100

	def update(self, y_im, y_flow = None):
		#import rpdb2 
		#rpdb2.start_embedded_debugger("asdf")
		print 'Updating'
		X = self.state.X
		P = self.state.P
		R = self.state.R
		M = self.state.M
		I = np.eye(P.shape[0])

		for i in range(self.nI):
			z_tilde = self.observation(y_im)
			#print 'z = %f' % z_tilde 
			H = self.linearize_obs(z_tilde, y_im)	
			##Update equations 
			S = H*P*H.T + R
			Sinv = np.linalg.inv(S)
			K = P*H.T*Sinv
			self.state.X = X + K*(z_tilde - M)
		self.state.P = (I - K*H)*P

class KFStateMorph(KFState):
	def __init__(self, distmesh, im, cuda, eps_Q = 1, eps_R = 1e-3):
		#Set up initial geometry parameters and covariance matrices
		self._ver = np.array(distmesh.p, np.float32)
		#Morph basis connecting mesh points to morph bases
		#T and K have the property that T*K = X (positions of nodes)
		self._generate_morph_basis(distmesh)

		self._vel = np.zeros(self.K.shape, np.float32)
		#For testing we'll give some initial velocity
		self._vel = np.ones(self.K.shape, np.float32)

		#Set up initial guess for texture
		self.tex = im
		self.nx = im.shape[0]
		self.ny = im.shape[1]
		self.M = self.nx*self.ny

		#Number of 'observations'
		self.NZ = self.nx*self.ny 
		self.eps_Q = eps_Q
		self.eps_R = eps_R

		#Fixed quantities
		#Coordinates relative to texture. Stays constant throughout video
		self.N = self.K.shape[0]
		self.u = self._ver
		#Orientation of simplices
		self.tri = distmesh.t
		self.NT = self.tri.shape[0]
		#The SciPy documentation claims the edges are provided in an order that orients
		#them counter-clockwise, though this doesn't appear to be the case...
		#So we should keep the orientations here. Still not sure if the magnitude
		#of the cross product is useful or not. Probably not
		a = self._ver[self.tri[:,1],:] - self._ver[self.tri[:,0],:]
		b = self._ver[self.tri[:,2],:] - self._ver[self.tri[:,0],:]
		self.ori = np.sign(np.cross(a,b))

		#Form state vector
		self.X = np.vstack((self.K.reshape((-1,1)), self._vel.reshape((-1,1))))
		self.V = self.velocities().reshape((-1,1))
		e = np.eye(2*self.N)
		z = np.zeros((2*self.N,2*self.N))
		self.F = np.bmat([[e, e], [z, e]])
		self.Q = eps_Q * np.bmat([[e/4, e/2], [e/2, e]])
		self.R = eps_R * np.ones((self.NZ,self.NZ))
		self.P = np.eye(self._vel.shape[0]*4)

		#Renderer
		self.renderer = Renderer(distmesh, self.V, self.nx, im, self.eps_R, cuda)

	def _generate_morph_basis(self, distmesh):
		#import rpdb2 
		#rpdb2.start_embedded_debugger("asdf")
		self.T = np.ones((distmesh.p.shape[0], 3))
		#K here is just translation of points
		self.K = np.mean(distmesh.p, axis = 0)
		self.T[:,0:2] = distmesh.p - self.K 

	def vertices(self):
		ver = self.X[0:2].reshape((-1,2))
		K = np.vstack((np.np.eye(2), ver))
		return np.dot(self.T,K)

	def velocities(self):
		vel = self.X[2:].reshape((-1,2))
		K = np.vstack((np.zeros((2,2)), vel))
		return np.dot(self.T,K)

	def z(self, y):
		return self.renderer.z(y)

class KalmanFilterMorph(KalmanFilter):
	def __init__(self, distmesh, im, cuda):
		self.distmesh = distmesh
		self.N = distmesh.size()
		print 'Creating filter with ' + str(self.N) + ' nodes'
		self.state = KFStateMorph(distmesh, im, cuda)

	def linearize_obs(self, z_tilde, y_im, deltaX = 2):
		H = np.zeros((self.state.M, self.size()))
		for idx in range(self.state.N*2):
			self.state.X[idx,0] += deltaX
			zp = self.observation(y_im)
			self.state.X[idx,0] -= deltaX
			H[:,idx] = (z_tilde - zp)/deltaX
		return H

def test_data(nx, ny):
	nframes = 10
	speed = 3
	video = np.zeros((nx, ny, nframes), dtype = np.uint8)
	im = np.zeros((nx, ny))
	#Set up a box in the first frame, with some basic changes in intensity
	start = nx//3
	end = 2*nx//3
	width = end-start
	height = end-start
	flow = np.zeros((nx, ny, 2, nframes))

	for i in range(start,end):
		for j in range(start,end):
			if i > j:
				col = 128
			else:
				col = 255
			im[i,j] = col
	im = np.flipud(im)
	#Translate the box for a few frames
	for i in range(nframes):
		imtrans = im[speed*i:,speed*i:]
		s = start-speed*i
		flow[s:s+width, s:s+height,0,i] = -speed
		flow[s:s+width, s:s+height,1,i] = -speed
		#flow[s:s+width, s:s+height,0,i] = -speed
		#flow[s:s+width, s:s+height,1,i] = -speed
		if i > 0:
			video[:-speed*i,:-speed*i,i] = imtrans 
		else:
			video[:,:,i] = imtrans
	return video, flow

def test_data_up(nx, ny):
	nframes = 30
	speed = 2
	video = np.zeros((nx, ny, nframes), dtype = np.uint8)
	im = np.zeros((nx, ny))
	#Set up a box in the first frame, with some basic changes in intensity
	start = nx//3
	end = 2*nx//3
	for i in range(start,end):
		for j in range(start,end):
			if i > j:
				col = 128
			else:
				col = 255
			im[i,j] = col
	#Translate the box for a few frames
	for i in range(nframes):
		imtrans = im[:,speed*i:]
		if i > 0:
			video[:,:-speed*i,i] = imtrans 
		else:
			video[:,:,i] = imtrans
	return video 

def test_data_texture(nx, ny):
	noise = 0
	nframes = 10
	speed = 3
	video = np.zeros((nx, ny, nframes), dtype = np.uint8)
	im = np.zeros((nx, ny))
	#Set up a box in the first frame, with some basic changes in intensity
	start = nx//3
	end = 2*nx//3
	for i in range(start,end):
		for j in range(start,end):
			if i > j:
				col = 128
			else:
				col = 200
			im[i,j] = col + noise*np.random.normal(size = (1,1))
	#Add noise
	#noise = 
	#Apply Gaussian blur 
	#im = im + noise 
	im = cv2.GaussianBlur(im,(15,15),0)
	#Translate the box for a few frames
	for i in range(nframes):
		imtrans = im[speed*i:,speed*i:]
		if i > 0:
			video[:-speed*i,:-speed*i,i] = imtrans 
		else:
			video[:,:,i] = imtrans
	return video 

def test_data_image(fn = './video/milkyway.jpg'):
	imsrc = cv2.imread(fn,0)
	nx = imsrc.shape[0]
	ny = imsrc.shape[1]
	im = np.zeros((nx, ny), dtype=np.uint8)
	imsrc = cv2.resize(imsrc,None,fx=1./2, fy=1./2, interpolation = cv2.INTER_CUBIC)
	width = imsrc.shape[0]
	height = imsrc.shape[1]
	start = nx//3
	im[start:start+width, start:start+height] = imsrc

	nframes = 10
	speed = 3
	noise = 10
	video = np.zeros((nx, ny, nframes), dtype = np.uint8)
	flow = np.zeros((nx, ny, 2, nframes))

	#Translate the box for a few frames
	for i in range(nframes):
		start = nx//3-i*speed
		imtrans = im[speed*i:,speed*i:]
		if i > 0:
			flow[start:start+width, start:start+height,0,i] = -speed
			flow[start:start+width, start:start+height,1,i] = -speed
			video[:-speed*i,:-speed*i,i] = imtrans 
		else:
			video[:,:,i] = imtrans

	imnoisex = np.zeros((nx, ny))
	imnoisey = np.zeros((nx, ny))
	#Set up a box in the first frame, with some basic changes in intensity
	start = nx//3
	end = nx//3+width
	for i in range(start,end):
		for j in range(start,end):
			imnoisex[i,j] = noise*np.random.normal(size = (1,1))
			imnoisey[i,j] = noise*np.random.normal(size = (1,1))
	#Add noise
	#noise = 
	#Apply Gaussian blur 
	#im = im + noise 
	imnoisex = cv2.GaussianBlur(imnoisex,(15,15),0)
	imnoisey = cv2.GaussianBlur(imnoisey,(15,15),0)

	flow[:,:,0,0] = imnoisex
	flow[:,:,0,0] = imnoisey

	return video, flow 