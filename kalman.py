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
		self.M = self.nx*self.ny

		#Number of 'observations'
		self.NZ = 1000
		self.eps_Q = eps_Q
		self.eps_R = eps_R

		self.Rp = np.zeros((self.NZ, self.M))
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
		return self.renderer.z(y, self.Rp)

class KalmanFilter:
	def __init__(self, distmesh, im, cuda):
		self.distmesh = distmesh
		self.N = distmesh.size()
		print 'Creating filter with ' + str(self.N) + ' nodes'
		self.state = KFState(distmesh, im, cuda)
		print self.state.M 

	def compute(self, y_im, y_flow = None):
		self.state.renderer.update_frame(y_im)
		self.predict()
		self.update(y_im, y_flow = None)

	def predict(self):
		print 'Predicting'
		X = self.state.X 
		F = self.state.F 
		Q = self.state.Q
		P = self.state.P 
		#Prediction equations 
		self.state.X = F*X
		self.state.P = F*P*F.T + Q 

	def size(self):
		#State space size
		return self.N*4

	def Rp(self):
		a = np.random.normal(size = (self.state.NZ, self.state.M))
		ni = 1/np.linalg.norm(a, axis=1)
		an = (a.T*ni).T
		return an

	def update(self, y_im, y_flow = None):
		#import rpdb2 
		#rpdb2.start_embedded_debugger("asdf")
		print 'Updating'
		#Update a random projection matrix 
		self.state.Rp = self.Rp()
		z_res = self.observation(y_im)
		X = self.state.X
		P = self.state.P
		R = self.state.R
		H = self.linearize_obs(z_res, y_im)
		I = np.eye(P.shape[0])

		##Update equations 
		S = H*P*H.T + R
		#print S.shape 
		Sinv = np.linalg.inv(S)
		K = P*H.T*Sinv 
		self.state.P = (I - K*H)*P 
		self.state.X = X + K*z_res

	def observation(self, y_im):
		self.state.refresh()
		self.state.render()
		return self.state.z(y_im)

	def linearize_obs(self, z_tilde, y_im, deltaX = 2):
		print 'Linearizing z(x) around current estimate'
		H = np.zeros((self.state.NZ, self.size()))
		Xorig = self.state.X.copy()
		#print self.state.X.shape 
		for idx in range(self.state.N*2):
			#Perturb positions
			self.state.X[idx,0] += deltaX
			#Update and render
			zp = self.observation(y_im)

			#self.state.X[idx,0] -= 2*deltaX
			#Update and render
			#zp2 = self.observation(y_im)

			#Record change in z_tilde given change in position
			#H[0,idx] = (zp2 - zp)/deltaX/2
			H[:,idx] = -(zp-z_tilde)[:,0]/deltaX
			self.state.X = Xorig.copy()

		#We don't need to perturb the velocities, since we assume that we don't
		#observe them (yet), thus any perturbation will have no effect on z,
		#thus this part of H stays zero
		return H

def test_data(nx, ny):
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
				col = 255
			im[i,j] = col
	#Translate the box for a few frames
	for i in range(nframes):
		imtrans = im[speed*i:,speed*i:]
		if i > 0:
			video[:-speed*i,:-speed*i,i] = imtrans 
		else:
			video[:,:,i] = imtrans
	return video 


def test_data_texture(nx, ny):
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
			im[i,j] = col
	#Add noise
	noise = 5*np.random.normal(size = im.shape)
	#Apply Gaussian blur 
	im = im + noise 
	im = cv2.GaussianBlur(im,(15,15),0)
	#Translate the box for a few frames
	for i in range(nframes):
		imtrans = im[speed*i:,speed*i:]
		if i > 0:
			video[:-speed*i,:-speed*i,i] = imtrans 
		else:
			video[:,:,i] = imtrans
	return video 