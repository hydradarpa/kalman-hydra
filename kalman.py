#################################
#Extended iterated Kalman filter#
#################################

#lansdell. Feb 9th 2016

import numpy as np
import cv2
import distmesh as dm 
import scipy.spatial as spspatial
import distmesh.mlcompat as ml
import distmesh.utils as dmutils

from imgproc import * 
from distmesh_dyn import *
from renderer import *

class KFState:
	def __init__(self, distmesh, im, eps_Q = 1, eps_R = 1):
		#Set up initial geometry parameters and covariance matrices
		self._ver = np.array(distmesh.p, np.float32)

		#self._vel = np.zeros(self._ver.shape, np.float32)
		#For testing, we'll give some initial velocity
		self._vel = np.ones(self._ver.shape, np.float32)

		#Set up initial guess for textures and covariance matrix
		self.tex = im
		self.nx = im.shape[0]
		self.c_tex = np.ones(im.shape)

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
		self.Q = eps_Q * np.eye(self._vel.shape[0]*4)
		self.R = eps_R * np.ones(im.shape)
		self.P = np.eye(self._vel.shape[0]*4)

		#Renderer
		self.renderer = Renderer(distmesh, self._vel, self.nx, im)
		app.run()

	def refresh(self):
		self.renderer.update_vertex_buffer(self.vertices(), self.velocities())

	def render(self):
		self.renderer.on_draw(None)
		im = self.renderer.get_buffers()
		#return image and flow buffers
		return im

	def vertices(self):
		return self.X[0:(2*self.N)].reshape((-1,2))

	def velocities(self):
		return self.X[(2*self.N):].reshape((-1,2))

class KalmanFilter:
	def __init__(self, distmesh, im):
		self.distmesh = distmesh
		self.N = distmesh.size()
		self.state = KFState(distmesh, im)

	def compute(self, z_im, z_flow):
		self.predict()
		self.update(z_im, z_flow)

	def predict(self):
		X = self.state.X 
		F = self.state.F 
		Q = self.state.Q
		P = self.state.P 
		#Prediction equations 
		self.state.X = F*X
		self.state.P = F*P*F.T + Q 

	def update(self, z_im, z_flow):
		#im, flow = self.observation()
		im = self.observation()
		P = self.state.P 
		R = self.state.R 
		X = self.state.X 
		H = self.linearize_obs()			#(expensive)

		##Update equations 
		#y_tilde_i = z_im - im; 
		#y_tilde_f = z_flow - flow;
		#y_tilde = np.vstack((y_tilde_i.reshape((1,-1)), y_tilde_f.reshape((1,-1))))
		#S = H*P*H.T + R
		#Sinv = np.linalg.inv(S)				#(expensive)
		#K = P*H.T*Sinv 
		#self.state.P = (I - K*H)*P 
		#self.state.X = X + K*y_tilde  

	def observation(self):
		self.state.refresh()
		return self.state.render()

	def linearize_obs(self):
		return None