#################################
#Extended iterated Kalman filter#
#################################

#lansdell. Feb 9th 2016

import numpy as np
import cv2 

from distmesh_dyn import DistMesh
from renderer import Renderer

import pdb
import sys
from timeit import timeit 

np.set_printoptions(threshold = 'nan', linewidth = 150, precision = 1)

class KFState:
	def __init__(self, distmesh, im, flow, cuda, eps_F = 1, eps_H = 1e-3, vel = None):
		#Set up initial geometry parameters and covariance matrices
		self._ver = np.array(distmesh.p, np.float32)
		#self._vel = np.zeros(self._ver.shape, np.float32)
		if vel is None:
			self._vel = np.zeros(self._ver.shape, np.float32)
			#For testing we'll give some initial velocity
			#self._vel = -3*np.ones(self._ver.shape, np.float32)
		else:
			self._vel = vel.reshape(self._ver.shape)

		#Set up initial guess for texture
		self.tex = im
		self.nx = im.shape[0]
		self.ny = im.shape[1]
		self.M = self.nx*self.ny

		#Number of observations
		self.NZ = self.M
		self.eps_F = eps_F
		self.eps_H = eps_H

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
		self.Weps = eps_F * np.bmat([[e/4, e/2], [e/2, e]])
		self.W = np.eye(self._vel.shape[0]*4)

		#Renderer
		self.renderer = Renderer(distmesh, self._vel, flow, self.nx, im, cuda, showtracking = True)

	def get_flow(self):
		return self.renderer.get_flow()

	def size(self):
		return self.X.shape[0]

	def refresh(self):
		self.renderer.update_vertex_buffer(self.vertices(), self.velocities())

	def render(self):
		self.renderer.on_draw(None)

	def update(self, y_im, y_flow):
		Hz = self._jacobian(y_im, y_flow)
		HTH = self._hessian(y_im, y_flow)
		return (Hz, HTH)

	def _jacobian(self, y_im, y_flow, deltaX = 2):
		Hz = np.zeros((self.size(),1))
		self.refresh() 
		self.render()
		#Set reference image to unperturbed images
		self.renderer.initjacobian(y_im, y_flow)
		for idx in range(self.size()):
			self.X[idx,0] += deltaX
			self.refresh()
			self.render()
			hz = self.renderer.jz(self)
			Hz[idx,0] = hz/deltaX
			self.X[idx,0] -= deltaX

			self.X[idx,0] -= deltaX
			self.refresh()
			self.render()
			hz = self.renderer.jz(self)
			Hz[idx,0] -= hz/deltaX
			self.X[idx,0] += deltaX
			Hz[idx,0] = Hz[idx,0]/2
		self.refresh() 
		self.render()
		return Hz

	def _hessian(self, y_im, y_flow, deltaX = 2):
		HTH = np.zeros((self.size(),self.size()))
		self.refresh() 
		self.render()
		#Set reference image to unperturbed images
		self.renderer.initjacobian(y_im, y_flow)
		#Very inefficient... for now 
		for i in range(self.size()):
			for j in range(i, self.size()):
				hij = self.renderer.j(self, deltaX, i, j)
				HTH[i,j] = hij/deltaX/deltaX
				#Fill in the other diagonal
				HTH[j,i] = HTH[i,j]
		self.refresh() 
		self.render()
		return HTH

	def vertices(self):
		return self.X[0:(2*self.N)].reshape((-1,2))

	def velocities(self):
		return self.X[(2*self.N):].reshape((-1,2))

class KalmanFilter:
	def __init__(self, distmesh, im, flow, cuda, vel = None):
		self.distmesh = distmesh
		self.N = distmesh.size()
		print 'Creating filter with ' + str(self.N) + ' nodes'
		self.state = KFState(distmesh, im, flow, cuda, vel=vel)
		self.predtime = 0
		self.updatetime = 0

	def compute(self, y_im, y_flow, mask = None, imageoutput = None):
		self.state.renderer.update_frame(y_im, y_flow)
		#Mask optic flow frame by contour of y_im
		if mask:
			y_flowx_mask = np.multiply(mask, y_flow[:,:,0])
			y_flowy_mask = np.multiply(mask, y_flow[:,:,1])
			y_flow_mask = np.dstack((y_flowx_mask, y_flowy_mask))
		else:
			y_flow_mask = y_flow
		pt = timeit(self.predict, number = 1)
		ut = timeit(lambda: self.update(y_im, y_flow_mask), number = 1)
		self.predtime += pt
		self.updatetime += ut

		print 'Current state:', self.state.X.T
		print 'Prediction time:', pt 
		print 'Update time: ', ut 
		#Save state of each frame
		if imageoutput is not None:
			self.state.renderer.screenshot(saveall=True, basename = imageoutput)
		#Compute error between predicted image and actual frames
		return self.error(y_im, y_flow)

	def predict(self):
		print '-- predicting'
		#import rpdb2 
		#rpdb2.start_embedded_debugger("asdf")

		X = self.state.X 
		F = self.state.F 
		Weps = self.state.Weps
		W = self.state.W 

		#Prediction equations 
		self.state.X = np.dot(F,X)
		self.state.W = np.dot(F, np.dot(W,F.T)) + Weps 
		#print np.sum(self.state.velocities())

	def size(self):
		#State space size
		return self.N*4

	def update(self, y_im, y_flow):
		#import rpdb2 
		#rpdb2.start_embedded_debugger("asdf")
		print '-- updating'
		X = self.state.X
		W = self.state.W
		eps_H = self.state.eps_H
		(Hz, HTH) = self.state.update(y_im, y_flow)
		invW = np.linalg.inv(W) + HTH/eps_H
		W = np.linalg.inv(invW)
		self.state.X = X + np.dot(W,Hz)/eps_H
		self.state.W = W 

	def error(self, y_im, y_flow):
		#Compute error of current state and current images and flow data
		return self.state.renderer.error(self.state, y_im, y_flow)

class IteratedKalmanFilter(KalmanFilter):
	def __init__(self, distmesh, im, flow, cuda):
		KalmanFilter.__init__(self, distmesh, im, flow, cuda)
		self.nI = 10

	def update(self, y_im, y_flow = None):
		#import rpdb2
		#rpdb2.start_embedded_debugger("asdf")
		#np.set_printoptions(threshold = 'nan', linewidth = 150, precision = 1)
		print '-- updating'
		X = self.state.X
		X_orig = X.copy()
		W = self.state.W
		W_orig = W.copy()
		invW_orig = np.linalg.inv(W)
		invW = np.linalg.inv(W)
		eps_H = self.state.eps_H
		for i in range(self.nI):
			sys.stdout.write('   IEKF K = %d\n'%i)
			sys.stdout.flush()
			(Hz, HTH) = self.state.update(y_im, y_flow)
			invW = invW_orig + HTH/eps_H
			W = np.linalg.inv(invW)
			X = X + np.dot(W,Hz)/eps_H
			self.state.X = X
			self.state.W = W 
			e_im, e_fx, e_fy, fx, fy = self.error(y_im, y_flow)
			sys.stdout.write('-- e_im: %d, e_fx: %d, e_fy: %d\n'%(e_im, e_fx, e_fy))

class KFStateMorph(KFState):
	def __init__(self, distmesh, im, flow, cuda, eps_Q = 1, eps_R = 1e-3):
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
		self.NZ = self.M
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
		self.renderer = Renderer(distmesh, self.V, flow, self.nx, im, self.eps_R, cuda)

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

class KalmanFilterMorph(KalmanFilter):
	def __init__(self, distmesh, im, flow, cuda):
		self.distmesh = distmesh
		self.N = distmesh.size()
		print 'Creating filter with ' + str(self.N) + ' nodes'
		self.state = KFStateMorph(distmesh, im, flow, cuda)

	def linearize_obs(self, z_tilde, y_im, deltaX = 2):
		H = np.zeros((self.state.M, self.size()))
		for idx in range(self.state.N*2):
			self.state.X[idx,0] += deltaX
			zp = self.observation(y_im)
			self.state.X[idx,0] -= deltaX
			H[:,idx] = (z_tilde - zp)/deltaX
		return H

class UnscentedKalmanFilter(KalmanFilter):
	def __init__(self, distmesh, im, flow, cuda):
		KalmanFilter.__init__(self, distmesh, im, flow, cuda)

		self.L=numel(x);                                 #numer of states
		self.m=numel(z);                                 #numer of measurements
		self.alpha=1e-3;                                 #default, tunable
		self.ki=0;                                       #default, tunable
		self.beta=2;                                     #default, tunable
		#scaling factor
		self.lmda=self.alpha*alpha*(self.L+self.ki)-self.L;
		self.c=self.L+self.lmda;                         #scaling factor

	def predict(self):
		#Setup sigma points and weights
		#Wm=[self.lmda/self.c, 0.5/self.c+zeros(1,2*self.L)];           #weights for means
		#Wc=Wm;
		#Wc(1)=Wc(1)+(1-self.alpha*self.alpha+self.beta);           #weights for covariance
		#c=sqrt(c);
		#X=sigmas(x,P,c);                            #sigma points around x
		#propagate sigma points 
		#[x1,X1,P1,X2]=ut(fstate,X,Wm,Wc,L,Q);		#unscented transformation of process
		return False 

	def update(self, y_im, y_flow = None):
		#[z1,Z1,P2,Z2]=ut(hmeas,X1,Wm,Wc,m,R)        #Unscented transformation of measurments
		#P12=X2*diag(Wc)*Z2.T                        #Transformed cross-covariance
		#K=P12*inv(P2)
		#x=x1+K*(z-z1)                               #State update
		#The problem here is that K is a NxM matrix, which is very large. 
		#Worse is that P2 is MxM... too big :(
		#P=P1-K*P12.T                                #Covariance update
		return False

	def ut(self,f,X,Wm,Wc,n,R):
		#Unscented Transformation
		#Input:
		#        f: nonlinear map
		#        X: sigma points
		#       Wm: weights for mean
		#       Wc: weights for covraiance
		#        n: numer of outputs of f
		#        R: additive covariance
		#Output:
		#        y: transformed mean
		#        Y: transformed smapling points
		#        P: transformed covariance
		#       Y1: transformed deviations
		L=size(X,2);
		y=zeros(n,1);
		Y=zeros(n,L);
		#for k in range(self.L):                   
		#	Y(:,k)=f(X(:,k));       
		#	y=y+Wm(k)*Y(:,k);       
		#Y1=Y-y(:,ones(1,L));
		#P=Y1*diag(Wc)*Y1.T+R;          
		return False #[y,Y,P,Y1]

	def sigmas(self,x,P,c):
		#Sigma points around reference point
		#Inputs:
		#       x: reference point
		#       P: covariance
		#       c: coefficient
		#Output:
		#       X: Sigma points		
		#A = c*chol(P).T
		#Y = x(:,ones(1,numel(x)))
		#X = [x Y+A Y-A]
		return False