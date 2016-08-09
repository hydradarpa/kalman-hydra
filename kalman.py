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
from time import gmtime, strftime
from timeit import timeit 
import time 
from numpy.linalg import norm, inv 

from useful import * 

np.set_printoptions(threshold = 'nan', linewidth = 150, precision = 1)

class Statistics:
	def __init__(self):
		self.niter = 0
		self.meshpts = 0
		self.gridsize = 0
		self.jacobianpartitions = 0
		self.hessianpartitions = 0
		self.nzj = 0
		#number and timing of renders and updates
		#stored as mutable lists for scoping purposes in decorator...
		self.jacobianrenderstc = [0, 0]
		self.hessianrenderstc = [0, 0]
		self.renders = [0]
		self.statepredtime = [0]
		self.stateupdatetc = [0, 0]
		self.hessinc = [0]
		self.jacinc = [0]
		self.hessincsparse = [0]

	def reset(self):
		self.__init__()

#Decorator to count update times... uses time module so not as accurate
def timer(runtimer):
	def counter_wrapper(func):
		def func_wrapper(*args, **kwargs):
			st = time.time()
			ret = func(*args, **kwargs)
			et = time.time()
			runtimer[0] += et - st 
			return ret
		return func_wrapper
	return counter_wrapper

def counter(ctr, inc):
	def counter_wrapper(func):
		def func_wrapper(*args, **kwargs):
			st = time.time()
			ret = func(*args, **kwargs)
			et = time.time()
			ctr[0] += inc[0]
			return ret
		return func_wrapper
	return counter_wrapper

def timer_counter(tc, inc):
	def counter_wrapper(func):
		def func_wrapper(*args, **kwargs):
			st = time.time()
			ret = func(*args, **kwargs)
			et = time.time()
			#print runtimer 
			#print counter 
			tc[0] += et - st 
			tc[1] += inc[0]
			return ret
		return func_wrapper
	return counter_wrapper

stats = Statistics()

class KFState:
	def __init__(self, distmesh, im, flow, cuda, eps_F = 1, eps_Z = 1e-3, eps_J = 1e-3, eps_M = 1e-3, vel = None, sparse = True):
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

		#Take advantage of sparsity in Hessian H (default to True)
		self.sparse = sparse

		#Number of observations
		self.NZ = self.M
		self.eps_F = eps_F
		self.eps_Z = eps_Z
		self.eps_J = eps_J
		self.eps_M = eps_M

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

		#Remove faces that are too thin: sin(\theta) is too small
		self.sineface = np.zeros(len(a))
		for i in range(len(a)):
			self.sineface[i] = np.cross(a[i,:],b[i,:])/(np.linalg.norm(a[i,:])*np.linalg.norm(b[i,:]))
		nz = abs(self.sineface) > 0.06
		self.sineface = self.sineface[nz]
		self.ori = self.ori[nz]
		self.tri = self.tri[nz]
		distmesh.t = self.tri 

		#Form state vector
		self.X = np.vstack((self._ver.reshape((-1,1)), self._vel.reshape((-1,1))))
		e = np.eye(2*self.N)
		z = np.zeros((2*self.N,2*self.N))
		self.F = np.bmat([[e, e], [z, e]])
		self.Weps = eps_F * np.bmat([[e/4, e/2], [e/2, e]])
		self.W = np.eye(self._vel.shape[0]*4)

		#Note connectivity of vertices for efficient computing of Hessian H
		Jv = np.eye(self.N)
		for t in self.tri:
			Jv[t[0],t[1]] = 1
			Jv[t[0],t[2]] = 1
			Jv[t[1],t[2]] = 1
			Jv[t[1],t[0]] = 1
			Jv[t[2],t[0]] = 1
			Jv[t[2],t[1]] = 1

		self.Jv = Jv 

		#Can actually save more space by not computing the vx and vy cross perturbations
		#as these will also be orthogonal. But that shouldn't have too big an impact really...
		e = np.ones((2,2))
		J = np.kron(Jv, e)
		#print J 
		self.J = np.kron(e,J)
		self.I = distmesh.L.shape[0]

		#Compute incidence matrix
		self.Kp = np.zeros((self.N, self.I))
		for idx,[i1,i2] in enumerate(distmesh.bars):
			#An edge exists, add these pairs
			self.Kp[i1,idx] = 1
			self.Kp[i2,idx] = -1
		self.K = np.kron(self.Kp, np.eye(2))

		#Compute initial edge lengths...
		self.l0 = self.lengths()
		self.L = distmesh.L 

		#Compute partitions for multiperturbation rendering
		self.E = []
		self.labels = []
		Q = np.arange(self.N)
		A = np.arange(self.N)
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
			#self.E += [ee]
			self.E += [e]

		#print self.E 

		#For each element of the partitions we label the triangles and assign them colors
		#for the mask 
		#We check there're no conflicts in the labels
		#Horribly inefficient... 
		self.labels = -1*np.ones((len(self.tri), len(self.E)))
		for k,e in enumerate(self.E):
			label = -1*np.ones(len(self.tri))
			#For each triangle, find it any of its vertices are mentioned in e,
			#give it a color...
			for i, node in enumerate(e):
				for j, t in enumerate(self.tri):
					if node in t:
						label[j] = node
			self.labels[:,k] = label

		#print self.labels 

		#Compute independent pairs of dependent pairs of vertices
		Q = []
		for i in range(self.N):
			for j in range(i+1,self.N):
				if self.Jv[i,j]:
					Q = Q + [[i,j]] 

		while len(Q) > 0:
			#print 'Outer loop'
			P = []
			e = []
			while len(Q) > 0:
				#print '  * Inner loop'
				#Current vertices
				q = Q[0]

				#Things connected to current vertex
				p = np.nonzero(Jv[q,:])[0]
				p = np.setdiff1d(p, q)

				#Add current vertex
				e += [q]

				#Add things connected to current vertex to the 'do later' list
				P = np.intersect1d(np.union1d(P, p), A)

				A = np.setdiff1d(A, q)
				#Remove q and p from Q
				Q = np.setdiff1d(Q, p)
				Q = np.setdiff1d(Q, q)
			Q = P
			self.E += [e]			

		########################################################################
		#Compute independent pairs of dependent pairs of vertices###############
		########################################################################
		E_hessian = []
		E_hessian_idx = []
		Q = []
		for i in range(self.N):
			for j in range(i,self.N):
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
		
		self.E_hessian = E_hessian 
		self.E_hessian_idx = E_hessian_idx

		#Create labels for the hessian multiperturbation
		Q = []
		for i in range(self.N):
			for j in range(i,self.N):
				if Jv[i,j]:
					Q = Q + [[i,j]]
		Q = np.array(Q)
		self.Q = Q 

		self.labels_hess = -1*np.ones((len(self.tri), len(self.E_hessian)))
		#print self.E_hessian 
		for k,e in enumerate(self.E_hessian):
			label = -1*np.ones(len(self.tri))
			#For each triangle, find it any of its vertices are mentioned in e,
			#give it a color...
			if len(e.shape) < 2:
				e = np.reshape(e, (-1,2))
			for i, nodes in enumerate(e):
				#print nodes 
				n1, n2 = nodes 
				for j, t in enumerate(self.tri):
					if n1 in t or n2 in t:
						label[j] = E_hessian_idx[k][i]
			self.labels_hess[:,k] = label

		#Renderer
		self.renderer = Renderer(distmesh, self._vel, flow, self.nx, im, cuda, eps_Z, eps_J, eps_M, self.labels, self.labels_hess, self.Q, showtracking = True)

		#stats = Statistics()
		stats.meshpts = self.N
		stats.gridsize = distmesh.h0 
		stats.nzj = np.sum(self.J)/2+(self.N*4)/2
		stats.hessinc[0] = 2+(stats.meshpts*4)*(stats.meshpts*4)
		stats.jacinc[0] = 2+stats.meshpts*4*2
		stats.hessincsparse[0] = 2+(stats.nzj)*2
		try:
			stats.jacobianpartitions = len(self.E)
			stats.hessianpartitions = len(self.E_hessian)
		except AttributeError:
			pass 

	def __del__(self):
		self.renderer.__del__()

	def get_flow(self):
		return self.renderer.get_flow()

	def size(self):
		return self.X.shape[0]

	def refresh(self, multi_idx = -1, hess = False):
		if not hess:
			self.renderer.update_vertex_buffer(self.vertices(), self.velocities(), multi_idx)
		else:
			self.renderer.update_vertex_buffer(self.vertices(), self.velocities(), multi_idx, hess)

	@counter(stats.renders, [1])
	def render(self):
		self.renderer.render()

	#@timer_counter(stats.stateupdatecount, stats.stateupdatetime)
	@timer_counter(stats.stateupdatetc, [1])
	def update(self, y_im, y_flow, y_m):
		(Hz, Hz_components) = self._jacobian_multi(y_im, y_flow, y_m)
		if self.sparse:
			#HTH = self._hessian_sparse(y_im, y_flow, y_m)
			HTH = self._hessian_sparse_multi(y_im, y_flow, y_m)
		else: 
			HTH = self._hessian(y_im, y_flow, y_m)
		return (Hz, HTH, Hz_components)

	@timer_counter(stats.jacobianrenderstc, stats.jacinc)
	def _jacobian_multi(self, y_im, y_flow, y_m, deltaX = 2):
		Hz = np.zeros((self.size(),1))
		Hz_components = np.zeros((self.size(),4))

		#Perturb groups of vertices
		#This loop is ~32 renders, instead of ~280, for hydra1 synthetic mesh of 35 nodes
		for idx, e in enumerate(self.E):
			self.refresh(idx) 
			self.render()
			#Set reference image to unperturbed images
			self.renderer.initjacobian(y_im, y_flow, y_m)
			for i in range(2):
				for j in range(2):
					offset = i+2*self.N*j 
					ee = offset + 2*e 
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

		self.refresh() 
		self.render()
		return (Hz, Hz_components)

	def _jacobian(self, y_im, y_flow, y_m, deltaX = 2):
		Hz = np.zeros((self.size(),1))
		Hz_components = np.zeros((self.size(),4))
		self.refresh() 
		self.render()
		#Set reference image to unperturbed images
		self.renderer.initjacobian(y_im, y_flow, y_m)
		for idx in range(self.size()):
			self.X[idx,0] += deltaX
			self.refresh()
			self.render()
			(hz, hzc) = self.renderer.jz(self)
			Hz[idx,0] = hz/deltaX
			Hz_components[idx,:] = hzc/deltaX
			self.X[idx,0] -= deltaX

			self.X[idx,0] -= deltaX
			self.refresh()
			self.render()
			(hz, hzc) = self.renderer.jz(self)
			Hz[idx,0] -= hz/deltaX
			Hz_components[idx,:] -= hzc/deltaX
			self.X[idx,0] += deltaX
			Hz[idx,0] = Hz[idx,0]/2
			Hz_components[idx,:] = Hz_components[idx,:]/2
		self.refresh() 
		self.render()
		return (Hz, Hz_components)

	@timer_counter(stats.hessianrenderstc, stats.hessinc)
	def _hessian(self, y_im, y_flow, y_m, deltaX = 2):
		HTH = np.zeros((self.size(),self.size()))
		self.refresh() 
		self.render()
		#Set reference image to unperturbed images
		self.renderer.initjacobian(y_im, y_flow, y_m)
		#Very inefficient... for now 
		for i in range(self.size()):
			for j in range(i, self.size()):
				hij = self.renderer.j(self, deltaX, i, j)
				HTH[i,j] = hij/deltaX/deltaX
				#Fill in the other triangle
				HTH[j,i] = HTH[i,j]
		self.refresh() 
		self.render()
		return HTH

	@timer_counter(stats.hessianrenderstc, stats.hessincsparse)
	def _hessian_sparse_multi(self, y_im, y_flow, y_m, deltaX = 2):
		HTH = np.zeros((self.size(),self.size()))

		for idx, e in enumerate(self.E_hessian):
			self.refresh(idx, hess = True) 
			self.render()
			#Set reference image to unperturbed images
			self.renderer.initjacobian(y_im, y_flow, y_m)
			ee = e.copy()
			eeidx = self.E_hessian_idx[idx]
			for i1 in range(2):
				for j1 in range(2):
					for i2 in range(2):
						for j2 in range(2):
							offset1 = i1+2*self.N*j1 
							offset2 = i2+2*self.N*j2 
							ee[:,0] = 2*e[:,0] + offset1 
							ee[:,1] = 2*e[:,1] + offset2 
							#Do the render
							(h, h_hist) = self.renderer.j_multi(self, deltaX, ee, idx, eeidx)
							#Unpack the answers into the hessian matrix
							h = h[h_hist > 0]
							qidx = self.Q[h_hist > 0]
							for idx2 in range(len(qidx)):
								q = qidx[idx2]
								q1 = 2*q[0]+i1+2*self.N*j1
								q2 = 2*q[1]+i2+2*self.N*j2
								HTH[q1,q2] = h[idx2]/deltaX/deltaX
								HTH[q2,q1] = HTH[q1,q2]

		self.refresh() 
		self.render()
		return HTH

	def _hessian_sparse(self, y_im, y_flow, y_m, deltaX = 2):
		HTH = np.zeros((self.size(),self.size()))
		self.refresh() 
		self.render()
		#Set reference image to unperturbed images
		self.renderer.initjacobian(y_im, y_flow, y_m)
		#Actually(!) here we only need compute this for vertices that are connected
		#! this will speed things up significantly. 
		#! also don't need to do for cross terms between vx and vy
		#This should become roughly linear... which makes the whole thing more
		#doable.........for instance, this will chop the time for square1 geometry
		#by a factor of ~4. Expect greater gains for larger geometries...
		for i in range(self.size()):
			for j in range(i, self.size()):
				if self.J[i,j] == 1:
					hij = self.renderer.j(self, deltaX, i, j)
				else:
					hij = 0.
				HTH[i,j] = hij/deltaX/deltaX
				#Fill in the other triangle
				HTH[j,i] = HTH[i,j]
		self.refresh() 
		self.render()
		return HTH

	def vertices(self):
		return self.X[0:(2*self.N)].reshape((-1,2))

	def velocities(self):
		return self.X[(2*self.N):].reshape((-1,2))

	def lengths(self):
		#Compute initial edge lengths...
		l = np.zeros((self.I,1))
		d = np.dot(self.K.T,self.vertices().reshape((-1,1)))
		for i in range(self.I):
			di = d[2*i:2*i+2,0]
			l[i,0] = np.sqrt(np.dot(di.T,di))
		return l

	def setforce(self,f):
		self.renderer.force = f 

class KalmanFilter:
	def __init__(self, distmesh, im, flow, cuda, vel = None, sparse = True, eps_F = 1, eps_Z = 1e-3, eps_J = 1e-3, eps_M = 1e-3):
		self.distmesh = distmesh
		self.N = distmesh.size()
		print 'Creating filter with ' + str(self.N) + ' nodes'
		self.state = KFState(distmesh, im, flow, cuda, vel=vel, sparse = sparse, eps_F = eps_F, eps_Z = eps_Z, eps_J = eps_J, eps_M = eps_M)
		self.predtime = 0
		self.updatetime = 0

	def __del__(self):
		self.state.__del__()

	def plotforces(self, overlay, imageoutput):
		sc = 2
		#Get original pt locations
		ox = self.orig_x[0:(2*self.N)].reshape((-1,2))
		#Get prediction location
		px = self.pred_x[0:(2*self.N)].reshape((-1,2))
		#Get template, flow and mask 'force'
		tv = self.tv[0:(2*self.N)].reshape((-1,2))
		fv = self.fv[0:(2*self.N)].reshape((-1,2))
		mv = self.mv[0:(2*self.N)].reshape((-1,2))
		blank = np.zeros(overlay.shape, dtype=np.uint8)
		blank[:,:,3] = 255
		overlay = cv2.addWeighted(overlay, 0.5, blank, 0.5, 0)
		#Resize image
		overlay = cv2.resize(overlay, (0,0), fx = sc, fy = sc)
		for idx in range(self.N):
			cv2.arrowedLine(overlay, (sc*int(ox[idx,0]),sc*int(ox[idx,1])),\
			 (sc*int(px[idx,0]),sc*int(px[idx,1])), (255,255,255, 255), thickness = 2)
			cv2.arrowedLine(overlay, (sc*int(px[idx,0]),sc*int(px[idx,1])),\
			 (sc*int(px[idx,0]+10*tv[idx,0]),sc*int(px[idx,1]+10*tv[idx,1])), (255,0,0, 255), thickness = 2)
			cv2.arrowedLine(overlay, (sc*int(px[idx,0]),sc*int(px[idx,1])),\
			 (sc*int(px[idx,0]+10*fv[idx,0]),sc*int(px[idx,1]+10*fv[idx,1])), (0,255,0, 255), thickness = 2)
			cv2.arrowedLine(overlay, (sc*int(px[idx,0]),sc*int(px[idx,1])),\
			 (sc*int(px[idx,0]+10*mv[idx,0]),sc*int(px[idx,1]+10*mv[idx,1])), (0,0,255, 255), thickness = 2)

		font = cv2.FONT_HERSHEY_SIMPLEX
		#cv2.putText(img,'Hello World!',(10,500), font, 1,(255,255,255),2)
		legendtext = 'red = mask force\ngreen = flow force\nblue = template force\nwhite = prediction'
		x0, y0 = (20,20)
		dy = 20
		for i, line in enumerate(legendtext.split('\n')):
			y = y0 + i*dy
			cv2.putText(overlay, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255,255))
		
		#cv2.putText(overlay, legendtext, (20, 20), font, 1, (255, 255, 255), 2)
		fn = './' + imageoutput + '_forces_' + strftime("%Y-%m-%d_%H:%M:%S", gmtime()) + '.png'
		cv2.imwrite(fn, overlay)

	def compute(self, y_im, y_flow, y_m, maskflow = True, imageoutput = None):
		self.state.renderer.update_frame(y_im, y_flow, y_m)
		#Mask optic flow frame by contour of y_im
		if maskflow is True:
			y_flowx_mask = np.multiply(y_m, y_flow[:,:,0])
			y_flowy_mask = np.multiply(y_m, y_flow[:,:,1])
			y_flow_mask = np.dstack((y_flowx_mask, y_flowy_mask))
		else:
			y_flow_mask = y_flow
		pt = timeit(self.predict, number = 1)
		ut = timeit(lambda: self.update(y_im, y_flow_mask, y_m), number = 1)
		self.predtime += pt
		self.updatetime += ut

		print 'Current state:', self.state.X.T
		print 'Prediction time:', pt 
		print 'Update time: ', ut 
		#Save state of each frame
		if imageoutput is not None:
			overlay = self.state.renderer.screenshot(saveall=True, basename = imageoutput)
			#Compute error between predicted image and actual frames
			self.plotforces(overlay, imageoutput)
		return self.error(y_im, y_flow, y_m)

	@timer(stats.statepredtime)
	def predict(self):
		print '-- predicting'
		#import rpdb2 
		#rpdb2.start_embedded_debugger("asdf")

		X = self.state.X 
		self.orig_x = X.copy()
		F = self.state.F 
		Weps = self.state.Weps
		W = self.state.W 

		#Prediction equations 
		self.state.X = np.dot(F,X)
		self.pred_x = self.state.X.copy()
		self.state.W = np.dot(F, np.dot(W,F.T)) + Weps 
		#print np.sum(self.state.velocities())

	def size(self):
		#State space size
		return self.N*4

	def update(self, y_im, y_flow, y_m):
		#import rpdb2 
		#rpdb2.start_embedded_debugger("asdf")
		print '-- updating'
		X = self.state.X
		W = self.state.W
		#eps_Z = self.state.eps_Z
		(Hz, HTH, Hz_components) = self.state.update(y_im, y_flow, y_m)
		invW = np.linalg.inv(W) + HTH
		W = np.linalg.inv(invW)
		self.state.X = X + np.dot(W,Hz)
		self.state.W = W 

		#Determine updates per component for diagnostics 
		self.tv = np.dot(W, np.squeeze(Hz_components[:,0]))
		self.fv = np.dot(W, np.squeeze(Hz_components[:,1]+Hz_components[:,2]))
		self.mv = np.dot(W, np.squeeze(Hz_components[:,3]))

	def error(self, y_im, y_flow, y_m):
		#Compute error of current state and current images and flow data
		return self.state.renderer.error(self.state, y_im, y_flow, y_m)

class IteratedKalmanFilter(KalmanFilter):
	def __init__(self, distmesh, im, flow, cuda, sparse = True, nI = 10, eps_F = 1, eps_Z = 1e-3, eps_J = 1e-3, eps_M = 1e-3):
		KalmanFilter.__init__(self, distmesh, im, flow, cuda, sparse = sparse, eps_F = eps_F, eps_Z = eps_Z, eps_J = eps_J, eps_M = eps_M)
		self.nI = nI
		self.reltol = 1e-4

	def update(self, y_im, y_flow, y_m):
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

		eold = 0
		enew = 0
		conv = False

		#eps_Z = self.state.eps_Z
		for i in range(self.nI):
			sys.stdout.write('   IEKF K = %d\n'%i)
			sys.stdout.flush()
			(Hz, HTH, Hz_components) = self.state.update(y_im, y_flow, y_m)
			invW = invW_orig + HTH
			W = np.linalg.inv(invW)
			X = X_orig + np.dot(W,Hz) - np.dot(W,np.dot(HTH,X_orig - X))
			#Original version: seems quite wrong/different and yet gives OK results...
			#X = X + np.dot(W,Hz)
			self.state.X = X
			self.state.W = W 
			e_im, e_fx, e_fy, e_m, fx, fy = self.error(y_im, y_flow, y_m)
			enew = np.sqrt(e_im*e_im + e_fx*e_fx + e_fy*e_fy + e_m*e_m)
			sys.stdout.write('-- e_im: %d, e_fx: %d, e_fy: %d, e_m: %d\n'%(e_im, e_fx, e_fy, e_m))
			if abs(enew-eold)/enew < self.reltol:
				conv = True
				print 'Reached error tolerance.'
				break 
			eold = enew

		#Determine updates per component for diagnostics 
		self.tv = np.dot(W, np.squeeze(Hz_components[:,0]))
		self.fv = np.dot(W, np.squeeze(Hz_components[:,1]+Hz_components[:,2]))
		self.mv = np.dot(W, np.squeeze(Hz_components[:,3]))

		if conv is False:
			print 'Reached max iterations.'

#Iterated mass-spring Kalman filter
class IteratedMSKalmanFilter(IteratedKalmanFilter):
	def __init__(self, distmesh, im, flow, cuda, sparse = True, nI = 10, eps_F = 1, eps_Z = 1e-3, eps_J = 1e-3, eps_M = 1e-3):
		IteratedKalmanFilter.__init__(self, distmesh, im, flow, cuda, sparse = sparse, eps_F = eps_F, eps_Z = eps_Z, eps_J = eps_J, eps_M = eps_M)
		#Mass of vertices
		self.M = 1
		#Spring stiffness
		self.kappa = -1
		self.deltat = 0.1
		self.maxiter = 1000
		self.tol = 1e-4
		#Force equation
		self.force = lambda l1, l2: -self.kappa*(l1-l2)
		self.state.setforce(self.force)

	@timer(stats.statepredtime)
	def predict(self):
		print '-- predicting'
		X = self.state.X 
		self.orig_x = X.copy()

		#Generate linearized Jacobian F to update self.state.W
		F = self._dfdx()
		Weps = self.state.Weps
		W = self.state.W 

		#Use Newton's method to solve for new state self.state.X
		self._newton()
		self.pred_x = self.state.X.copy()
		self.state.W = np.dot(F, np.dot(W, F.T)) + Weps 

	def _jacobian(self):
		#Here we compute the Jacobian from linearizing around the current
		#state estimate. Note that this is slightly different from the Jacobian
		#evaluated at the future state, which is how the dynamics are specified
		#i.e. implicitly: x_k = x_k-1 + f(x_k)
		#However the form of the dynamics in the Kalman filter are given in an
		#explicit form, so this is simpler. I'll try it and see if it makes a 
		#large difference.

		#We're computing:
		#dfdy = -K[diag_i \kappa_i (1-l_i0/|d_i|)\otimes I_2] K^T
		# 	    -K diag(d)[da_i \kappa_i l_i0 /{|d_i|^3} y^T [(K^T)_i]^T (K^T)_i \otimes 1_{2x1}]

		n = self.state.N
		I = self.state.I 
		kappa = self.kappa 
		l0 = self.state.l0 
		K = self.state.K 
		y = self.state.vertices().reshape((-1,1))

		d = np.dot(K.T,y)
		l = self.state.lengths()
		e = np.eye(2*n)
		e2 = np.eye(2)
		o = np.ones((2,1))
		dfdy = np.zeros((2*n,2*n))

		k = np.diag(np.array([kappa*(1-l0[i,0]/l[i,0]) for i in range(I)]))
		k = np.kron(k, e2)

		dk = np.zeros((I, 2*n))
		for i in range(I):
			KTi = K.T[2*i:2*i+2,:]
			dk[i,:] = kappa*l0[i,0]/np.power(l[i,0],3)*np.dot(y.T,np.dot(KTi.T, KTi))
		dk = np.kron(dk, o)

		dfdy = -np.dot(K, np.dot(k, K.T)) - np.dot(K, np.dot(np.diagflat(d), dk))
		return dfdy 

	def _dfdx(self):
		n = self.state.N
		M = self.M 
		deltat = self.deltat
		e = np.eye(2*n)
		dfdy = self._jacobian()
		F = np.bmat([[e, deltat*e], [deltat*dfdy/M, e]])
		return F 

	def _dgdx(self):
		n = self.state.N
		M = self.M 
		deltat = self.deltat
		e = np.eye(2*n)
		dfdy = self._jacobian()
		G = np.bmat([[e, -deltat*e], [-deltat*dfdy/M, e]])
		return G 

	def _newton(self):
		#For small meshes we can invert the Jacobian directly...
		deltat = self.deltat 
		M = self.M 
		kappa = self.kappa 
		I = self.state.I 
		maxiter = self.maxiter 
		tol = self.tol 
		K = self.state.K 
		l0 = self.state.l0 
		e2 = np.eye(2)

		print "   using Newton's method"
		N = int(np.ceil(1/deltat))
		for i in range(N):
			print 't = ', deltat*i 
			n = 0
			x = self.state.X.copy()
			xp = x.copy()
			xo = np.zeros(x.shape)
			while n < maxiter and norm(xo-xp)>tol*norm(xp):
				xo = xp.copy()
				v = self.state.velocities().reshape((-1,1))
				y = self.state.vertices().reshape((-1,1))
				d = np.dot(K.T,y)
				l = self.state.lengths()
				k = np.diag(np.array([kappa*(1-l0[i,0]/l[i,0]) for i in range(I)]))
				k = np.kron(k, e2)
				L = np.dot(k,d)
				f = np.dot(K,L)
				dv = f/M 
				dg = np.bmat([[v],[dv]])
				g = xp - x - deltat*dg 
				dgdx = self._dgdx()
				xp = xp - inv(dgdx)*g 
				self.state.X = xp
				print '      iteration:', n, 'relative change:', norm(xo-xp)/norm(xp)
				n += 1

#Mass-spring Kalman filter
class MSKalmanFilter(KalmanFilter):
	def __init__(self, distmesh, im, flow, cuda, sparse = True, nI = 10, eps_F = 1, eps_Z = 1e-3, eps_J = 1e-3, eps_M = 1e-3):
		KalmanFilter.__init__(self, distmesh, im, flow, cuda, sparse = sparse, eps_F = eps_F, eps_Z = eps_Z, eps_J = eps_J, eps_M = eps_M)
		#Mass of vertices
		self.M = 1
		#Spring stiffness
		self.kappa = 1

	@timer(stats.statepredtime)
	def predict(self):
		print '-- predicting'
		X = self.state.X 
		self.orig_x = X.copy()
		F = self.state.F 
		Weps = self.state.Weps
		W = self.state.W 

		#Prediction equations 
		self.state.X = np.dot(F,X)
		self.pred_x = self.state.X.copy()
		self.state.W = np.dot(F, np.dot(W,F.T)) + Weps 

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
