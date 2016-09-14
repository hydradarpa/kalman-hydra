#################################
#Extended iterated Kalman filter#
#################################

#lansdell. Feb 9th 2016

import numpy as np
import cv2 

from distmesh_dyn import DistMesh
from renderer import Renderer
from imgproc import findObjectThreshold

import pdb
import logging 
import sys
from time import gmtime, strftime
from timeit import timeit 
import time 
from numpy.linalg import norm, inv 

import matplotlib.pyplot as plt

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
		self.niter = 0
		self.meshpts = 0
		self.gridsize = 0
		self.jacobianpartitions = 0
		self.hessianpartitions = 0
		self.nzj = 0

		self.jacobianrenderstc[0] = 0
		self.jacobianrenderstc[1] = 0
		self.hessianrenderstc[0] = 0
		self.hessianrenderstc[1] = 0
		self.renders[0] = 0
		self.statepredtime[0] = 0
		self.stateupdatetc[0] = 0
		self.stateupdatetc[1] = 0
		self.hessinc[0] = 0
		self.jacinc[0] = 0
		self.hessincsparse[0] = 0

#Decorators to count update times... uses time module so not as accurate as profiling
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
			#if inc[0] == 1:
			#	print 'Counting updatetc'
			return ret
		return func_wrapper
	return counter_wrapper

#'Global variable' not the best solution... 
stats = Statistics()

class KFState:
	def __init__(self, distmesh, im, flow, cuda, eps_F = 1, eps_Z = 1e-3, eps_J = 1e-3, eps_M = 1e-3, vel = None, sparse = True, multi = True, alpha = 0.3):

		self.multi = multi 
		self.alpha = alpha

		print 'multiperturbation rendering:', multi 
		print 'edge length adaptation rate:', alpha
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
		nz = abs(self.sineface) > 0.3
		print 'Removing %d faces for being too flat' % np.sum(nz == 0)
		self.sineface = self.sineface[nz]
		self.ori = self.ori[nz]
		self.tri = self.tri[nz]
		distmesh.t = self.tri 

		#Remove distmesh.bars and distmesh.L that are orphaned by faces being removed
		allbars = []
		for [v1, v2, v3] in self.tri:
			allbars += [[min(v1, v2), max(v1,v2)]]
			allbars += [[min(v2, v3), max(v2,v3)]]
			allbars += [[min(v1, v3), max(v1,v3)]]
		allbars = np.array(allbars)
		allbars = unique2d(allbars)
		t_allbars = [tuple(i) for i in allbars]
		t_dmbars = [tuple(i) for i in distmesh.bars]
		nz = np.array([i in t_allbars for i in t_dmbars])
		distmesh.bars = distmesh.bars[nz, :]
		distmesh.L = distmesh.L[nz]

		#Form state vector
		self.X = np.vstack((self._ver.reshape((-1,1)), self._vel.reshape((-1,1))))
		e = np.eye(2*self.N)
		z = np.zeros((2*self.N,2*self.N))
		self.F = np.bmat([[e, e], [z, e]])
		self.Weps = eps_F * np.bmat([[e/4, e/2], [e/2, e]])

		#self.W = np.eye(self._vel.shape[0]*4)
		#Initial position is certain, velocity is uncertain
		self.W = np.bmat([[1e-2*e,z],[z,e]])

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
		self.L = self.lengths()

		########################################################################
		#Compute partitions for jacobian multi-rendering########################
		########################################################################
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

		#while len(Q) > 0:
		#	#print 'Outer loop'
		#	P = []
		#	e = []
		#	while len(Q) > 0:
		#		#print '  * Inner loop'
		#		#Current vertices
		#		q = Q[0]
		#
		#		#Things connected to current vertex
		#		p = np.nonzero(Jv[q,:])[0]
		#		p = np.setdiff1d(p, q)
		#
		#		#Add current vertex
		#		e += [q]
		#
		#		#Add things connected to current vertex to the 'do later' list
		#		P = np.intersect1d(np.union1d(P, p), A)
		#
		#		A = np.setdiff1d(A, q)
		#		#Remove q and p from Q
		#		Q = np.setdiff1d(Q, p)
		#		Q = np.setdiff1d(Q, q)
		#	Q = P
		#	self.E += [e]			

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
				p_self1 = np.all(Q == [q[0],q[0]],1)
				p_self2 = np.all(Q == [q[1],q[1]],1)
				p_all_idx = p_all1 + p_all2 + p_self1 + p_self2
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
			if len(e.shape) == 1:
				e = np.reshape(e, (-1,2))
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
			#For each triangle, find if any of its vertices are mentioned in e,
			#give it a color...
			if len(e.shape) < 2:
				e = np.reshape(e, (-1,2))
			for i, nodes in enumerate(e):
				#print nodes 
				n1, n2 = nodes 
				for j, t in enumerate(self.tri):
					if (n1 in t) or (n2 in t):
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

	def update_orientation(self):
		ver = self.vertices()
		a = ver[self.tri[:,1],:] - ver[self.tri[:,0],:]
		b = ver[self.tri[:,2],:] - ver[self.tri[:,0],:]
		self.ori = np.sign(np.cross(a,b))
		self.sineface = np.zeros(len(a))
		for i in range(len(a)):
			self.sineface[i] = np.cross(a[i,:],b[i,:])/(np.linalg.norm(a[i,:])*np.linalg.norm(b[i,:]))

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
		logging.debug("----KFState update")
		if self.multi:
			(Hz, Hz_components) = self._jacobian_multi(y_im, y_flow, y_m)
		else:
			(Hz, Hz_components) = self._jacobian(y_im, y_flow, y_m)
		if self.sparse:
			if self.multi:
				HTH = self._hessian_sparse_multi(y_im, y_flow, y_m)
			else:
				HTH = self._hessian_sparse(y_im, y_flow, y_m)
		else:
			HTH = self._hessian(y_im, y_flow, y_m)
		return (Hz, HTH, Hz_components)

	@timer_counter(stats.jacobianrenderstc, stats.jacinc)
	def _jacobian_multi(self, y_im, y_flow, y_m, deltaX = 2):
		logging.debug("------KFState multi jacobian")
		Hz = np.zeros((self.size(),1))
		Hz_components = np.zeros((self.size(),4))

		#Perturb groups of vertices
		#This loop is ~32 renders, instead of ~280, for hydra1 synthetic mesh of 35 nodes
		for idx, e in enumerate(self.E):
			logging.debug("--------Perturbing partition %d"%idx)
			self.refresh(idx) 
			self.render()
			#Set reference image to unperturbed images
			logging.debug("--------initjacobian")
			self.renderer.initjacobian(y_im, y_flow, y_m)
			for i in range(2):
				for j in range(2):
					logging.debug("--------Rendering perturbation")
					offset = i+2*self.N*j 
					#print e 
					ee = offset + 2*np.array(e, dtype=np.int)
					#print ee 
					self.X[ee,0] = np.squeeze(self.X[ee, 0] + deltaX)
					self.refresh(idx)
					self.render()
					(hz, hzc) = self.renderer.jz_multi(self)
					Hz[ee,0] = np.squeeze(hz[e]/deltaX)
					Hz_components[ee,:] = hzc[e,:]/deltaX
					
					self.X[ee,0] = np.squeeze(self.X[ee,0] - 2*deltaX)
					self.refresh(idx)
					self.render()
					(hz, hzc) = self.renderer.jz_multi(self)
					Hz[ee,0] = np.squeeze(Hz[ee,0] - hz[e].T/deltaX)
					Hz_components[ee,:] -= hzc[e,:]/deltaX
					self.X[ee,0] = np.squeeze(self.X[ee,0] + deltaX)
					
					Hz[ee,0] = Hz[ee,0]/2
					Hz_components[ee,:] = Hz_components[ee,:]/2

		logging.debug("------Finished")
		self.refresh() 
		self.render()
		return (Hz, Hz_components)

	def _jacobian(self, y_im, y_flow, y_m, deltaX = 2):
		logging.debug("------KFState single jacobian")
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
		logging.debug("------KFState single dense Hessian")
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
		logging.debug("------KFState multi sparse Hessian")
		HTH = np.zeros((self.size(),self.size()))
		HTH_c = np.zeros((4, self.size(), self.size()))

		for idx, e in enumerate(self.E_hessian):
			logging.debug("-------- Perturbing partition %d"%idx)
			self.refresh(idx, hess = True) 
			self.render()
			#Set reference image to unperturbed images
			logging.debug("-------- initjacobian")
			self.renderer.initjacobian(y_im, y_flow, y_m)
			ee = e.copy()
			eeidx = self.E_hessian_idx[idx]
			#print e 
			for i1 in range(2):
				for j1 in range(2):
					for i2 in range(2):
						for j2 in range(2):
							logging.debug("-------- Rendering")
							offset1 = i1+2*self.N*j1 
							offset2 = i2+2*self.N*j2 
							ee[:,0] = 2*e[:,0] + offset1 
							ee[:,1] = 2*e[:,1] + offset2 
							#Do the render
							(h, h_hist, hcomp) = self.renderer.j_multi(self, deltaX, ee, idx, eeidx)
							#Unpack the answers into the hessian matrix
							h = h[h_hist > 0]
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

		logging.debug("------Finished")
		self.refresh() 
		self.render()
		return HTH

	def _hessian_sparse(self, y_im, y_flow, y_m, deltaX = 2):
		logging.debug("------KFState single sparse Hessian")
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
	def __init__(self, distmesh, im, flow, cuda, vel = None, sparse = True, multi = True, eps_F = 1, eps_Z = 1e-3, eps_J = 1e-3, eps_M = 1e-3, alpha = 0.3):
		self.distmesh = distmesh
		self.N = distmesh.size()
		print 'Creating filter with ' + str(self.N) + ' nodes'
		self.state = KFState(distmesh, im, flow, cuda, vel=vel, sparse = sparse, multi = multi, eps_F = eps_F, eps_Z = eps_Z, eps_J = eps_J, eps_M = eps_M, alpha = alpha)
		self.predtime = 0
		self.updatetime = 0
		self.adapttime = 0

	def __del__(self):
		self.state.__del__()

	def save(self, fn_out):
		return 

	def plotforces(self, overlay, imageoutput):
		sc = 10
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
		legendtext = 'red = mask force\ngreen = flow force\nblue = template force\nwhite = prediction\nscale factor = %d'%sc
		x0, y0 = (20,20)
		dy = 20
		for i, line in enumerate(legendtext.split('\n')):
			y = y0 + i*dy
			cv2.putText(overlay, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255,255))
		
		#cv2.putText(overlay, legendtext, (20, 20), font, 1, (255, 255, 255), 2)
		fn = './' + imageoutput + '_forces_' + strftime("%Y-%m-%d_%H:%M:%S", gmtime()) + '.png'
		cv2.imwrite(fn, overlay)

	def compute(self, y_im, y_flow, y_m, maskflow = True, imageoutput = None):

		logging.info("Computing frame")
		self.state.renderer.update_frame(y_im, y_flow, y_m)
		#Mask optic flow frame by contour of y_im
		if maskflow is True:
			y_flowx_mask = np.multiply(y_m, y_flow[:,:,0])
			y_flowy_mask = np.multiply(y_m, y_flow[:,:,1])
			y_flow_mask = np.dstack((y_flowx_mask, y_flowy_mask))
		else:
			y_flow_mask = y_flow
		pt = timeit(self.predict, number = 1)
		jt = timeit(lambda: self.projectmask(y_m), number = 1)
		ut = timeit(lambda: self.update(y_im, y_flow_mask, y_m), number = 1)
		at = timeit(self.adaptlength, number = 1)
		logging.info("Finished computing frame")
		self.predtime += pt
		self.updatetime += ut
		self.adapttime += at 

		print 'Current state:', self.state.X.T
		print 'Prediction time:', pt 
		print 'Projection time:', jt 
		print 'Update time: ', ut 
		print 'Length adjustment time:', at

		#Update history of tracking

		#Save state of each frame
		if imageoutput is not None:
			overlay = self.state.renderer.screenshot(saveall=True, basename = imageoutput)
			#Compute error between predicted image and actual frames
			self.plotforces(overlay, imageoutput)
		return self.error(y_im, y_flow, y_m)

	def adaptlength(self):
		#Adjust length of edges based on how stretched they currently are
		#and an 'adjustment' rate alpha

		minlength = 5
		logging.info("--Adapt lengths")
		l = self.state.lengths()
		l0 = self.state.l0
		alpha = self.state.alpha
		l0 += (l-l0)*alpha

		l0[l0 < minlength] = minlength

		self.state.l0 = l0


	@timer(stats.statepredtime)
	def predict(self):
		print '-- predicting'
		logging.info("--KF prediction")
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

	def projectmask(self, y_m):
		print '-- projecting outliers onto contour'
		logging.info("--KF projection")
		#Project vertices onto boundary 
		ddeps = 1e-1
		(mask2, ctrs, fd) = findObjectThreshold(y_m, threshold = 0.5)
		p = self.state.vertices()
		p_orig = p.copy()

		#Find vertices outside of mask and project these onto contour
		for idx in range(10):
			d = fd(p)
			ix = d > 1
			if ix.any():
				#First order differencing
				#dgradx = (fd(p[ix]+[ddeps,0])-d[ix])/(ddeps) # Numerical
				#dgrady = (fd(p[ix]+[0,ddeps])-d[ix])/(ddeps) # gradient
				#Central differencing
				dgradx = (fd(p[ix]+[ddeps,0])-fd(p[ix]-[ddeps,0]))/(2*ddeps) # Numerical
				dgrady = (fd(p[ix]+[0,ddeps])-fd(p[ix]-[0,ddeps]))/(2*ddeps) # gradient
				
				dgrad2 = dgradx**2 + dgrady**2
				p[ix] -= (d[ix]*np.vstack((dgradx, dgrady))/dgrad2).T # Project

		#Find _outer_ vertices inside of mask and project these onto contour
		#Get rendered mask

		rend_mask = self.state.renderer.rendermask()[:,:,2]
		#rend_mask = np.flipud(rend_mask)

		#For each vertex, find if its on the border of the mask... 
		border = np.zeros(self.N, dtype = bool)
		for idx, pi in enumerate(p):
			pii = pi.astype(int)
			i = np.zeros(8, dtype = bool)
			i[0] = rend_mask[pii[0,1]+2, pii[0,0]+2]
			i[1] = rend_mask[pii[0,1]+2, pii[0,0]-2]
			i[2] = rend_mask[pii[0,1]-2, pii[0,0]+2]
			i[3] = rend_mask[pii[0,1]-2, pii[0,0]-2]
			i[4] = rend_mask[pii[0,1],   pii[0,0]+2]
			i[5] = rend_mask[pii[0,1],   pii[0,0]-2]
			i[6] = rend_mask[pii[0,1]+2, pii[0,0]]
			i[7] = rend_mask[pii[0,1]-2, pii[0,0]]
			if (np.sum(i) < 8) and (np.sum(i) > 0):
				border[idx] = 1

		#Plot the mask...
		#prend_mask = rend_mask.copy()
		#for pi in p:
		#	pii = pi.astype(int)
		#	prend_mask[pii[0,1],   pii[0,0]] = 300
		#	prend_mask[pii[0,1]+1, pii[0,0]] = 300
		#	prend_mask[pii[0,1],   pii[0,0]+1] = 300
		#	prend_mask[pii[0,1]+1, pii[0,0]+1] = 300
		#plt.imshow(prend_mask)
		#plt.colorbar()
		#plt.show()

		d = fd(p)
		ix = (d < -1) * border 
		for idx in range(1):
			if ix.any():
				#First order differencing
				#dgradx = (fd(p[ix]+[ddeps,0])-d[ix])/(ddeps) # Numerical
				#dgrady = (fd(p[ix]+[0,ddeps])-d[ix])/(ddeps) # gradient
				#Central differencing
				dgradx = (fd(p[ix]+[ddeps,0])-fd(p[ix]-[ddeps,0]))/(2*ddeps) # Numerical
				dgrady = (fd(p[ix]+[0,ddeps])-fd(p[ix]-[0,ddeps]))/(2*ddeps) # gradient
				dgrad2 = dgradx**2 + dgrady**2
				p[ix] -= (d[ix]*np.vstack((dgradx, dgrady))/dgrad2).T # Project

		#Write changes
		self.state.X[0:(2*self.N)] = np.reshape(p, (-1,1))

		#Update velocities also... or else it might crash...
		self.state.X[(2*self.N):] += np.reshape(p - p_orig, (-1,1))

	def update(self, y_im, y_flow, y_m):
		#import rpdb2 
		#rpdb2.start_embedded_debugger("asdf")
		print '-- updating'
		logging.info("--KF update")
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
	#def __init__(self, distmesh, im, flow, cuda, sparse = True, nI = 10, eps_F = 1, eps_Z = 1e-3, eps_J = 1e-3, eps_M = 1e-3):
	def __init__(self, distmesh, im, flow, cuda, sparse = True, multi = True, nI = 10, eps_F = 1e-3, eps_Z = 1e-3, eps_J = 1e-3, eps_M = 1e10, alpha = 0.3):
		KalmanFilter.__init__(self, distmesh, im, flow, cuda, sparse = sparse, multi = multi, eps_F = eps_F, eps_Z = eps_Z, eps_J = eps_J, eps_M = eps_M, alpha = alpha)
		self.nI = nI
		self.reltol = 1e-4

	def update(self, y_im, y_flow, y_m):
		#import rpdb2
		#rpdb2.start_embedded_debugger("asdf")
		#np.set_printoptions(threshold = 'nan', linewidth = 150, precision = 1)
		print '-- updating'
		logging.info("--KF update")
		X = self.state.X
		X_orig = X.copy()
		X_old = X.copy()
		W = self.state.W
		W_orig = W.copy()
		W_old = W.copy()
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

			#Check still valid position, if not, use most recent valid pos
			self.state.update_orientation()
			if np.any(self.state.ori < 0):
				self.state.X = X_old 
				self.state.W = W_old 
				print '** Mesh inconsistent ** Reverting to last good state and continuing'
				break 

			e_im, e_fx, e_fy, e_m, fx, fy = self.error(y_im, y_flow, y_m)
			enew = np.sqrt(e_im*e_im + e_fx*e_fx + e_fy*e_fy + e_m*e_m)
			sys.stdout.write('-- e_im: %d, e_fx: %d, e_fy: %d, e_m: %d\n'%(e_im, e_fx, e_fy, e_m))
			if abs(enew-eold)/enew < self.reltol:
				conv = True
				print 'Reached error tolerance.'
				break 
			eold = enew

			X_old = X.copy()
			W_old = W.copy()

		#Determine updates per component for diagnostics 
		self.tv = np.dot(W, np.squeeze(Hz_components[:,0]))
		self.fv = np.dot(W, np.squeeze(Hz_components[:,1]+Hz_components[:,2]))
		self.mv = np.dot(W, np.squeeze(Hz_components[:,3]))

		if conv is False:
			print 'Reached max iterations.'

#Iterated mass-spring Kalman filter
class IteratedMSKalmanFilter(IteratedKalmanFilter):
	def __init__(self, distmesh, im, flow, cuda, sparse = True, multi = True, nI = 10, eps_F = 1e-1, eps_Z = 1e-3, eps_J = 1, eps_M = 1, alpha = 0.3):
		#def __init__(self, distmesh, im, flow, cuda, sparse = True, nI = 10, eps_F = 1, eps_Z = 1e-3, eps_J = 1e-3, eps_M = 10):
		IteratedKalmanFilter.__init__(self, distmesh, im, flow, cuda, sparse = sparse, multi = multi, eps_F = eps_F, eps_Z = eps_Z, eps_J = eps_J, eps_M = eps_M, nI = nI, alpha = alpha)
		#Mass of vertices
		self.M = 1
		#Spring stiffness
		self.kappa = -1
		self.deltat = 0.05
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
		#Should the deltat be here... since we're time stepping over units of 1, not deltat
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
