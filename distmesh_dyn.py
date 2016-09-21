import numpy as np
import cv2
import distmesh as dm 
import cPickle 
import scipy.spatial as spspatial
import distmesh.mlcompat as ml
import distmesh.utils as dmutils

from imgproc import drawGrid

class DistMesh:
	def __init__(self, frame, h0 = 35, dptol = 0.01):
		self.bars = None 
		self.frame = frame 
		self.dptol = dptol
		self.h0 = h0
		self.N = 0
		self.fh = dm.huniform; 
		nx,ny = np.shape(frame)[0:2]
		self.nx = nx
		self.ny = ny
		self.bbox = (0, 0, ny, nx)
		self.ttol=.1
		self.Fscale=1.2
		self.deltat=.2
		self.geps=.001*h0;
		self.deps=np.sqrt(np.finfo(np.double).eps)*h0;
		self.densityctrlfreq = 1;
		self.k = 1.5
		self.maxiter = 100
		#Force law
		#self.F = lambda L: self.k/(L*(40-L))**2-1/400**2
		self.F = lambda L: -self.k*(L-h0)

	def plot(self):
		fc = self.frame.copy()
		if self.bars is not None:
			drawGrid(fc, self.p, self.bars)
			cv2.imshow('Current mesh',fc)
			k = cv2.waitKey(30) & 0xff

	def createMesh(self, ctrs, fd, frame, plot = False):
		self.frame = frame 
		pfix = None 
	
		# Extract bounding box
		xmin, ymin, xmax, ymax = self.bbox
		if pfix is not None:
			pfix = np.array(pfix, dtype='d')
		
		#1. Set up initial points 
		x, y = np.mgrid[xmin:(xmax+self.h0):self.h0,
										ymin:(ymax+self.h0*np.sqrt(3)/2):self.h0*np.sqrt(3)/2]
		x[:, 1::2] += self.h0/2                               # Shift even rows
		p = np.vstack((x.flat, y.flat)).T                # List of node coordinates
		
		# 2. Remove points outside the region, apply the rejection method
		a = fd(p)
		p = p[np.where(a<self.geps)]                          # Keep only d<0 points
		r0 = 1/self.fh(p)**2                                  # Probability to keep point
		p = p[np.random.random(p.shape[0])<r0/r0.max()]  # Rejection method
		if pfix is not None:
			p = ml.setdiff_rows(p, pfix)                 # Remove duplicated nodes
			pfix = ml.unique_rows(pfix); nfix = pfix.shape[0]
			p = np.vstack((pfix, p))                     # Prepend fix points
		else:
			nfix = 0
		N = p.shape[0]                                   # Number of points N
		self.N = N
		
		count = 0
		pold = float('inf')                              # For first iteration
		
		################################################################################
		#Mesh creation
		################################################################################
		
		while count < self.maxiter: 
			print 'DistMesh create count: %d/%d' % (count, self.maxiter) 
			count += 1
			# 3. Retriangulation by the Delaunay algorithm
			dist = lambda p1, p2: np.sqrt(((p1-p2)**2).sum(1))
			if (dist(p, pold)/self.h0).max() > self.ttol:          # Any large movement?
				pold = p.copy()                          # Save current positions
				self.delaunay = spspatial.Delaunay(p)
				t = self.delaunay.vertices       # List of triangles
				pmid = p[t].sum(1)/3                     # Compute centroids
				t = t[fd(pmid) < -self.geps]                  # Keep interior triangles
				# 4. Describe each bar by a unique pair of nodes
				bars = np.vstack((t[:, [0,1]],
									t[:, [1,2]],
									t[:, [2,0]]))          # Interior bars duplicated
				bars.sort(axis=1)
				bars = ml.unique_rows(bars)              # Bars as node pairs
				#Plot
				fc = frame.copy()
				if frame is not None:
					drawGrid(fc, p, bars)
					if plot:
						cv2.imshow('Initial mesh',fc)
						k = cv2.waitKey(30) & 0xff
						if k == 27:
							break
		
			# 6. Move mesh points based on bar lengths L and forces F
			barvec = p[bars[:,0]] - p[bars[:,1]]         # List of bar vectors
			L = np.sqrt((barvec**2).sum(1))              # L = Bar lengths
			hbars = self.fh(p[bars].sum(1)/2)
			L0 = 1.5*self.h0*np.ones_like(L); #(hbars*Fscale
						#*np.sqrt((L**2).sum()/(hbars**2).sum()))  # L0 = Desired lengths
		
			F = self.k*(L0-L)
			F[F<0] = 0#F[F<0]*.5#0                         # Bar forces (scalars)
			Fvec = F[:,None]/L[:,None].dot([[1,1]])*barvec # Bar forces (x,y components)
			Ftot = ml.dense(bars[:,[0,0,1,1]],
											np.repeat([[0,1,0,1]], len(F), axis=0),
											np.hstack((Fvec, -Fvec)),
											shape=(N, 2))
			Ftot[:nfix] = 0                              # Force = 0 at fixed points
			p += self.deltat*Ftot                             # Update node positions
		
			# 7. Bring outside points back to the boundary
			d = fd(p); ix = d>0                          # Find points outside (d>0)
			ddeps = 1e-1
			for idx in range(10):
				if ix.any():
					dgradx = (fd(p[ix]+[ddeps,0])-d[ix])/ddeps # Numerical
					dgrady = (fd(p[ix]+[0,ddeps])-d[ix])/ddeps # gradient
					dgrad2 = dgradx**2 + dgrady**2
					p[ix] -= (d[ix]*np.vstack((dgradx, dgrady))/dgrad2).T # Project
		
			# 8. Termination criterion: All interior nodes move less than dptol (scaled)
			if (np.sqrt((self.deltat*Ftot[d<-self.geps]**2).sum(1))/self.h0).max() < self.dptol:
				break
		
		self.p = p 
		self.t = t 
		self.bars = bars
		self.L = L 

	def updateMesh(self, ctrs, fd, frame_orig, pfix = None, n_iter = 20):
		deltat = 0.1
		xmin, ymin, xmax, ymax = self.bbox
		
		if pfix is not None:
			self.p = ml.setdiff_rows(self.p, pfix)
			pfix = ml.unique_rows(pfix)
			nfix = pfix.shape[0]
		else:
			nfix = 0
	
		N = self.p.shape[0]                                   # Number of points N
		pold = float('inf')                              # For first iteration
	
		################################################################################
		#Mesh updates
		################################################################################
		
		#self.delaunay = spspatial.Delaunay(self.p)
		#self.t = self.delaunay.vertices       # List of triangles
		for ii in range(n_iter):
			dist = lambda p1, p2: np.sqrt(((p1-p2)**2).sum(1))
			if (dist(self.p, pold)/self.h0).max() > self.ttol:          # Any large movement?
				pold = self.p.copy()                          # Save current positions
				pmid = self.p[self.t].sum(1)/3                     # Compute centroids
				self.t = self.t[fd(pmid) < -self.geps]                  # Keep interior triangles
				bars = np.vstack((self.t[:, [0,1]],
									self.t[:, [1,2]],
									self.t[:, [2,0]]))          # Interior bars duplicated
				bars.sort(axis=1)
				bars = ml.unique_rows(bars)              # Bars as node pairs
	
			barvec = self.p[bars[:,0]] - self.p[bars[:,1]]         # List of bar vectors
			L = np.sqrt((barvec**2).sum(1))              # L = Bar lengths
			hbars = self.fh(self.p[bars].sum(1)/2)
			L0 = 1.4*self.h0*np.ones_like(L);
		
			#F = self.k*(L0-L)
			#F[F<0] = 0                         # Bar forces (scalars)
			F = self.F(L) 
			Fvec = F[:,None]/L[:,None].dot([[1,1]])*barvec # Bar forces (x,y components)
			Ftot = ml.dense(bars[:,[0,0,1,1]],
											np.repeat([[0,1,0,1]], len(F), axis=0),
											np.hstack((Fvec, -Fvec)),
											shape=(N, 2))
			Ftot[:nfix] = 0                              # Force = 0 at fixed points
			#self.p += self.deltat*Ftot                             # Update node positions
			self.p += deltat*Ftot                             # Update node positions
		
			d = fd(self.p); ix = d>0                          # Find points outside (d>0)
			ddeps = 1e-1
			for idx in range(10):
				if ix.any():
					dgradx = (fd(self.p[ix]+[ddeps,0])-d[ix])/ddeps # Numerical
					dgrady = (fd(self.p[ix]+[0,ddeps])-d[ix])/ddeps # gradient
					dgrad2 = dgradx**2 + dgrady**2
					self.p[ix] -= (d[ix]*np.vstack((dgradx, dgrady))/dgrad2).T # Project	

		self.bars = bars
		self.L = L 

	def size(self):
		return self.N

	def save(self, fn_out):
		f = open(fn_out, 'wb')
		for obj in [self.N, self.bars, self.frame, self.dptol, self.nx, self.ny,\
		 self.bbox, self.ttol, self.Fscale, self.deltat, self.geps, self.deps,\
		 self.densityctrlfreq, self.k, self.maxiter, self.p, self.t, self.bars, self.L]:
			cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
		f.close()

	def load(self, fn_in):
		f = open(fn_in, 'rb')
		loaded_objects = []
		for i in range(19):
			loaded_objects.append(cPickle.load(f))
		f.close()
		[self.N, self.bars, self.frame, self.dptol, self.nx, self.ny,\
		 self.bbox, self.ttol, self.Fscale, self.deltat, self.geps, self.deps,\
		 self.densityctrlfreq, self.k, self.maxiter, self.p, self.t, self.bars,\
		 self.L] = loaded_objects