import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.spatial as spspatial
import distmesh as dm 
import distmesh.mlcompat as ml
import distmesh.utils as dmutils

import imgproc as ip
from imgproc import * 

cap = cv2.VideoCapture("./video/local_prop_cb_with_bud.avi")
#cap = cv2.VideoCapture("./video/20150306_GCaMP_Chl_EC_local_prop.avi")

ret, frame = cap.read()
frame_orig = frame.copy()
#img = cv2.imread('./video/testcontours.jpg')
threshold = 7

#Just try simple thresholding instead: (quick!, seems to work fine)
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ret1, mask = cv2.threshold(frame_gray, threshold, 255, cv2.THRESH_TOZERO)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
im2, c, h = cv2.findContours(mask2.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
ctrs = ip.Contours(c, h)

#Remove contours that are smaller than 40 square pixels, or that are above
#level one
changed = True
maxarea = 0
for (ct, level) in ctrs.traverse():
	maxarea = max(maxarea, cv2.contourArea(ct))
while changed:
	changed = False 
	for idx, (ct, level) in enumerate(ctrs.traverse()):
		area = cv2.contourArea(ct)
		if area < maxarea and level == 0:
			ctrs.remove(idx)
			changed = True 
			break 
		if area < 40:
			ctrs.remove(idx)
			changed = True
			break 
		if level > 1:
			ctrs.remove(idx)
			changed = True 
			break 

#Make the signed diff function
#All of them
fd = lambda p: dm.ddiff(ip.ppt(ctrs.contours[0], p, True), ip.ddunion(ctrs.contours[1:], p, True))
#One of them
#fd = lambda p: dm.dunion(ppt(ctrs.contours[2], p, True), ppt(ctrs.contours[1], p, True))
#One of them
#fd = lambda p: ip.ppt(ctrs.contours[5], p, True)


#The distance function we'll be using...
cv2.drawContours(frame, ctrs.contours, -1, (0,255,0), 3)
cv2.imshow('frame',frame)
k = cv2.waitKey(30) & 0xff

################################################################################
################################################################################

fh = dm.huniform; 
h0 = 25
nx,ny = np.shape(frame)[0:2]
bbox = (0, 0, ny, nx)
pfix = None 

#Set up tolerances
dptol=.01; ttol=.1; Fscale=1.2; deltat=.2; geps=.001*h0;
deps=np.sqrt(np.finfo(np.double).eps)*h0;
densityctrlfreq=30;

# Extract bounding box
xmin, ymin, xmax, ymax = bbox
if pfix is not None:
	pfix = np.array(pfix, dtype='d')

#1. Set up initial points 
x, y = np.mgrid[xmin:(xmax+h0):h0,
								ymin:(ymax+h0*np.sqrt(3)/2):h0*np.sqrt(3)/2]
x[:, 1::2] += h0/2                               # Shift even rows
p = np.vstack((x.flat, y.flat)).T                # List of node coordinates

# 2. Remove points outside the region, apply the rejection method
a = fd(p)
p = p[np.where(a<geps)]                          # Keep only d<0 points
r0 = 1/fh(p)**2                                  # Probability to keep point
p = p[np.random.random(p.shape[0])<r0/r0.max()]  # Rejection method
if pfix is not None:
	p = ml.setdiff_rows(p, pfix)                 # Remove duplicated nodes
	pfix = ml.unique_rows(pfix); nfix = pfix.shape[0]
	p = np.vstack((pfix, p))                     # Prepend fix points
else:
	nfix = 0
N = p.shape[0]                                   # Number of points N

count = 0
pold = float('inf')                              # For first iteration

while True:
	count += 1
	# 3. Retriangulation by the Delaunay algorithm
	dist = lambda p1, p2: np.sqrt(((p1-p2)**2).sum(1))
	if (dist(p, pold)/h0).max() > ttol:          # Any large movement?
		pold = p.copy()                          # Save current positions
		t = spspatial.Delaunay(p).vertices       # List of triangles
		pmid = p[t].sum(1)/3                     # Compute centroids
		t = t[fd(pmid) < -geps]                  # Keep interior triangles
		# 4. Describe each bar by a unique pair of nodes
		bars = np.vstack((t[:, [0,1]],
											t[:, [1,2]],
											t[:, [2,0]]))          # Interior bars duplicated
		bars.sort(axis=1)
		bars = ml.unique_rows(bars)              # Bars as node pairs
		# 5. Graphical output of the current mesh
		#Plot:
		frame = frame_orig.copy()
		#drawPoints(frame, p)
		drawGrid(frame, p, bars)
		cv2.imshow('frame',frame)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break

	# 6. Move mesh points based on bar lengths L and forces F
	barvec = p[bars[:,0]] - p[bars[:,1]]         # List of bar vectors
	L = np.sqrt((barvec**2).sum(1))              # L = Bar lengths
	hbars = fh(p[bars].sum(1)/2)
	L0 = 1.4*h0*np.ones_like(L); #(hbars*Fscale
				#*np.sqrt((L**2).sum()/(hbars**2).sum()))  # L0 = Desired lengths

	# Density control - remove points that are too close
	#if (count % densityctrlfreq) == 0 and (L0 > 2*L).any():
	#	ixdel = np.setdiff1d(bars[L0 > 2*L].reshape(-1), np.arange(nfix))
	#	p = p[np.setdiff1d(np.arange(N), ixdel)]
	#	N = p.shape[0]; pold = float('inf')
	#	continue

	F = L0-L; F[F<0] = 0                         # Bar forces (scalars)
	Fvec = F[:,None]/L[:,None].dot([[1,1]])*barvec # Bar forces (x,y components)
	Ftot = ml.dense(bars[:,[0,0,1,1]],
									np.repeat([[0,1,0,1]], len(F), axis=0),
									np.hstack((Fvec, -Fvec)),
									shape=(N, 2))
	Ftot[:nfix] = 0                              # Force = 0 at fixed points
	p += deltat*Ftot                             # Update node positions

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
	if (np.sqrt((deltat*Ftot[d<-geps]**2).sum(1))/h0).max() < dptol:
		break

plt.show()