import numpy as np
import numpy.matlib as npml
import matplotlib as mpl 
import cv2
import distmesh as dm 
import scipy.spatial as spspatial
import distmesh.mlcompat as ml
import distmesh.utils as dmutils
import functions.video
from functions.common import draw_str

from matplotlib import pyplot as plt

class Contours:
	def __init__(self, contours, hierarchy = None):
		self.contours = contours
		#If hierarchy specified then we create an iterator to iterate over it...
		self.hierarchy = hierarchy
		self.nC = len(contours)

	def traverse(self):
		levels = np.zeros((self.nC, 1)).tolist()
		for idx,ct in enumerate(self.contours):
			if self.hierarchy is not None: 
				parent = self.hierarchy[0,idx,3]
				if parent == -1:
					level = 0 
				else:
					level = levels[parent]+1 
				levels[idx] = level			
			else:
				level = None 
			yield (ct, level)

	def remove(self, idx):
		#Remove a contour and all of its children
		toremove = []
		#Do recursively
		self._remove(toremove, idx, 1)
		#We have the nodes to delete, now we remove these from contours list, and 
		#reindex the hierarchy variable
		nR = len(toremove)
		minidx = min(toremove)
		maxidx = max(toremove)
		contours = []
		hierarchy = []
		for idx in range(self.nC):
			if idx not in toremove:
				contours.append(self.contours[idx])
				hierarchy.append(self.hierarchy[0,idx,:].tolist())
		self.contours = contours
		hierarchy = np.array(hierarchy, dtype = 'int32')
		hierarchy[np.where(np.logical_and(hierarchy >= minidx, hierarchy <= maxidx))] = -1 
		hierarchy[np.where(hierarchy > maxidx)] = hierarchy[np.where(hierarchy > maxidx)] - nR 
		self.hierarchy = hierarchy[np.newaxis, :,:]
		self.nC = len(contours)

	def _remove(self, toremove, idx, top):
		#If I have children, remove these first
		if self.hierarchy[0,idx,2] != -1:
			toremove.append(self._remove(toremove, self.hierarchy[0,idx,2], 0))
		#If I have siblings and I am a subtree, delete my siblings too
		if self.hierarchy[0,idx,0] != -1 and top == 0:
			toremove.append(self._remove(toremove, self.hierarchy[0,idx,0], 0))
		#Finally remove myself
		if top == 0:
			return idx 
		else:
			toremove.append(idx)
			return toremove

class OpticalFlowLK:
	def __init__(self, cam, lk_params, feature_params):
		self.cam = cam
		self.p0 = None
		self.p1 = None 
		self.use_ransac = True
		self.lk_params = lk_params
		self.feature_params = feature_params

	def checkedTrace(self, img0, img1, p0, back_threshold = 1.0):
		p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
		p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
		d = abs(p0-p0r).reshape(-1, 2).max(-1)
		status = d < back_threshold
		return p1, status
	
	def run(self, frame_gray, prev_gray, vis):
		green = (0, 255, 0)
		red = (0, 0, 255)
		p0 = cv2.goodFeaturesToTrack(prev_gray, **self.feature_params)
		p2, trace_status = self.checkedTrace(prev_gray, frame_gray, p0)

		self.p1 = p2[trace_status].copy()
		self.p0 = p0[trace_status].copy()

		for (x0, y0), (x1, y1), good in zip(self.p0[:,0], self.p1[:,0], trace_status[:]):
			if good:
				cv2.line(vis, (x0, y0), (x1, y1), (0, 128, 0))
			cv2.circle(vis, (x1, y1), 2, (red, green)[good], -1)
		draw_str(vis, (20, 20), 'feature count: %d' % len(self.p1))


def pointPolygonGrid(f, nx, ny):
	grid = np.zeros((nx, ny))
	for x in range(nx):
		for y in range(ny):
			grid[x,y] = f((y,x))
	return grid 

def ppt(ct, p, b):	 
	if type(p) == tuple:
		return -cv2.pointPolygonTest(ct, p, b)
	elif len(np.shape(p)) == 1:
		return -cv2.pointPolygonTest(ct, tuple(p), b)
	else:
		return np.array([-cv2.pointPolygonTest(ct, tuple(pp), b) for pp in np.array(p)])

def ddunion(ctrs, p, b):
	if len(ctrs) == 1:
		return ppt(ctrs[0], p, b)
	else:
		return dm.dunion(ppt(ctrs[0], p, b), ddunion(ctrs[1:], p, b))

def findObject(img):
	"""Find object within image

	Input:
		-img: input image object

	Output: 
		-mask: mask containing object
		-contours: contours outlining mask
		-hierarchy: hierarchy of contours
	"""
	mask = np.zeros(img.shape[:2],np.uint8)	
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	
	#Cut out object (this is quite slow)
	rect = (0,0,1023,833)
	cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)	
	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

	#Instead just set a threshold on intensity

	#Mask out background
	img = img*mask2[:,:,np.newaxis]
	
	im2, contours, hierarchy = cv2.findContours(mask2.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
	#cv2.drawContours(img, contours, -1, (0,255,0), 3)
	#plt.imshow(img),plt.colorbar(),plt.show()
	return (mask2, contours, hierarchy)

def findObjectThreshold(img, threshold = 7):
	"""Find object within image using simple thresholding

	Input:
		-img: input image object
		-threshold: threshold intensity to apply (can be chosen from looking at 
			histograms)

	Output: 
		-mask: mask containing object
		-contours: contours outlining mask
		-hierarchy: hierarchy of contours
	"""
	#Test code:
	#img = cv2.imread('./video/testcontours.jpg')

	#Just try simple thresholding instead: (quick!, seems to work fine)
	frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#Global threshold
	ret1, mask = cv2.threshold(frame_gray, threshold, 255, cv2.THRESH_TOZERO)
	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

	#Find contours of mask
	im2, c, h = cv2.findContours(mask2.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
	ctrs = Contours(c, h)
	
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
	if len(ctrs.contours) > 1:
		fd = lambda p: dm.ddiff(ppt(ctrs.contours[0], p, True), ddunion(ctrs.contours[1:], p, True))
	else:
		fd = lambda p: ppt(ctrs.contours[0], p, True)
	#Two of them
	#fd = lambda p: dm.dunion(ppt(ctrs.contours[2], p, True), ppt(ctrs.contours[1], p, True))
	#One of them
	#fd = lambda p: ppt(ctrs.contours[0], p, True)
	#Diff of two of them
	#fd = lambda p: dm.ddiff(ppt(ctrs.contours[0], p, True), ppt(ctrs.contours[1], p, True))

	#Draw contours
	#cv2.drawContours(grid, ctrs.contours[0:2], -1, (0,255,0), 3)
	#cv2.imshow('image', grid)
	#cv2.waitKey(0)
	return (mask2, ctrs, fd)

def placeInteriorPoints(mask, npoints):
	"""Place a given number of points in interior of an object 

	Input:
		-mask: mask specifying region in which to generate points 
		-npoints: number of points to generate

	Output:
		-points: list of npoints points
	"""
	#Sampe npoints from inside mask
	pts = np.argwhere(mask)
	indices = np.random.choice(len(pts), npoints)
	intpts = pts[indices]
	return intpts[:,[1, 0]]


def placeBorderPoints(contours, hierarchy, npoints):
	"""Place a given number of points on border of object

	Input:
		-mask: mask specifying region in which to generate points 
		-npoints: number of points to generate

	Output:
		-points: list of npoints points
	"""
	largestcontour = -1
	maxarea = 0
	#Find the outer contour with largest area
	for i in range(np.shape(hierarchy)[1]):
		if hierarchy[0,i,3] == -1:
			area = cv2.contourArea(contours[i])
			if area > maxarea:
				maxarea = area
				largestcontour = i
	#Take from these points regularly
	contourlength = np.shape(contours[largestcontour])[0]
	indices = np.linspace(0,contourlength-1, npoints).astype(int)
	return np.squeeze(contours[largestcontour][indices,:,:])

def placeGoodTrackingPoints(img, mask):
	feature_params = dict( maxCorners = 100, qualityLevel = 0.05, minDistance = 7, blockSize = 7)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	#Find contours of mask
	return cv2.goodFeaturesToTrack(gray, mask = mask, **feature_params)

def drawPoints(img, pts, types = None):
	npts = len(pts)
	colors = [[0, 255, 0], [255, 0, 0], [0, 0, 255]]
	if types is None:
		types = [0]*npts 
	for (i,pt) in enumerate(pts):
		cv2.circle(img,tuple(pt.astype(int)),3,colors[types[i]],-1)

def drawGrid(img, pts, bars, L = None, F = None):
	npts = len(bars)
	if L is not None:
		colors = np.zeros((npts, 3))
		forces = F(L)
		maxF = np.max(forces)
		colors[:,1] = 100+155*forces/maxF
	else:
		colors = npml.repmat([0, 255, 0], npts, 1)

	for (bar, color) in zip(bars, colors):
		cv2.line(img, tuple(pts[bar[0]].astype(int)), tuple(pts[bar[1]].astype(int)), color)

def interpolateSparseOpticFlow():
	return 