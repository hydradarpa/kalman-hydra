import numpy as np
import numpy.matlib as npml
import cv2

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

def findObjectThreshold(img, threshold = 10):
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
	#Just try simple thresholding instead: (quick!, seems to work fine)
	frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#Global threshold
	#mask = np.where(frame_gray < 10,0,1)
	ret1, mask = cv2.threshold(frame_gray, 7, 255, cv2.THRESH_TOZERO)
	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	#frame = frame*mask2[:,:,np.newaxis]
	#Find contours of mask
	im2, contours, hierarchy = cv2.findContours(mask2.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
	#Remove small contours
	contours2 = []
	for ct in contours:
		area = cv2.contourArea(ct)
		if area > 40:
			contours2.append(ct)
	contours = contours2

	#Draw contours
	cv2.drawContours(img, contours, -1, (0,255,0), 3)
	return (mask2, contours, hierarchy)

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
		types = np.zeros(npts, 1)
	for (i,pt) in enumerate(pts):
		cv2.circle(img,tuple(pt),3,colors[types[i]],-1)

def drawTriangles(img, grid):
	color = [0, 255, 0]
	for pt in pts:
		cv2.circle(img,tuple(pt),3,color,-1)

def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def drawDelaunay(img, subdiv, borderpts, delaunay_color) :
	triangleList = subdiv.getTriangleList();
	size = img.shape
	r = (0, 0, size[1], size[0])
	borderlist = np.ndarray.tolist(borderpts)
	for t in triangleList :         
		pt1 = [t[0], t[1]]
		pt2 = [t[2], t[3]]
		pt3 = [t[4], t[5]]
		#If all three pts are in borderpts, don't draw it
		if pt1 in borderlist and pt2 in borderlist and pt3 in borderlist:
			continue 
		if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :         
			cv2.line(img, tuple(pt1), tuple(pt2), delaunay_color, 1, cv2.LINE_AA, 0)
			cv2.line(img, tuple(pt2), tuple(pt3), delaunay_color, 1, cv2.LINE_AA, 0)
			cv2.line(img, tuple(pt3), tuple(pt1), delaunay_color, 1, cv2.LINE_AA, 0)

def computeDelaunay(pts, img):
	retur