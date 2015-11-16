import numpy as np
import cv2
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("./video/local_prop_cb_with_bud.avi")
#cap = cv2.VideoCapture("./video/20150306_GCaMP_Chl_EC_local_prop.avi")

ret, frame = cap.read()

filename = './video/testgrabcut.avi'
fps = 20
framesize = np.shape(frame)[0:2]
fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
#output = cv2.VideoWriter(filename,fourcc, fps, framesize)


mask = np.zeros(frame.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
cv2.grabCut(frame,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

rect = (0,0,1023,833)

n=0 
while(1):
	ret,frame = cap.read()
	print n 
	n = n + 1
	if not ret: 
		break 
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#Grabcut: probably best but very slow, even when we used previous info
	#and only update one iteration at a time...
	cv2.grabCut(frame,mask,rect,bgdModel,fgdModel,1,cv2.GC_EVAL)

	#Just try simple thresholding instead: (quick!)
	#mask = np.where(frame_gray < 10,0,1)

	#Try simple edge detection instead (doesn't work that great):
	#edges = cv2.Canny(frame_gray, 20, 30)
	#dst = cv2.addWeighted(edges, .5, frame_gray,.5, 0)

	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	frame = frame*mask2[:,:,np.newaxis]

	#Find contours of mask
	im2, contours, hierarchy = cv2.findContours(mask2.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)	
	#Draw contours
	cv2.drawContours(frame, contours, -1, (0,255,0), 3)

	cv2.imshow('frame',frame)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
#output.release()
cv2.destroyAllWindows()