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

while(1):
	ret,frame = cap.read()
	print n 
	n = n + 1
	if not ret: 
		break 

	#Just try simple thresholding instead: (quick!, seems to work fine)
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#Global threshold
	#mask = np.where(frame_gray < 10,0,1)
	ret1, mask = cv2.threshold(frame_gray, 7, 255, cv2.THRESH_TOZERO)

	#Adaptive threshold
	#th2 = cv2.adaptiveThreshold(frame_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,15,2)
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
	cv2.drawContours(frame, contours, -1, (0,255,0), 3)

	cv2.imshow('frame',frame)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap.release()
#output.release()
cv2.destroyAllWindows()