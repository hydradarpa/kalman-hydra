import numpy as np
import cv2
from matplotlib import pyplot as plt

import imgproc as ip

fn_in = "./video/local_prop_cb_with_bud.avi"
#fn_in = "./video/20150306_GCaMP_Chl_EC_local_prop.avi"

cap = cv2.VideoCapture(fn_in)

#Read first frame
ret, img = cap.read()

hsv = np.zeros_like(img)
hsv[...,1] = 255

inpoints = 400
borderpoints = 200

(mask, contours, hierarchy) = ip.findObjectThreshold(img)

#Setup Delaunay triangulation
#grid = cv2.Subdiv2D((0,0,1023,833))
#gridcolor = (0, 120, 0)
#for pt in pts:
#	grid.insert(tuple(pt))

#Redistribute interior points given good points and border points an anchors
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

#pts = pts[:,np.newaxis,:].astype('float32')

while(1):
	ret,frame = cap.read()
	if not ret: 
		break 
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# calculate optical flow
	#p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, pts, None, **lk_params)
	flow = cv2.calcOpticalFlowFarneback(old_gray,frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

	#Draw points 
	#ip.drawPoints(frame, np.squeeze(p1).astype(int), types)
	#Draw triangles
	#ip.drawDelaunay(frame, grid, border, gridcolor)

	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	#mag[mag<.5] = 0
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
	dst = cv2.addWeighted(frame,0.7,bgr,0.3,0)
	#output.write(dst)

	cv2.imshow('frame2',dst)
	k = cv2.waitKey(5) & 0xff
	if k == 27:
		break
	elif k == ord('s'):
		cv2.imwrite('opticalfb.png',frame2)
		cv2.imwrite('opticalhsv.png',bgr)
	old_gray = frame_gray

cv2.destroyAllWindows()
cap.release()