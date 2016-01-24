import cv2
import numpy as np
#cap = cv2.VideoCapture("./video/local_prop_cb_with_bud.avi")
cap = cv2.VideoCapture("./video/20150306_GCaMP_Chl_EC_local_prop.avi")

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

#Save the output as an avi file
filename = "./flows/20150306_GCaMP_Chl_EC_local_prop_.mp4"
fps = 20.0
framesize = np.shape(frame1)[0:2]
framesize = framesize[::-1]

fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
output = cv2.VideoWriter(filename,fourcc, fps, framesize)
mmag = 20;

deepflow = cv2.optflow.createOptFlow_SimpleFlow()
f = np.zeros(np.shape(next))

while(cap.isOpened()):
      ret, frame2 = cap.read()
      if ret == False:
        break 
      next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
      #calcOpticalFlowFarneback(prev, next, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags[, flow])  flow
      #Parameters: 
      #prev  first 8-bit single-channel input image.
      #next  second input image of the same size and the same type as prev.
      #pyr_scale  parameter, specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
      #levels  number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
      #winsize  averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
      #iterations  number of iterations the algorithm does at each pyramid level.
      #poly_n  size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
      #poly_sigma  standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
      #flags 

      flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

      #Deep flow (seems very slow)
      #flow = deepflow.calc(prvs, next, f)

      #Simple flow (also seems very slow)
      #flow = cv2.optflow.calcOpticalFlowSF(prvs, next, f, 3, 2, 4, 4.1, 25, 18, 55, 25, 0, 18, 55, 25, 10)

      mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
      #mag[mag<.5] = 0
      hsv[...,0] = ang*180/np.pi/2
      hsv[...,2] = mag/mmag*255; #cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
      #hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
      bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
      dst = cv2.addWeighted(frame2,0.7,bgr,0.3,0)
      output.write(dst)

      cv2.imshow('frame2',dst)
      k = cv2.waitKey(5) & 0xff
      if k == 27:
          break
      elif k == ord('s'):
          cv2.imwrite('opticalfb.png',frame2)
          cv2.imwrite('opticalhsv.png',bgr)
      prvs = next

cap.release()
output.release()
cv2.destroyAllWindows()