import numpy as np
import cv2

cap = cv2.VideoCapture("/home/lansdell/projects/hydra/video/local_prop_cb_with_bud.avi")

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

# Define the codec and create VideoWriter object
#fourcc = cv2.cv.CV_FOURCC(*'MP4V')
#fourcc = cv2.cv.CV_FOURCC(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
framesize = np.shape(frame1)[0:2]
out = cv2.VideoWriter('output.avi',fourcc, 20.0, framesize[::-1])

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('output.avi',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
print 'Done'
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()