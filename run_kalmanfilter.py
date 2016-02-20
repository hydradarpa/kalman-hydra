#!/usr/bin/env python
import sys, argparse 
from kalman import KalmanFilter
from renderer import VideoStream
from distmesh_dyn import DistMesh

import pdb 

def main():
	usage = """run_kalmanfilter.py [input_avi_file] [output_avi_file] [threshold]

HydraGL. State space model using an extended Kalman filter to track Hydra in video

Dependencies:
-Vispy*
-Numpy
-PyCuda**
-DistMesh 
-HDF
-OpenCV2

Notes:
*  Uses OpenGL rendering. If using remotely, you'll need to set up a VirtualGL server
** Requires a CUDA compatible graphics card

Ben Lansdell
02/16/2016
"""

	parser = argparse.ArgumentParser()
	parser.add_argument('fn_in', default='./video/GCaMP_local_prop.avi', 
		help='avi input video file', nargs = '?')
	parser.add_argument('fn_out', default='./video/GCaMP_local_prop_kalman.avi', 
		help='avi output video file', nargs='?')
	parser.add_argument('-t', '--threshold', default=9,
		help='Threshold intensity below which is background', type = int)
	parser.add_argument('-s', '--gridsize', default=15,
		help='Edge length for mesh (smaller is finer)', type = int)
	parser.add_argument('-c', '--cuda', default=False,
		help='Whether or not to do analysis on CUDA', type = bool)
	args = parser.parse_args()

	if len(sys.argv) == 1:
		print("No command line arguments provided, using defaults")
	
	capture = VideoStream(args.fn_in, args.threshold)
	frame = capture.current_frame()
	mask, ctrs, fd = capture.backsub()
	distmesh = DistMesh(frame, h0 = args.gridsize)
	distmesh.createMesh(ctrs, fd, frame)
	
	#Load flow data from .hdf file
	#Not implemented
	flowframe = None #capture.backsub(hdf.read())

	kf = KalmanFilter(distmesh, frame, args.cuda)
	kf.compute(capture.gray_frame(), flowframe)

	nI = 3
	count = 0
	while(capture.isOpened()):
		count += 1
		print 'Frame %d' % count 
		ret, frame, grayframe = capture.read()
	#	flowframe = capture.backsub(hdf.read())
		if ret == False:
			break 
		for i in range(nI):
			print 'Iteration %d' % i 
			raw_input("Finished. Press Enter to continue")
			kf.compute(grayframe, flowframe)
		#kf.compute(grayframe, flowframe)
	
	#capture.release()
	#output.release()
	#cv2.destroyAllWindows()

if __name__ == "__main__":
	sys.exit(main())