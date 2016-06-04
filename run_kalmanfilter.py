#!/usr/bin/env python
import sys, argparse 
from kalman import KalmanFilter
from renderer import VideoStream, FlowStream
from distmesh_dyn import DistMesh

import pdb 

def main():
	usage = """run_kalmanfilter.py [input_avi_file] [optic_flow_path] [output_avi_file] [threshold]

HydraGL. State space model using an extended Kalman filter to track Hydra in video

Dependencies:
-Vispy*
-Numpy
-PyCuda**
-DistMesh 
-HDF
-OpenCV2
-matplotlib

Notes:
*  Uses OpenGL rendering. If using remotely, you'll need to set up a VirtualGL server
** If have a CUDA compatible graphics card

Example: 
./run_kalmanfilter.py ./video/johntest_brightcontrast_short.avi ... 
 ./video/johntest_brightcontrast_short/flow ./video/output.avi -s 15

For help:
./run_kalmanfilter.py -h 

Ben Lansdell
02/16/2016
"""

	parser = argparse.ArgumentParser()
	parser.add_argument('fn_in', default='./video/johntest_brightcontrast_short.avi', 
		help='input video file, any format readable by OpenCV', nargs = '?')
	parser.add_argument('flow_in', default='./video/johntest_brightcontrast_short/', 
		help='input optic flow path', nargs = '?')
	parser.add_argument('fn_out', default='./video/johntest_brightcontrast_short_output.avi', 
		help='avi output video file', nargs='?')
	parser.add_argument('-n', '--name', default='johntest_brightcontrast_short', 
		help='name for saving run images', nargs='?')
	parser.add_argument('-t', '--threshold', default=9,
		help='threshold intensity below which is background', type = int)
	parser.add_argument('-s', '--gridsize', default=18,
		help='edge length for mesh (smaller is finer)', type = int)
	parser.add_argument('-c', '--cuda', default=False,
		help='whether or not to do analysis on CUDA', type = bool)
	args = parser.parse_args()

	if len(sys.argv) == 1:
		print("No command line arguments provided, using defaults")
	
	capture = VideoStream(args.fn_in, args.threshold)
	frame = capture.current_frame()
	mask, ctrs, fd = capture.backsub()
	distmesh = DistMesh(frame, h0 = args.gridsize)
	distmesh.createMesh(ctrs, fd, frame, plot = True)
	
	#Load flow data from directory
	flowstream = FlowStream(args.flow_in)

	kf = KalmanFilter(distmesh, frame, args.cuda)
	kf.compute(capture.gray_frame(), flowframe)
	nI = 3
	count = 0
	while(capture.isOpened()):
		count += 1
		print 'Frame %d' % count 
		ret, frame, grayframe, mask = capture.read()
		ret_flow, flowframe = flowstream.read()
		if ret is False or ret_flow is False:
			break
		#for i in range(nI):
		#	print 'Iteration %d' % i 
		#	raw_input("Finished. Press Enter to continue")
		#	kf.compute(grayframe, flowframe)
		kf.compute(grayframe, flowframe, mask, imageoutput = 'screenshots/' + name + '_frame_' + str(i))
	capture.release()
	output.release()
	cv2.destroyAllWindows()
	raw_input("Finished. Press enter to exit")

if __name__ == "__main__":
	sys.exit(main())