#!/usr/bin/env python
import sys, argparse 
from kalman import KalmanFilter, IteratedMSKalmanFilter
from renderer import VideoStream, FlowStream
from distmesh_dyn import DistMesh

import pdb 

class Arguments:
	pass

args = Arguments()

args.fn_in='./video/johntest_brightcontrast_short.avi'
args.flow_in='./video/johntest_brightcontrast_short/flow'
args.fn_out='./video/johntest_brightcontrast_short_output.avi'
args.name='johntest_brightcontrast_short'
args.threshold=9
args.gridsize=22
args.cuda=True

dm_out = './testrun_kalmanfilter.pkl'

capture = VideoStream(args.fn_in, args.threshold)

frame = capture.current_frame()
mask, ctrs, fd = capture.backsub()

#distmesh = DistMesh(frame, h0 = args.gridsize)
#distmesh.createMesh(ctrs, fd, frame, plot = True)
#Save this distmesh and reload it for quicker testing
#distmesh.save(dm_out)

#Load...
distmesh = DistMesh(frame, h0 = args.gridsize)
distmesh.load(dm_out)

#Load flow data from directory
flowstream = FlowStream(args.flow_in)
ret_flow, flowframe = flowstream.peek()

if ret_flow:
	kf = IteratedMSKalmanFilter(distmesh, frame, flowframe, cuda = args.cuda, sparse = True, multi = True)
else:
	print 'Cannot read flow stream'

#kf.compute(capture.gray_frame(), flowframe)
nI = 3
count = 1

print 'Frame %d' % count 
ret, frame, grayframe, mask = capture.read()
ret_flow, flowframe = flowstream.read()

kf.predict()
kf.projectmask(mask)
kf.state.refresh()

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
	kf.compute(grayframe, flowframe, mask, imageoutput = 'screenshots/' + args.name + '_frame_' + str(count))

capture.release()
output.release()
cv2.destroyAllWindows()
raw_input("Finished. Press ENTER to exit")
