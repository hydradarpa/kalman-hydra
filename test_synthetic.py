from kalman import KalmanFilter, IteratedKalmanFilter, IteratedMSKalmanFilter
from renderer import VideoStream, FlowStream
from distmesh_dyn import DistMesh
import os.path 
import cv2
import numpy as np 

from matplotlib import pyplot as plt

#def test_synthetic(name = 'square2_gradient' ,ff = 'translate_leftup' ,notes = \
#	'masked_iekf', cuda = True ,sparse = True):

#name = 'hydra1'
name = 'hydra_neurons1'
#name = 'square3_gradient_texture'
#name = 'square2_gradient'
#name = 'square1'
#ff = 'translate_leftup_stretch'
#ff = 'translate_leftup'
#ff = 'warp'
ff = 'rotate'
notes = 'masked_iekf_multi'
cuda = True
sparse = True
multi = True 

#Input
m_in = './synthetictests/' + name + '/' + ff + '_mesh.txt'
dm_in = './synthetictests/' + name + '/' + ff + '_initmesh.pkl'
v_in = './synthetictests/' + name + '/' + ff + '/' + ff + '.avi'
flow_in = './synthetictests/' + name + '/' + ff + '/' + ff + '_flow'

#Output
img_out = './synthetictests/' + name + '/' + ff + '_' + notes + '_pred/'
if not os.path.isdir(img_out):
	os.makedirs(img_out)

gridsize = 18
threshold = 8

#Create KF
print 'Loading synthetic data streams'
capture = VideoStream(v_in, threshold)
frame = capture.current_frame()
mask, ctrs, fd = capture.backsub()
distmesh = DistMesh(frame, h0 = gridsize)
distmesh.load(dm_in)

#Load true data
f_mesh = open(m_in, 'r')
lines = f_mesh.readlines()
nF = len(lines)-1
nX = int(lines[0].split(',')[1])
truestates = np.zeros((nF, nX*4), dtype = np.float32)
predstates = np.zeros((nF, nX*4), dtype = np.float32)
for i in range(1,nF+1):
	line = lines[i]
	truestates[i-1,:] = [float(x) for x in line.split(',')[1:]]

predstates[0,:] = truestates[0,:]

rms_vel = np.zeros(nF)
rms_pos = np.zeros(nF)

flowstream = FlowStream(flow_in)
ret_flow, flowframe = flowstream.read()
#kf = IteratedMSKalmanFilter(distmesh, frame, flowframe, cuda = cuda, sparse = sparse)
kf = IteratedMSKalmanFilter(distmesh, frame, flowframe, cuda = cuda, sparse = sparse, multi = multi)

count = 0
print 'Tracking with Kalman filter'
while(capture.isOpened()):
#for idx in range(1):
	count += 1
	ret, frame, grayframe, mask = capture.read()
	ret_flow, flowframe = flowstream.read()
	if ret is False or ret_flow is False:
		break

	print 'Frame %d' % count 
	kf.compute(grayframe, flowframe, mask, imageoutput = img_out+'solution_frame_%03d'%count)
	#kf.compute(grayframe, flowframe, mask, imageoutput = './test_multipert_validation_%03d'%count)
	#kf.compute(grayframe, flowframe, mask)

	predstates[count,:] = np.squeeze(kf.state.X)
	r_pos = truestates[count,0:(2*nX)]-predstates[count,0:(2*nX)]
	r_vel = truestates[count,(2*nX):]-predstates[count,(2*nX):]
	rms_pos[count] = np.sqrt(np.mean(np.multiply(r_pos, r_pos)))
	rms_vel[count] = np.sqrt(np.mean(np.multiply(r_vel, r_vel)))
	print 'RMS_pos:', rms_pos[count], 'RMS_vel:', rms_vel[count]

print 'Saving'
np.savez('./synthetictests/' + name + '/' + ff + '_' + notes + '_pred.npz', predstates, truestates, rms_pos, rms_vel)

print 'Done... how\'d we do?'

#Make plot of tracking error
plt.plot(range(nF), rms_pos, label='RMS position')
plt.plot(range(nF), rms_vel, label='RMS velocity')
plt.legend(loc='upper left')
plt.ylabel('RMS')
plt.xlabel('Frame')
plt.savefig('./synthetictests/' + name + '/' + ff + '_' + notes + '_pred_rms.eps')