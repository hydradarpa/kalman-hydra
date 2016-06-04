from kalman import KalmanFilter, IteratedKalmanFilter
from renderer import VideoStream, FlowStream
from distmesh_dyn import DistMesh
from flowfields import flowfields 

import os.path 
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

import seaborn as sns
import pandas as pd 

#Here we load a computed optic flow from the Brox GPU method
#and compare it to the true flow... and plot the magnitude of the discrepency 
#for each frame.................

#name = 'square5_gradient_texture_rot2'
#name = 'square4_gradient_texture_rot1'
#name = 'square3_gradient_texture'
#name = 'square2_gradient'
#name = 'square1'
name = 'hydra_neurons1'
ff = 'translate_leftup_stretch'
#ff = 'translate_leftup'
cuda = True
sparse = True

#Brox flow:
flownotes = ''
#flownotes = '_solver_it_20' 
#flownotes = '_solver_it_5'
#flownotes = '_inner_it_20'
#flownotes = '_inner_it_5'
#flownotes = '_alpha_0.4'
#flownotes = '_alpha_0.1'
#flownotes = '_gamma_100'
#flownotes = '_gamma_25'

#DeepFlow
#flownotes = '_deep'

#SimpleFlow
#flownotes = '_simple'

m_in = './synthetictests/' + name + '/' + ff + '_mesh.txt'
#Video and Brox flow in
v_in = './synthetictests/' + name + '/' + ff + '/' + ff + '.avi'
flow_in = './synthetictests/' + name + '/' + ff + '/' + ff + '_flow' + flownotes

#Neuron tracking data in
true_neurons = './synthetictests/' + name + '/' + ff + '_neurons.txt'

#Output
img_out = './synthetictests/' + name + '/' + ff + '_testflow' + flownotes + '/'
if not os.path.isdir(img_out):
	os.makedirs(img_out)

threshold = 8

print 'Loading synthetic data streams'
capture = VideoStream(v_in, threshold)
frame = capture.current_frame()
mask, ctrs, fd = capture.backsub()

f_mesh = open(m_in, 'r')
lines = f_mesh.readlines()
nF = len(lines)-1
nX = frame.shape[0]

flowstream = FlowStream(flow_in)
ret_flow, flowframe = flowstream.read()

rms_flow = np.zeros(nF)
true_flow = np.zeros((nX, nX, 2))
rel_abs_res = np.zeros((nF, nX, nX))
abs_res = np.zeros((nF, nX, nX))
video = np.zeros((nF, nX, nX))

samp_frac = 0.1

count = 0
print 'Comparing true flow to computed flow'

abs_flow_sample = []
abs_res_sample = []
video_sample = []

while(capture.isOpened()):
	count += 1
	ret, frame, grayframe, mask = capture.read()
	ret_flow, flowframe = flowstream.read()
	if ret is False or ret_flow is False:
		break
	print 'Frame %d'%count

	video[count,:,:] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#Compute true flow based on all points in object undergoing flow field
	#Find all pts in mask
	#.... HORRIBLY slow
	for i in range(nX):
		for j in range(nX):
			m = mask[i,j]
			(fx, fy) = flowfields[ff]((i,j),0)
			true_flow[i,j,:] = [fx*m, fy*m]
			res = m*(true_flow[i,j,:] - flowframe[i,j,:])
			abs_res[count,i,j] = np.sqrt(np.sum(res*res))
			if m > 0:
				rel_abs_res[count,i,j] = np.sqrt(np.sum(res*res))/np.sqrt(fx*fx+fy*fy)
				u = np.random.uniform()
				if u < samp_frac:
					abs_flow_sample += [np.sqrt(fx*fx+fy*fy)]
					abs_res_sample += [abs_res[count,i,j]]
					video_sample += [video[count,i,j]]

	#Load computed flow and compare
	r = true_flow - flowframe
	s = np.sum(mask)
	rms_flow[count] = np.sqrt(np.sum(np.multiply(r, r))/s)
	plt.clf()
	plt.imshow(abs_res[count,:,:])
	plt.colorbar()
	plt.savefig(img_out + 'flow_abs_res_%03d.png'%count)
	plt.clf()
	plt.imshow(rel_abs_res[count,:,:])
	plt.colorbar()
	plt.savefig(img_out + 'flow_rel_abs_res_%03d.png'%count)
	#cv2.imwrite(img_out + 'flow_abs_res_%03d.png'%count, abs_res[count,:,:])
	print 'RMS_flow:', rms_flow[count]

print 'Done... how\'d we do?'

df = pd.DataFrame({'abs_flow_sample':pd.Series(abs_flow_sample),\
	'abs_res_sample':pd.Series(abs_res_sample), \
	'video_sample':pd.Series(video_sample)})

g1 = sns.jointplot("video_sample", "abs_res_sample", data = df, kind="kde", color="b")
g1.savefig(img_out + 'intensity_abs_res.eps')
g2 = sns.jointplot("abs_flow_sample", "abs_res_sample", data = df, kind="kde", color="b")
g2.savefig(img_out + 'flow_abs_res.eps')

#Plot abs error in flow vs abs magnitude of flow
plt.clf()
plt.plot(range(nF), rms_flow, label='RMS flow')
plt.legend(loc='upper left')
plt.ylabel('RMS')
plt.xlabel('Frame')
plt.savefig(img_out + 'flow_rms.eps')