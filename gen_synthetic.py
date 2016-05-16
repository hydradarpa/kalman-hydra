from synthetic import TestMesh
import os.path
import os
import cv2

flowfields = {}

#Velocity fields:
flowfields['translate_leftup'] = lambda pt,t: (-3, -3)
flowfields['translate_leftup_stretch'] = lambda pt,t: (-3+pt[0]/200., -3+pt[1]/200.)

#Codecs
codecs = ['h264', 'libx264', 'huffyuv']

#Select geometry, flow field, grid size
name = 'square2_gradient'
ff = 'translate_leftup'
gridsize = 50

fn_in = './synthetictests/' + name + '.png'
v_out = './synthetictests/' + name + '/' + ff + '/' + ff 
m_out = './synthetictests/' + name + '/' + ff + '_mesh.txt'

if os.path.isfile(fn_in):
	if not os.path.isdir('./synthetictests/' + name):
		os.makedirs('./synthetictests/' + name)
	if not os.path.isdir('./synthetictests/' + name + '/' + ff):
		os.makedirs('./synthetictests/' + name + '/' + ff)
	#Load image as grayscale
	img = cv2.imread(fn_in,0)
	tm = TestMesh(img, flowfields[ff], gridsize = gridsize)
	tm.run(v_out, m_out)
	#Save frames as video, compute optic flow 
	os.system("avconv -i %s_frame_%%03d.png -c:v huffyuv %s.avi" % (v_out, v_out))
	os.system("./bin/optical_flow_ext %s.avi %s_flow" % (v_out, v_out))
else:
	print "Cannot find", fn_in, ", exiting."

