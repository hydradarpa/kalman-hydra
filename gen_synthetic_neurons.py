from synth import TestMeshNeurons
import os.path
import os
import cv2
import sys

from synthetic.flowfields import flowfields 

#Note that from this we can compute the _exact_ optic flow of an object...
#and so can measure the discrepency between the Brox flow and the true flow

def main(argv):
	if len(argv) < 3:
		print 'Not enough arguments'
	ff = argv[1]
	name = argv[2]
	#name = 'hydra_neurons1'
	#ff = 'warp'

	#Codecs
	codecs = ['h264', 'libx264', 'huffyuv']
	
	#Select geometry, flow field, grid size
	gridsize = 30
	
	fn_in = './synthetictests/' + name + '.png'
	n_in = './synthetictests/' + name + '_neurons.csv'
	
	v_out = './synthetictests/' + name + '/' + ff + '/' + ff 
	m_out = './synthetictests/' + name + '/' + ff + '_mesh.txt'
	n_out = './synthetictests/' + name + '/' + ff + '_neurons.txt'
	
	if os.path.isfile(fn_in):
		if not os.path.isdir('./synthetictests/' + name):
			os.makedirs('./synthetictests/' + name)
		if not os.path.isdir('./synthetictests/' + name + '/' + ff):
			os.makedirs('./synthetictests/' + name + '/' + ff)
		#Load image as grayscale
		img = cv2.imread(fn_in,0)
		tm = TestMeshNeurons(img, n_in, flowfields[ff], gridsize = gridsize, plot = True)
		tm.run(v_out, m_out, n_out)
		#Save frames as video, compute optic flow 
		os.system("avconv -i %s_frame_%%03d.png -c:v huffyuv -y %s.avi" % (v_out, v_out))
		os.system("avconv -i %s_neuron_frame_%%03d.png -c:v huffyuv -y %s_neurons.avi" % (v_out, v_out))
		os.system("./bin/optical_flow_ext %s.avi %s_flow" % (v_out, v_out))
	else:
		print "Cannot find %s, exiting."%fn_in	

if __name__ == '__main__':
	main(sys.argv)