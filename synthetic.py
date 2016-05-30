"""Generate synthetic datasets"""
from imgproc import findObjectThreshold
from distmesh_dyn import DistMesh
from kalman2 import KalmanFilter

import numpy as np 
import cv2 

def test_data(nx, ny):
	nframes = 10
	speed = 3
	video = np.zeros((nx, ny, nframes), dtype = np.uint8)
	im = np.zeros((nx, ny))
	#Set up a box in the first frame, with some basic changes in intensity
	start = nx//3
	end = 2*nx//3
	width = end-start
	height = end-start
	flow = np.zeros((nx, ny, 2, nframes), dtype=np.float32)

	for i in range(start,end):
		for j in range(start,end):
			if i > j:
				col = 128
			else:
				col = 255
			im[i,j] = col
	im = np.flipud(im)
	#Translate the box for a few frames
	for i in range(nframes):
		imtrans = im[speed*i:,speed*i:]
		s = start-speed*i
		flow[s:s+width, s:s+height,0,i] = -speed
		flow[s:s+width, s:s+height,1,i] = -speed
		#flow[s:s+width, s:s+height,0,i] = -speed
		#flow[s:s+width, s:s+height,1,i] = -speed
		if i > 0:
			video[:-speed*i,:-speed*i,i] = imtrans 
		else:
			video[:,:,i] = imtrans
	return video, flow

def test_data_up(nx, ny):
	nframes = 30
	speed = 2
	video = np.zeros((nx, ny, nframes), dtype = np.uint8)
	im = np.zeros((nx, ny))
	#Set up a box in the first frame, with some basic changes in intensity
	start = nx//3
	end = 2*nx//3
	for i in range(start,end):
		for j in range(start,end):
			if i > j:
				col = 128
			else:
				col = 255
			im[i,j] = col
	#Translate the box for a few frames
	for i in range(nframes):
		imtrans = im[:,speed*i:]
		if i > 0:
			video[:,:-speed*i,i] = imtrans 
		else:
			video[:,:,i] = imtrans
	return video 

def test_data_texture(nx, ny):
	noise = 0
	nframes = 10
	speed = 3
	video = np.zeros((nx, ny, nframes), dtype = np.uint8)
	im = np.zeros((nx, ny))
	#Set up a box in the first frame, with some basic changes in intensity
	start = nx//3
	end = 2*nx//3
	for i in range(start,end):
		for j in range(start,end):
			if i > j:
				col = 128
			else:
				col = 200
			im[i,j] = col + noise*np.random.normal(size = (1,1))
	#Add noise
	#noise = 
	#Apply Gaussian blur 
	#im = im + noise 
	im = cv2.GaussianBlur(im,(15,15),0)
	#Translate the box for a few frames
	for i in range(nframes):
		imtrans = im[speed*i:,speed*i:]
		if i > 0:
			video[:-speed*i,:-speed*i,i] = imtrans 
		else:
			video[:,:,i] = imtrans
	return video 

def test_data_image(fn = './video/milkyway.jpg'):
	imsrc = cv2.imread(fn,0)
	nx = imsrc.shape[0]
	ny = imsrc.shape[1]
	im = np.zeros((nx, ny), dtype=np.uint8)
	imsrc = cv2.resize(imsrc,None,fx=1./2, fy=1./2, interpolation = cv2.INTER_CUBIC)
	width = imsrc.shape[0]
	height = imsrc.shape[1]
	start = nx//3
	im[start:start+width, start:start+height] = imsrc

	nframes = 10
	speed = 3
	noise = 10
	video = np.zeros((nx, ny, nframes), dtype = np.uint8)
	flow = np.zeros((nx, ny, 2, nframes), dtype=np.float32)

	#Translate the box for a few frames
	for i in range(nframes):
		start = nx//3-i*speed
		imtrans = im[speed*i:,speed*i:]
		if i > 0:
			flow[start:start+width, start:start+height,0,i] = -speed
			flow[start:start+width, start:start+height,1,i] = -speed
			video[:-speed*i,:-speed*i,i] = imtrans 
		else:
			video[:,:,i] = imtrans

	imnoisex = np.zeros((nx, ny))
	imnoisey = np.zeros((nx, ny))
	#Set up a box in the first frame, with some basic changes in intensity
	start = nx//3
	end = nx//3+width
	for i in range(start,end):
		for j in range(start,end):
			imnoisex[i,j] = noise*np.random.normal(size = (1,1))
			imnoisey[i,j] = noise*np.random.normal(size = (1,1))
	#Add noise
	#noise = 
	#Apply Gaussian blur 
	#im = im + noise 
	imnoisex = cv2.GaussianBlur(imnoisex,(15,15),0)
	imnoisey = cv2.GaussianBlur(imnoisey,(15,15),0)

	flow[:,:,0,0] = imnoisex
	flow[:,:,0,0] = imnoisey

	return video, flow 

class TestMesh:
	"""Takes an initial frame (object), creates a mesh and morphs the mesh points
	over time according to a provided flow field. Saves the results and the true
	set of mesh points"""
	def __init__(self, img, flowfield, gridsize = 20, threshold = 8, plot = False):
		self.img = img
		self.nx = img.shape[0]
		self.ny = img.shape[1]
		self.threshold = threshold 
		mask, ctrs, fd = self.backsub()
		self.distmesh = DistMesh(img, h0 = gridsize)
		self.distmesh.createMesh(ctrs, fd, img, plot = plot)
		self.t = 0
		self.flowfield = flowfield
		self.writer = None
		flowzeros = np.zeros((self.nx, self.ny, 2))
		self.kf = KalmanFilter(self.distmesh, img, flowzeros, cuda = False)
		self.N = self.kf.state.N

	def forward(self):
		#Update mesh points according to the velocity flow field
		for i in range(self.N):
			x = self.kf.state.X[2*i]
			y = self.kf.state.X[2*i+1]
			(vx, vy) = self.flowfield([x, y], self.t)
			self.kf.state.X[2*i] += vx
			self.kf.state.X[2*i+1] += vy
			self.kf.state.X[2*self.N + 2*i] = vx
			self.kf.state.X[2*self.N + 2*i+1] = vy

	def render(self):
		self.kf.state.renderer.update_vertex_buffer(self.kf.state.vertices(), self.kf.state.velocities())
		#self.kf.state.renderer.on_draw(None)
		pred_img = self.kf.state.renderer.getpredimg()
		return pred_img

	def _createwriter(self, video_out):
		fourcc = cv2.VideoWriter_fourcc(*'XVID') 
		framesize = (self.nx, self.ny)
		self.writer = cv2.VideoWriter(video_out, fourcc, 2.0, framesize[::-1])

	def _strState(self):
		return "X," + ','.join([str(x[0]) for x in self.kf.state.X]) + '\n'

	def run(self, video_out, mesh_out, steps = 50):
		#Number of time steps
		f_out = open(mesh_out, 'w')
		#Write: mesh size
		f_out.write("size,%d\n"%self.N)
		#self._createwriter(video_out)
		#assert self.writer is not None, "Cannot create VideoWriter object"
		print "Simulating", steps, "steps of mesh warping"
		for i in range(steps):
			self.forward()
			pred_img = self.render()
			#self.writer.write(pred_img)
			#Or just save the images
			fn_out = video_out + "_frame_%03d"%i + ".png"
			cv2.imwrite(fn_out,pred_img)
			f_out.write(self._strState())
		f_out.close()
		#self.writer.release()
		cv2.destroyAllWindows()
		print "Done"

	def backsub(self):
		(mask, ctrs, fd) = findObjectThreshold(self.img, threshold = self.threshold)
		return mask, ctrs, fd


class TestMeshNeurons(TestMesh):
	"""Takes an initial frame (object), creates a mesh and morphs the mesh points
	over time according to a provided flow field. Saves the results and the true
	set of mesh points"""
	def __init__(self, img, n_in, flowfield, gridsize = 20, threshold = 8, plot = False):
		TestMesh.__init__(self, img, flowfield, gridsize, threshold, plot)
		#Read in neurons
		neurons = []
		for line in open(n_in, 'r'):
			coords = [float(x) for x in line.split(',')[1:3]]
			neurons.append([coords[0], coords[1]])
		self.neurons = np.array(neurons)
		self.nn = len(self.neurons)

	def forward(self):
		TestMesh.forward(self)
		#Update neuron positions
		for i in range(self.nn):
			x = self.neurons[i,0]
			y = self.neurons[i,1]
			(vx, vy) = self.flowfield([x, y], self.t)
			self.neurons[i,0] += vx
			self.neurons[i,1] += vy 

	def _strNeurons(self):
		return "neurons," + ','.join([str(x[0]) for x in self.neurons.reshape(1,-1)]) + '\n'

	def run(self, video_out, mesh_out, n_out, steps = 50):
		#Number of time steps
		f_out = open(mesh_out, 'w')
		#Write: mesh size
		f_out.write("size,%d\n"%self.N)
		fneuron_out = open(n_out, 'w')
		#self._createwriter(video_out)
		#assert self.writer is not None, "Cannot create VideoWriter object"
		print "Simulating", steps, "steps of mesh warping"
		for i in range(steps):
			self.forward()
			pred_img = self.render()
			col_img = cv2.cvtColor(pred_img, cv2.COLOR_RGBA2RGB)
			#self.writer.write(pred_img)
			#Or just save the images
			fn_out = video_out + "_frame_%03d"%i + ".png"
			cv2.imwrite(fn_out,pred_img)
			f_out.write(self._strState())
			fneuron_out.write(self._strNeurons())

			#Draw neurons on top of frame
			fn_neuron_out = video_out + "_neuron_frame_%03d"%i + ".png"
			blank = np.zeros(col_img.shape)
			for j in range(self.nn):
				#print j
				x = self.neurons[j,0]
				y = self.neurons[j,1]
				center = (int(x),int(y))
				#ptColor = cv2.cv.CV_RGB(0, 255, 0)
				ptColor = (255,0,0)
				cv2.circle(col_img, center, 2, ptColor, -1)
			cv2.imwrite(fn_neuron_out,col_img)

		f_out.close()
		fneuron_out.close()
		#self.writer.release()
		cv2.destroyAllWindows()
		print "Done"

	def drawfirstframe(self, fn_out):
		return 0