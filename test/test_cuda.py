from cuda import * 
from renderer import Renderer
from kalman2 import KalmanFilter 
import dill 

#Test code to test functionality of CUDA related code
def multiply(a, b):
	return a*b

#class TestCUDA_zeros:
#	def setup(self):
#		#print ("TestCUDA:setup() before each test method")
#		#Load saved distmesh, vel, flow, nx, im1 data
#		with open('test/testdata_zeros.pkl', 'rb') as f:
#			objs = dill.load(f)
#		(distmesh, vel, flow, nx, im1) = objs
#		#Create renderer object 
#		self.rend = Renderer(distmesh, vel, flow, nx, im1, cuda=False)
#		self.cuda = CUDAGL(self.rend._rendertex1, self.rend._fbo1, self.rend._fbo2, self.rend._fbo3, cuda=False)
#		kf = KalmanFilter(distmesh, im1, flow, cuda=False)
#
#		self.state = kf.state 
#		self.y_im = im1 
#		self.y_flow = flow 
#		self.nx = nx 
#		self.deltaX = 3
#
#	def test_initjacobian_CPU_zeros(self):
#		print 'test_initjacobian_CPU_zeros'
#		self.cuda.initjacobian_CPU(self.y_im, self.y_flow)
#		assert multiply(3,4) == 12
#
#	def test_jz_CPU_zeros(self):
#		print 'test_jz_CPU_zeros'
#		self.cuda.initjacobian_CPU(self.y_im, self.y_flow)
#		#Perturb positions then compute jz 
#		#If everything is perturbed by one row, 
#		a = self.cuda.jz_CPU()
#		assert a == 0
#
#	def test_j_CPU_zeros(self):
#		print 'test_j_CPU_zeros'
#		i = 1
#		j = 1
#		self.cuda.initjacobian_CPU(self.y_im, self.y_flow)
#		#Perturb positions then compute jz 
#		a = self.cuda.j_CPU(self.state, self.deltaX, i, j)
#		print a 
#		assert multiply(3,4) == 12
#
#	def test_get_pixel_data(self):
#		print 'test_get_pixel_data'
#		(a,b,c) = self.cuda.get_pixel_data()
#		assert np.sum(a) == 0
#		assert np.sum(b) == 0
#		assert np.sum(c) == 0

#class TestCUDA_ones_nocuda:
#	def setup(self):
#		#print ("TestCUDA:setup() before each test method")
#		#Load saved distmesh, vel, flow, nx, im1 data
#		with open('test/testdata_ones.pkl', 'rb') as f:
#			objs = dill.load(f)
#		(distmesh, vel, flow, nx, im1) = objs
#		kf = KalmanFilter(distmesh, im1, flow, cuda=False, vel = vel)
#		self.rend = kf.state.renderer
#		self.cuda = kf.state.renderer.cudagl 
#		kf.state.render()
#
#		self.state = kf.state 
#		self.y_im = im1 
#		self.y_flow = flow 
#		self.nx = nx 
#		self.deltaX = 3
#
#	def test_initjacobian_CPU_ones(self):
#		eps = 1e-6
#		print 'test_initjacobian_CPU_ones'
#		self.cuda.initjacobian_CPU(self.y_im, self.y_flow)
#		start = self.nx//3
#		end = 2*self.nx//3
#		#print start, end 
#		#print 'z_fx:', np.sum(self.cuda.zfx)
#		#print 1.534*(self.nx*self.nx-(end-start)*(end-start))
#		#print 'z_fy:', np.sum(self.cuda.zfy)
#		#print 1.534*(self.nx*self.nx-(end-start)*(end-start))
#		#print np.sum(np.where(abs(self.cuda.zfx) < eps))
#		assert np.sum(self.cuda.z) - 128*(self.nx*self.nx-(end-start)*(end-start))/255. < eps
#		#assert np.sum(self.cuda.zfx) - 1.534*(self.nx*self.nx-(end-start)*(end-start)) < eps
#		#assert np.sum(self.cuda.zfy) - 1.534*(self.nx*self.nx-(end-start)*(end-start)) < eps
#		assert abs(self.cuda.zfx[start+1, start+1]) < eps
#		assert abs(self.cuda.zfy[start+1, start+1]) < eps
#		assert abs(self.cuda.zfx[start-1, start-1]-1.534) < eps
#		assert abs(self.cuda.zfy[start-1, start-1]-1.534) < eps
#		assert np.sum(abs(self.cuda.zfx) < eps) == (end-start)*(end-start)
#
#	def test_jz_CPU_ones(self):
#		eps = 1e-5
#		self.cuda.initjacobian_CPU(self.y_im, self.y_flow)
#		print 'test_jz_CPU_ones'
#		N = self.state.N
#		for i in range(0, 2*N):
#			self.state.X[i] += 1
#		self.state.refresh()
#		self.state.render()
#		a = self.cuda.jz_CPU()
#		#print a
#		#Contribution from image
#		#print (226+227)*(128./255)*(128./255)
#		#Contributions from flow in x and y
#		#print 2*(226+227)*(1.534*1.534)
#		#Should sum to a:
#		assert abs(a - (226+227)*(128./255)*(128./255) - 2*(226+227)*(1.534*1.534))/a < eps
#
#	def test_jz_CPU_ones_flow(self):
#		eps = 1e-7
#		self.cuda.initjacobian_CPU(self.y_im, self.y_flow)
#		print 'test_jz_CPU_ones_flow'
#		N = self.state.N
#		for i in range(2*N, 4*N):
#			self.state.X[i] += 1
#		self.state.refresh()
#		self.state.render()
#		a = self.cuda.jz_CPU()
#		assert abs(a) < eps
#
##	def test_j_CPU_ones(self):
##		#Hard to test even in very simple examples...
##		print 'test_j_CPU_ones'
##		i = 1
##		j = 1
##		self.cuda.initjacobian_CPU(self.y_im, self.y_flow)
##		#Perturb positions then compute jz 
##		a = self.cuda.j_CPU(self.state, self.deltaX, i, j)
##		print a
##		assert a == 0
#
#	def test_get_pixel_data(self):
#		print 'test_get_pixel_data'
#		(a,b,c) = self.cuda.get_pixel_data()
#		start = self.nx//3
#		end = 2*self.nx//3
#		#Why only 8 digits of precision?
#		#print abs(b[start+1,start+1]-1.534)
#		#print np.sum(a)
#		assert abs(b[start+1, start+1] - 1.534) < 1e-7
#		#Why negative?? Because I flipped it...
#		assert abs(-c[start+1, start+1] - 1.534) < 1e-7
#		assert np.sum(a) == (end-start)*(end-start)*128
#		#assert np.sum(c) - (end-start)*(end-start)*1.534 < (end-start)*(end-start)*1e-7

#class TestCUDA_texture:
#	def setup(self):
#		#Load saved distmesh, vel, flow, nx, im1 data
#		with open('test/testdata_texture.pkl', 'rb') as f:
#			objs = dill.load(f)
#		(distmesh, vel, flow, nx, im1) = objs
#		#Create renderer object 
#		self.rend = Renderer(distmesh, vel, flow, nx, im1, cuda=False)
#		self.cuda = CUDAGL(self.rend._rendertex1, self.rend._fbo1, self.rend._fbo2, self.rend._fbo3, cuda=False)
#		kf = KalmanFilter(distmesh, im1, flow, cuda=False)
#
#		self.state = kf.state 
#		self.y_im = im1 
#		self.nx = nx 
#		self.y_flow = flow 
#		self.deltaX = 3
#
#	def test_initjacobian_CPU_texture(self):
#		print 'test_initjacobian_CPU_texture'
#		self.cuda.initjacobian_CPU(self.y_im, self.y_flow)
#		assert multiply(3,4) == 12
#
#	def test_jz_CPU_texture(self):
#		print 'test_jz_CPU_texture'
#		self.cuda.initjacobian_CPU(self.y_im, self.y_flow)
#		#Perturb positions then compute jz 
#
#		#If everything is perturbed by one row, 
#		a = self.cuda.jz_CPU()
#		assert a == 0
#
#	def test_j_CPU_texture(self):
#		print 'test_j_CPU_texture'
#		i = 1
#		j = 1
#		self.cuda.initjacobian_CPU(self.y_im, self.y_flow)
#		#Perturb positions then compute jz 
#		a = self.cuda.j_CPU(self.state, self.deltaX, i, j)
#		print a 
#		assert multiply(3,4) == 12
#
#	def test_get_pixel_data(self):
#		print 'test_get_pixel_data'
#		(a,b,c) = self.cuda.get_pixel_data()
#		assert multiply(3,4) == 12

class TestCUDA_ones:
	def setup(self):
		#print ("TestCUDA:setup() before each test method")
		#Load saved distmesh, vel, flow, nx, im1 data
		with open('test/testdata_ones.pkl', 'rb') as f:
			objs = dill.load(f)
		(distmesh, vel, flow, nx, im1) = objs
		kf = KalmanFilter(distmesh, im1, flow, cuda=True, vel = vel)
		self.rend = kf.state.renderer
		self.cuda = kf.state.renderer.cudagl 
		kf.state.render()

		self.state = kf.state 
		self.y_im = im1 
		self.y_flow = flow 
		self.nx = nx 
		self.deltaX = 3

	def test_initjacobian_ones(self):
		eps = 1e-6
		print 'test_initjacobian_ones'
		cuda.initjacobian(im1, flow)
		#self.cuda.initjacobian(self.y_im, self.y_flow)
		start = self.nx//3
		end = 2*self.nx//3
		#print start, end 
		#print 'z_fx:', np.sum(self.cuda.zfx)
		#print 1.534*(self.nx*self.nx-(end-start)*(end-start))
		#print 'z_fy:', np.sum(self.cuda.zfy)
		#print 1.534*(self.nx*self.nx-(end-start)*(end-start))
		#print np.sum(np.where(abs(self.cuda.zfx) < eps))
		assert np.sum(self.cuda.z) - 128*(self.nx*self.nx-(end-start)*(end-start))/255. < eps
		#assert np.sum(self.cuda.zfx) - 1.534*(self.nx*self.nx-(end-start)*(end-start)) < eps
		#assert np.sum(self.cuda.zfy) - 1.534*(self.nx*self.nx-(end-start)*(end-start)) < eps
		assert abs(self.cuda.zfx[start+1, start+1]) < eps
		assert abs(self.cuda.zfy[start+1, start+1]) < eps
		assert abs(self.cuda.zfx[start-1, start-1]-1.534) < eps
		assert abs(self.cuda.zfy[start-1, start-1]-1.534) < eps
		assert np.sum(abs(self.cuda.zfx) < eps) == (end-start)*(end-start)

	def test_jz_ones(self):
		eps = 1e-5
		self.cuda.initjacobian(self.y_im, self.y_flow)
		print 'test_jz_ones'
		N = self.state.N
		for i in range(0, 2*N):
			self.state.X[i] += 1
		self.state.refresh()
		self.state.render()
		a = self.cuda.jz()
		#print a
		#Contribution from image
		#print (226+227)*(128./255)*(128./255)
		#Contributions from flow in x and y
		#print 2*(226+227)*(1.534*1.534)
		#Should sum to a:
		assert abs(a - (226+227)*(128./255)*(128./255) - 2*(226+227)*(1.534*1.534))/a < eps

	def test_jz_ones_flow(self):
		eps = 1e-7
		self.cuda.initjacobian(self.y_im, self.y_flow)
		print 'test_jz_ones_flow'
		N = self.state.N
		for i in range(2*N, 4*N):
			self.state.X[i] += 1
		self.state.refresh()
		self.state.render()
		a = self.cuda.jz()
		assert abs(a) < eps