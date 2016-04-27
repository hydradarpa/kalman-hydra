#################################
#measure image similarity on GPU#
#################################

#lansdell. Feb 15th 2016

import numpy as np, Image
import sys, time, os

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.vertex_buffer_object import *
from OpenGL.GL.ARB.pixel_buffer_object import *

try:
	print 'Importing pycuda'
	import pycuda.driver as cuda_driver
	import pycuda.gl as cuda_gl
	import pycuda
	from pycuda.compiler import SourceModule
	print '...success'
except:
	print "...pycuda not installed"

from vispy import gloo 

import pdb 
import cv2
from matplotlib import pyplot as plt

class CUDAGL:
	def __init__(self, texture, texture_fx, texture_fy, fbo, fbo_fx, fbo_fy, cuda):
		self.cuda = cuda 
		self._fbo1 = fbo 
		self._fbo2 = fbo_fx 
		self._fbo3 = fbo_fy 
		self.texture = texture
		self.texture_fx = texture_fx
		self.texture_fy = texture_fy
		self.width = texture.shape[0]
		self.height = texture.shape[1]
		self.size = texture.shape
		print 'cuda:', cuda 
		if self.cuda:
			import pycuda.gl.autoinit
			import pycuda.gl
			cuda_gl = pycuda.gl
			cuda_driver = pycuda.driver
		
			cuda_module = SourceModule("""
			extern "C" {
			__global__ void jz(unsigned char *y_tilde, unsigned char *yp_tilde, unsigned char *z)
			{
			  int block_num        = blockIdx.x + blockIdx.y * gridDim.x;
			  int thread_num       = threadIdx.y * blockDim.x + threadIdx.x;
			  int threads_in_block = blockDim.x * blockDim.y;
			  //Since the image is RGBA we multiply the index 4.
			  //We'll only use the first 3 (RGB) channels though
			  int idx              = 4 * (threads_in_block * block_num + thread_num);
			  //y_tilde[idx  ] = 255 - y_im[idx  ];
			  //y_tilde[idx+1] = 255 - y_im[idx+1];
			  //y_tilde[idx+2] = 255 - y_im[idx+2];
			}

			__global__ void j(unsigned char *y_tilde, unsigned char *yp_tilde, unsigned char *z)
			{
			  int block_num        = blockIdx.x + blockIdx.y * gridDim.x;
			  int thread_num       = threadIdx.y * blockDim.x + threadIdx.x;
			  int threads_in_block = blockDim.x * blockDim.y;
			  //Since the image is RGBA we multiply the index 4.
			  //We'll only use the first 3 (RGB) channels though
			  int idx              = 4 * (threads_in_block * block_num + thread_num);
			  //y_tilde[idx  ] = 255 - y_im[idx  ];
			  //y_tilde[idx+1] = 255 - y_im[idx+1];
			  //y_tilde[idx+2] = 255 - y_im[idx+2];
			}

			__global__ void initjac(unsigned char *y_tilde, unsigned char *y_im)
			{
			  int block_num        = blockIdx.x + blockIdx.y * gridDim.x;
			  int thread_num       = threadIdx.y * blockDim.x + threadIdx.x;
			  int threads_in_block = blockDim.x * blockDim.y;
			  //Since the image is RGBA we multiply the index 4.
			  //We'll only use the first 3 (RGB) channels though
			  int idx              = 4 * (threads_in_block * block_num + thread_num);
			  //y_tilde[idx  ] = 255 - y_im[idx  ];
			  //y_tilde[idx+1] = 255 - y_im[idx+1];
			  //y_tilde[idx+2] = 255 - y_im[idx+2];
			}

			//Sum all elements of an input array of floats...
			__global__ void total(float *input, float *output, int len) 
			{
				const int BLOCK_SIZE = 1024;
			    // Load a segment of the input vector into shared memory
			    __shared__ float partialSum[2*BLOCK_SIZE];
			    int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
			    unsigned int t = threadIdx.x;
			    unsigned int start = 2*blockIdx.x*blockDim.x;
			
			    if ((start + t) < len)
			    {
			        partialSum[t] = input[start + t];      
			    }
			    else
			    {       
			        partialSum[t] = 0.0;
			    }
			    if ((start + blockDim.x + t) < len)
			    {   
			        partialSum[blockDim.x + t] = input[start + blockDim.x + t];
			    }
			    else
			    {
			        partialSum[blockDim.x + t] = 0.0;
			    }
			
			    // Traverse reduction tree
			    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
			    {
			      __syncthreads();
			        if (t < stride)
			            partialSum[t] += partialSum[t + stride];
			    }
			    __syncthreads();
			
			    // Write the computed sum of the block to the output vector at correct index
			    if (t == 0 && (globalThreadId*2) < len)
			    {
			        output[blockIdx.x] = partialSum[t];
			    }
			}
			}
			""", no_extern_c=1)
			# The argument "PPP" indicates that the zscore function will take three PBOs as arguments
			self.cuda_jz = cuda_module.get_function("jz")
			self.cuda_jz.prepare("PPP")
			self.cuda_j = cuda_module.get_function("j")
			self.cuda_j.prepare("PPP")
			self.cuda_initjac = cuda_module.get_function("initjac")
			self.cuda_initjac.prepare("PP")
			self.cuda_total = cuda_module.get_function("total")
			#self.cuda_total.prepare("PPP")

			# create y_tilde and y_im pixel buffer objects for processing
			self._createPBOs()

	def _createPBOs(self):
		global pycuda_y_tilde_pbo, y_tilde_pbo,\
		 pycuda_y_fx_tilde_pbo, y_fx_tilde_pbo,\
		 pycuda_y_fy_tilde_pbo, y_fy_tilde_pbo,\
		 pycuda_yp_tilde_pbo, yp_tilde_pbo,\
		 pycuda_yp_fx_tilde_pbo, yp_fx_tilde_pbo,\
		 pycuda_yp_fy_tilde_pbo, yp_fy_tilde_pbo,\
		 pycuda_ypp_tilde_pbo, ypp_tilde_pbo,\
		 pycuda_ypp_fx_tilde_pbo, ypp_fx_tilde_pbo,\
		 pycuda_ypp_fy_tilde_pbo, ypp_fy_tilde_pbo,\
		 pycuda_y_im_pbo, y_im_pbo,\
		 pycuda_y_fx_pbo, y_fx_pbo,\
		 pycuda_y_fy_pbo, y_fy_pbo

		num_texels = self.width*self.height
		data = np.zeros((num_texels,4),np.uint8)

		###########
		#y_im data#
		###########
		y_tilde_pbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, y_tilde_pbo)
		glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		pycuda_y_tilde_pbo = cuda_gl.BufferObject(long(y_tilde_pbo))

		yp_tilde_pbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, yp_tilde_pbo)
		glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		pycuda_yp_tilde_pbo = cuda_gl.BufferObject(long(yp_tilde_pbo))

		ypp_tilde_pbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, ypp_tilde_pbo)
		glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		pycuda_ypp_tilde_pbo = cuda_gl.BufferObject(long(ypp_tilde_pbo))

		y_im_pbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, y_im_pbo)
		glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		pycuda_y_im_pbo = cuda_gl.BufferObject(long(y_im_pbo))

		#############
		#Flow x data#
		#############
		data = np.zeros((num_texels,4),np.float32)
		y_fx_tilde_pbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, y_fx_tilde_pbo)
		glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		pycuda_y_fx_tilde_pbo = cuda_gl.BufferObject(long(y_fx_tilde_pbo))

		yp_fx_tilde_pbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, yp_fx_tilde_pbo)
		glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		pycuda_yp_fx_tilde_pbo = cuda_gl.BufferObject(long(yp_fx_tilde_pbo))

		ypp_fx_tilde_pbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, ypp_fx_tilde_pbo)
		glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		pycuda_ypp_fx_tilde_pbo = cuda_gl.BufferObject(long(ypp_fx_tilde_pbo))

		y_fx_pbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, y_fx_pbo)
		glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		pycuda_y_fx_pbo = cuda_gl.BufferObject(long(y_fx_pbo))

		#############
		#Flow y data#
		#############
		y_fy_tilde_pbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, y_fy_tilde_pbo)
		glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		pycuda_y_fy_tilde_pbo = cuda_gl.BufferObject(long(y_fy_tilde_pbo))

		yp_fy_tilde_pbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, yp_fy_tilde_pbo)
		glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		pycuda_yp_fy_tilde_pbo = cuda_gl.BufferObject(long(yp_fy_tilde_pbo))

		ypp_fy_tilde_pbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, ypp_fy_tilde_pbo)
		glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		pycuda_ypp_fy_tilde_pbo = cuda_gl.BufferObject(long(ypp_fy_tilde_pbo))

		y_fy_pbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, y_fy_pbo)
		glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		pycuda_y_fy_pbo = cuda_gl.BufferObject(long(y_fy_pbo))

	def _destroy_PBOs():
		global pycuda_y_tilde_pbo, y_tilde_pbo,\
		 pycuda_y_fx_tilde_pbo, y_fx_tilde_pbo,\
		 pycuda_y_fy_tilde_pbo, y_fy_tilde_pbo,\
		 pycuda_yp_tilde_pbo, yp_tilde_pbo,\
		 pycuda_yp_fx_tilde_pbo, yp_fx_tilde_pbo,\
		 pycuda_yp_fy_tilde_pbo, yp_fy_tilde_pbo,\
		 pycuda_ypp_tilde_pbo, ypp_tilde_pbo,\
		 pycuda_ypp_fx_tilde_pbo, ypp_fx_tilde_pbo,\
		 pycuda_ypp_fy_tilde_pbo, ypp_fy_tilde_pbo,\
		 pycuda_y_im_pbo, y_im_pbo,\
		 pycuda_y_fx_pbo, y_fx_pbo,\
		 pycuda_y_fy_pbo, y_fy_pbo

		for pbo in [y_tilde_pbo, yp_tilde_pbo, ypp_tilde_pbo,\
		 y_fx_tilde_pbo, yp_fx_tilde_pbo, ypp_fx_tilde_pbo,\
		 y_fy_tilde_pbo, yp_fy_tilde_pbo, ypp_fy_tilde_pbo]:
			glBindBuffer(GL_ARRAY_BUFFER, long(pbo))
			glDeleteBuffers(1, long(pbo));
			glBindBuffer(GL_ARRAY_BUFFER, 0)

		pycuda_y_tilde_pbo, y_tilde_pbo,\
		 pycuda_y_fx_tilde_pbo, y_fx_tilde_pbo,\
		 pycuda_y_fy_tilde_pbo, y_fy_tilde_pbo,\
		 pycuda_yp_tilde_pbo, yp_tilde_pbo,\
		 pycuda_yp_fx_tilde_pbo, yp_fx_tilde_pbo,\
		 pycuda_yp_fy_tilde_pbo, yp_fy_tilde_pbo,\
		 pycuda_ypp_tilde_pbo, ypp_tilde_pbo,\
		 pycuda_ypp_fx_tilde_pbo, ypp_fx_tilde_pbo,\
		 pycuda_ypp_fy_tilde_pbo, ypp_fy_tilde_pbo,\
		 pycuda_y_im_pbo, y_im_pbo,\
		 pycuda_y_fx_pbo, y_fx_pbo,\
		 pycuda_y_fy_pbo, y_fy_pbo = [None]*24

	def initjacobian(self, y_im, y_flow):
		#Copy y_im, y_fx, y_fy to GPU 
		#Copy y_tilde to GPU 
		global pycuda_y_tilde_pbo, y_tilde_pbo,\
		 pycuda_y_fx_tilde_pbo, y_fx_tilde_pbo,\
		 pycuda_y_fy_tilde_pbo, y_fy_tilde_pbo,\
		 pycuda_y_im_pbo, y_im_pbo,\
		 pycuda_y_fx_pbo, y_fx_pbo,\
		 pycuda_y_fy_pbo, y_fy_pbo

		#Tell cuda we are going to get into these buffers
		pycuda_y_tilde_pbo.unregister()
		#pycuda_y_fx_tilde_pbo.unregister()
		#pycuda_y_fy_tilde_pbo.unregister()
		pycuda_y_im_pbo.unregister()
		#pycuda_y_fx_pbo.unregister()
		#pycuda_y_fy_pbo.unregister()

		#Load y_tilde texture info
		#activate y_tilde buffer
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(y_tilde_pbo))
		#Needed? is according to http://stackoverflow.com/questions/10507215/how-to-copy-a-texture-into-a-pbo-in-pyopengl
		#glBufferData(GL_PIXEL_PACK_BUFFER_ARB,
		#         bytesize,
		#         None, self.usage)
		#read data into pbo. note: use BGRA format for optimal performance
		glEnable(GL_TEXTURE_2D)
		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_2D, self.texture.id)
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)

		#Load y_im (current frame) info from CPU memory
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(y_im_pbo))
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, self.width*self.height, y_im, GL_STREAM_DRAW)
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)

		pycuda_y_tilde_pbo = cuda_gl.BufferObject(long(y_tilde_pbo))
		#pycuda_y_fx_tilde_pbo = cuda_gl.BufferObject(long(y_tilde_pbo))
		#pycuda_y_fy_tilde_pbo = cuda_gl.BufferObject(long(y_tilde_pbo))
		pycuda_y_im_pbo = cuda_gl.BufferObject(long(y_im_pbo))
		#pycuda_y_fx_pbo = cuda_gl.BufferObject(long(y_fx_pbo))
		#pycuda_y_fy_pbo = cuda_gl.BufferObject(long(y_fy_pbo))

		z = self._process_initjac()
		return (0,0)

	def total(self):
		return self._process_total()

	def _process_initjac(self):
		"""Use PyCuda"""
		grid_dimensions = (self.width//10,self.height//10)
		y_tilde_mapping = pycuda_y_tilde_pbo.map()
		y_im_mapping = pycuda_y_im_pbo.map()
		self.cuda_initjac.prepared_call(grid_dimensions, (10, 10, 1),
				y_tilde_mapping.device_ptr(),
				y_im_mapping.device_ptr())
		cuda_driver.Context.synchronize()
		y_tilde_mapping.unmap()
		y_im_mapping.unmap()
		#Get result from CUDA...
		return 0		

	def _process_total(self):
		"""Use PyCuda"""
		#nElements = 16*1024*np.ones((1,1), dtype=np.float32)
		nElements = np.int32(16*1024)
		grid_dimensions = (16,1,1)
		a = np.random.randn(nElements).astype(np.float32)
		sum_cpu = np.sum(a)
		sum_gpu = np.zeros((16,1), dtype=np.float32)
		#self.cuda_total.prepared_call(grid_dimensions, (16, 1, 1), cuda_driver.In(a), \
		#	cuda_driver.Out(sum_gpu),cuda_driver.In(nElements))
		self.cuda_total(cuda_driver.In(a), \
			cuda_driver.Out(sum_gpu),np.uint32(nElements),\
			grid=grid_dimensions, block=(1024, 1, 1))
		cuda_driver.Context.synchronize()
		return sum_cpu, sum_gpu

#func(x_gpu.gpudata, np.uint32(N), np.uint32(h_minval), np.uint32(h_denom), block=(1024, 1, 1), grid=(number_of_blocks+1,1,1))


	def jz(self, y_im):
		global pycuda_yp_tilde_pbo, yp_tilde_pbo,\
		 pycuda_yp_fx_tilde_pbo, yp_fx_tilde_pbo,\
		 pycuda_yp_fy_tilde_pbo, yp_fy_tilde_pbo

		assert yp_tilde_pbo is not None

		#Tell cuda we are going to get into these buffers
		pycuda_yp_tilde_pbo.unregister()
		#Load yp_tilde texture info
		#activate y_tilde buffer
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(yp_tilde_pbo))
		#Needed? is according to http://stackoverflow.com/questions/10507215/how-to-copy-a-texture-into-a-pbo-in-pyopengl
		#glBufferData(GL_PIXEL_PACK_BUFFER_ARB,
		#         bytesize,
		#         None, self.usage)
		#read data into pbo. note: use BGRA format for optimal performance
		glEnable(GL_TEXTURE_2D)
		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_2D, self.texture.id)
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
		#    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
		#         w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)
		pycuda_yp_tilde_pbo = cuda_gl.BufferObject(long(yp_tilde_pbo))	

		#Do the same for flow x
		pycuda_yp_fx_tilde_pbo.unregister()
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(yp_fx_tilde_pbo))
		glEnable(GL_TEXTURE_2D)
		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_2D, self.texture.id)
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)
		pycuda_yp_fx_tilde_pbo = cuda_gl.BufferObject(long(yp_fx_tilde_pbo))	

		#Do the same for flow y
		pycuda_yp_fy_tilde_pbo.unregister()
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(yp_fy_tilde_pbo))
		glEnable(GL_TEXTURE_2D)
		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_2D, self.texture.id)
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)
		pycuda_yp_fy_tilde_pbo = cuda_gl.BufferObject(long(yp_fy_tilde_pbo))	

		#Copied perturbed image data to CUDA accessible memory, run the Cuda kernel
		z = self._process_jz()
		return z

	def _process_jz(self):
		""" Use PyCuda """
		grid_dimensions = (self.width//10,self.height//10)
		y_tilde_mapping = pycuda_y_tilde_pbo.map()
		yp_tilde_mapping = pycuda_yp_tilde_pbo.map()
		self.cuda_jz.prepared_call(grid_dimensions, (10, 10, 1),
				y_tilde_mapping.device_ptr(),
				yp_tilde_mapping.device_ptr())
		cuda_driver.Context.synchronize()
		y_tilde_mapping.unmap()
		yp_tilde_mapping.unmap()
		#Get result from CUDA...
		return 0

	############################################################################
	#CPU code###################################################################
	############################################################################
	def get_pixel_data(self):
		with self._fbo1:
			a = gloo.read_pixels()[:,:,0]
		with self._fbo2:
			b = gloo.read_pixels(out_type = np.float32)[:,:,0]
		with self._fbo3:
			c = gloo.read_pixels(out_type = np.float32)[:,:,0]
		return (a, b, c)

	def initjacobian_CPU(self, y_im, y_flow):
		(y_tilde, y_fx_tilde, y_fy_tilde) = self.get_pixel_data()
		self.z = (y_im.astype(float) - y_tilde.astype(float))/255
		self.y_tilde = y_tilde.astype(float)/255
		self.zfx = y_flow[:,:,0] - y_fx_tilde
		self.y_fx_tilde = y_fx_tilde
		self.zfy = y_flow[:,:,1] + y_fy_tilde
		self.y_fy_tilde = y_fy_tilde

	def jz_CPU(self):
		(yp_tilde, yp_fx_tilde, yp_fy_tilde) = self.get_pixel_data()
		hz = np.multiply((yp_tilde.astype(float)/255-self.y_tilde), self.z)
		hzx = np.multiply(yp_fx_tilde-self.y_fx_tilde, self.zfx)
		hzy = -np.multiply(yp_fy_tilde-self.y_fy_tilde, self.zfy)
		return np.sum(hz) + np.sum(hzx) + np.sum(hzy)

	def j_CPU(self, state, deltaX, i, j):
		state.X[i,0] += deltaX
		state.refresh()
		state.render()
		state.X[i,0] -= deltaX
		(yp_tilde, yp_fx_tilde, yp_fy_tilde) = self.get_pixel_data()
		state.X[j,0] += deltaX
		state.refresh()
		state.render()
		state.X[j,0] -= deltaX
		(ypp_tilde, ypp_fx_tilde, ypp_fy_tilde) = self.get_pixel_data()
		hz = np.multiply((yp_tilde.astype(float)/255-self.y_tilde), (ypp_tilde.astype(float)/255-self.y_tilde))
		hzx = np.multiply(yp_fx_tilde-self.y_fx_tilde, ypp_fx_tilde-self.y_fx_tilde)
		hzy = np.multiply(yp_fy_tilde-self.y_fy_tilde, ypp_fy_tilde-self.y_fy_tilde)
		return np.sum(hz)+np.sum(hzx)+np.sum(hzy)