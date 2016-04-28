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
	import pycuda.gpuarray as gpuarray
	BLOCK_SIZE = 1024
	print '...success'
except:
	print "...pycuda not installed"

from vispy import gloo 

from jinja2 import Template 

import pdb 
import cv2
from matplotlib import pyplot as plt

TEST_IMAGE = 1 
TEST_FLOWX = 2 
TEST_FLOWY = 3 

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
		
			cuda_tpl = Template("""
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

			//Sum all elements of an input array of floats...
			__global__ void initjac(unsigned char *y_tilde, unsigned char *y_im, int *output, int len) 
			{
			    // Load a segment of the input vector into shared memory
			    __shared__ int partialSum[2*{{ block_size }}];
			    int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
			    unsigned int t = threadIdx.x;
			    unsigned int start = 2*blockIdx.x*blockDim.x;
			    unsigned int stride = 1;

			    //pointwise multiplication here
			    //may need to take a stride of 4, here... will experiment and see
			    if ((start + t) < len)
			    {
			        partialSum[t] = y_tilde[start + t]*y_im[start + t];
			        //partialSum[t] = y_im[start + t];
			        //partialSum[t] = y_tilde[stride*(start + t)];
			    }
			    else
			    {       
			        partialSum[t] = 0;
			    }
			    if ((start + blockDim.x + t) < len)
			    {   
			        partialSum[blockDim.x + t] = y_tilde[start + blockDim.x + t]*y_im[start + blockDim.x + t];
			        //partialSum[blockDim.x + t] = y_im[start + blockDim.x + t];
			        //partialSum[blockDim.x + t] = y_tilde[stride*(start + blockDim.x + t)];
			    }
			    else
			    {
			        partialSum[blockDim.x + t] = 0;
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

			//Sum all elements of an input array of floats...
			__global__ void initjac_float(float *y_tilde, float *y_im, float *output, int len) 
			{
			    // Load a segment of the input vector into shared memory
			    __shared__ float partialSum[2*{{ block_size }}];
			    int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
			    unsigned int t = threadIdx.x;
			    unsigned int start = 2*blockIdx.x*blockDim.x;
			    unsigned int stride = 1;

			    //pointwise multiplication here
			    //may need to take a stride of 4, here... will experiment and see
			    if ((start + t) < len)
			    {
			        partialSum[t] = y_tilde[start + t]*y_im[start + t];
			        //partialSum[t] = y_im[start + t];
			        //partialSum[t] = y_tilde[stride*(start + t)];
			    }
			    else
			    {
			        partialSum[t] = 0.0;
			    }
			    if ((start + blockDim.x + t) < len)
			    {
			        partialSum[blockDim.x + t] = y_tilde[start + blockDim.x + t]*y_im[start + blockDim.x + t];
			        //partialSum[blockDim.x + t] = y_im[start + blockDim.x + t];
			        //partialSum[blockDim.x + t] = y_tilde[stride*(start + blockDim.x + t)];
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

			//Sum all elements of an input array of floats...
			__global__ void total(float *input, float *output, int len) 
			{
			    // Load a segment of the input vector into shared memory
			    __shared__ float partialSum[2*{{ block_size }}];
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
			""")
			cuda_source = cuda_tpl.render(block_size=BLOCK_SIZE)
			cuda_module = SourceModule(cuda_source, no_extern_c=1)
			# The argument "PPP" indicates that the zscore function will take three PBOs as arguments
			self.cuda_jz = cuda_module.get_function("jz")
			self.cuda_jz.prepare("PPP")
			self.cuda_j = cuda_module.get_function("j")
			self.cuda_j.prepare("PPP")
			self.cuda_initjac = cuda_module.get_function("initjac")
			self.cuda_initjac.prepare("PPPi")
			self.cuda_initjac_float = cuda_module.get_function("initjac_float")
			self.cuda_initjac_float.prepare("PPPi")
			self.cuda_total = cuda_module.get_function("total")
			self.cuda_total.prepare("PPP")

			# create y_tilde and y_im pixel buffer objects for processing
			self._createPBOs()

	def __del__(self):
		self._destroy_PBOs()

	def _initializePBO(self, data):
		pbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, pbo)
		glBufferData(GL_ARRAY_BUFFER, data, GL_DYNAMIC_DRAW)
		glBindBuffer(GL_ARRAY_BUFFER, 0)
		pycuda_pbo = cuda_gl.BufferObject(long(pbo))
		return (pbo, pycuda_pbo)

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
		###########
		#y_im data#
		###########
		data = np.zeros((num_texels,1),np.uint8)
		(y_tilde_pbo, pycuda_y_tilde_pbo) = self._initializePBO(data)
		(yp_tilde_pbo, pycuda_yp_tilde_pbo) = self._initializePBO(data)
		(ypp_tilde_pbo, pycuda_ypp_tilde_pbo) = self._initializePBO(data)
		(y_im_pbo, pycuda_y_im_pbo) = self._initializePBO(data)

		#############
		#Flow x data#
		#############
		data = np.zeros((num_texels,1),np.float32)
		(y_fx_tilde_pbo, pycuda_y_fx_tilde_pbo) = self._initializePBO(data)
		(yp_fx_tilde_pbo, pycuda_yp_fx_tilde_pbo) = self._initializePBO(data)
		(ypp_fx_tilde_pbo, pycuda_ypp_fx_tilde_pbo) = self._initializePBO(data)
		(y_fx_pbo, pycuda_y_fx_pbo) = self._initializePBO(data)

		#############
		#Flow y data#
		#############
		data = np.zeros((num_texels,1),np.float32)
		(y_fy_tilde_pbo, pycuda_y_fy_tilde_pbo) = self._initializePBO(data)
		(yp_fy_tilde_pbo, pycuda_yp_fy_tilde_pbo) = self._initializePBO(data)
		(ypp_fy_tilde_pbo, pycuda_ypp_fy_tilde_pbo) = self._initializePBO(data)
		(y_fy_pbo, pycuda_y_fy_pbo) = self._initializePBO(data)

	def _destroy_PBOs(self):
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
		#Copy y_im, y_fx, y_fy to GPU and copy y_tilde, y_fx_tilde, y_fy_tilde to GPU 
		global pycuda_y_tilde_pbo, y_tilde_pbo,\
		 pycuda_y_fx_tilde_pbo, y_fx_tilde_pbo,\
		 pycuda_y_fy_tilde_pbo, y_fy_tilde_pbo,\
		 pycuda_y_im_pbo, y_im_pbo,\
		 pycuda_y_fx_pbo, y_fx_pbo,\
		 pycuda_y_fy_pbo, y_fy_pbo

		#Tell cuda we are going to get into these buffers
		pycuda_y_tilde_pbo.unregister()
		pycuda_y_fx_tilde_pbo.unregister()
		pycuda_y_fy_tilde_pbo.unregister()
		pycuda_y_im_pbo.unregister()
		pycuda_y_fx_pbo.unregister()
		pycuda_y_fy_pbo.unregister()

		########################################################################
		#y_im###################################################################
		########################################################################

		#Load buffer for packing
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(y_tilde_pbo))
		bytesize = self.height*self.width
		usage = GL_STREAM_DRAW
		#Needed? is according to http://stackoverflow.com/questions/10507215/how-to-copy-a-texture-into-a-pbo-in-pyopengl
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, bytesize, None, usage)
		#Load y_tilde texture info
		glEnable(GL_TEXTURE_2D)
		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_2D, 1)#self.texture.id)
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)

		#Load y_im (current frame) info from CPU memory
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(y_im_pbo))
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, self.width*self.height, y_im, GL_STREAM_DRAW)
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)

		########################################################################
		#y_fx###################################################################
		########################################################################
		floatsize = 4 #32-bit

		#Load buffer for packing
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(y_fx_tilde_pbo))
		bytesize = self.height*self.width*floatsize
		usage = GL_STREAM_DRAW
		#Needed? is according to http://stackoverflow.com/questions/10507215/how-to-copy-a-texture-into-a-pbo-in-pyopengl
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, bytesize, None, usage)
		#Load y_tilde texture info
		glEnable(GL_TEXTURE_2D)
		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_2D, 3) #Need to guess the id :(
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, ctypes.c_void_p(0))
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)

		#Load y_fx (current frame) info from CPU memory
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(y_fx_pbo))
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, self.width*self.height*4, y_flow[:,:,0], GL_STREAM_DRAW)
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)

		########################################################################
		#y_fy###################################################################
		########################################################################

		#Load buffer for packing
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(y_fy_tilde_pbo))
		bytesize = self.height*self.width*floatsize
		usage = GL_STREAM_DRAW
		#Needed? is according to http://stackoverflow.com/questions/10507215/how-to-copy-a-texture-into-a-pbo-in-pyopengl
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, bytesize, None, usage)
		#Load y_tilde texture info
		glEnable(GL_TEXTURE_2D)
		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_2D, 4) #Need to guess the id :(
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, ctypes.c_void_p(0))
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)

		#Load y_fx (current frame) info from CPU memory
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(y_fy_pbo))
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, self.width*self.height*4, y_flow[:,:,1], GL_STREAM_DRAW)
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)

		pycuda_y_tilde_pbo = cuda_gl.BufferObject(long(y_tilde_pbo))
		pycuda_y_fx_tilde_pbo = cuda_gl.BufferObject(long(y_fx_tilde_pbo))
		pycuda_y_fy_tilde_pbo = cuda_gl.BufferObject(long(y_fy_tilde_pbo))
		pycuda_y_im_pbo = cuda_gl.BufferObject(long(y_im_pbo))
		pycuda_y_fx_pbo = cuda_gl.BufferObject(long(y_fx_pbo))
		pycuda_y_fy_pbo = cuda_gl.BufferObject(long(y_fy_pbo))

		#Check the data is loaded correctly by comparing the sum on CPU and GPU
		z_gpu = self._process_initjac(TEST_FLOWY)
		z_cpu = self.initjacobian_CPU(y_im, y_flow, test = True)
		return z_cpu, z_gpu

	def total(self):
		return self._process_total()

	def _process_initjac(self, mode=TEST_IMAGE):
		"""Use PyCuda"""
		nElements = self.width*self.height
		nBlocks = nElements/BLOCK_SIZE + 1
		print 'No. elements:', nElements
		print 'No. blocks:', nBlocks
		grid_dimensions = (nBlocks, 1)
		block_dimensions = (BLOCK_SIZE, 1, 1)

		if mode == TEST_IMAGE:
			tilde_mapping = pycuda_y_tilde_pbo.map()
			im_mapping = pycuda_y_im_pbo.map()
			kernel = self.cuda_initjac
			dtype = np.int32
		elif mode == TEST_FLOWX:
			tilde_mapping = pycuda_y_fx_tilde_pbo.map()
			im_mapping = pycuda_y_fx_pbo.map()
			kernel = self.cuda_initjac_float
			dtype = np.float32
		else:
			tilde_mapping = pycuda_y_fy_tilde_pbo.map()
			im_mapping = pycuda_y_fy_pbo.map()
			kernel = self.cuda_initjac_float
			dtype = np.float32
		
		partialsum = np.zeros((nBlocks,1), dtype=dtype)
		partialsum_gpu = gpuarray.to_gpu(partialsum)
		kernel.prepared_call(grid_dimensions, block_dimensions,\
			 tilde_mapping.device_ptr(), \
			 im_mapping.device_ptr(), partialsum_gpu.gpudata, np.uint32(nElements))
		cuda_driver.Context.synchronize()
		tilde_mapping.unmap()
		im_mapping.unmap()
		partialsum = partialsum_gpu.get()
		sum_gpu = np.sum(partialsum[0:np.ceil(nBlocks/2.)])
		return sum_gpu

	def _process_total(self):
		"""Use PyCuda"""
		nElements = np.int32(BLOCK_SIZE*16+10)
		nBlocks = nElements/BLOCK_SIZE + 1
		grid_dimensions = (nBlocks,1,1)
		a = np.random.randn(nElements).astype(np.float32)
		sum_cpu = np.sum(a)
		partialsum_gpu = np.zeros((nBlocks,1), dtype=np.float32)
		self.cuda_total(cuda_driver.In(a), cuda_driver.Out(partialsum_gpu), \
			np.uint32(nElements), grid=grid_dimensions, block=(BLOCK_SIZE, 1, 1))
		cuda_driver.Context.synchronize()
		#Sum result from GPU
		print nBlocks
		print partialsum_gpu
		sum_gpu = np.sum(partialsum_gpu[0:np.ceil(nBlocks/2.)])
		return sum_cpu, sum_gpu

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

	def initjacobian_CPU(self, y_im, y_flow, test = False):
		(y_tilde, y_fx_tilde, y_fy_tilde) = self.get_pixel_data()
		self.z = (y_im.astype(float) - y_tilde.astype(float))/255
		self.y_tilde = y_tilde.astype(float)/255
		self.zfx = y_flow[:,:,0] - y_fx_tilde
		self.y_fx_tilde = y_fx_tilde
		self.zfy = y_flow[:,:,1] + y_fy_tilde
		self.y_fy_tilde = y_fy_tilde
		if test is True:
			#Image
			#return np.sum(np.multiply(y_im,y_tilde, dtype=np.int32), dtype=np.int32)
			#return np.sum(y_tilde, dtype=np.int32)
			#return np.sum(y_im, dtype=np.int32)
			#Flow X
			#return np.sum(np.multiply(y_flow[:,:,0], y_fx_tilde, dtype=np.float32), dtype=np.float32)
			#return np.sum(y_flow[:,:,0], dtype = np.float32)
			#return np.sum(y_fx_tilde, dtype = np.float32)
			#Flow Y
			return np.sum(np.multiply(y_flow[:,:,1], y_fy_tilde, dtype=np.float32), dtype=np.float32)
			#return np.sum(y_flow[:,:,1], dtype = np.float32)
			#return np.sum(y_fy_tilde, dtype = np.float32)
		else:
			return

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