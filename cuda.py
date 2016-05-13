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
	def __init__(self, texture, texture_fx, texture_fy, fbo, fbo_fx, fbo_fy, texid, tex_fx_id, tex_fy_id, cuda):
		self.cuda = cuda 
		self.texid = texid
		self.tex_fx_id = tex_fx_id
		self.tex_fy_id = tex_fy_id
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
			//Sum all elements of an input array of floats...

			__global__ void jz(unsigned char *y_im, float *y_fx, float *y_fy,
								unsigned char *y_im_t, float *y_fx_t, float *y_fy_t,
								unsigned char *yp_im_t, float *yp_fx_t, float *yp_fy_t, 
								float *output, float *output_fx, float *output_fy, 
								int len) 
			{
			    // Load a segment of the input vector into shared memory
			    __shared__ float partialSum[2*{{ block_size }}];
			    __shared__ float partialSum_fx[2*{{ block_size }}];
			    __shared__ float partialSum_fy[2*{{ block_size }}];

			    int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
			    unsigned int t = threadIdx.x;
			    unsigned int s = 2*blockIdx.x*blockDim.x;

			    //CPU code for reference
	    		//self.z = (y_im.astype(float) - y_tilde.astype(float))/255
			  	//self.zfx = y_flow[:,:,0] - y_fx_tilde
				//self.zfy = y_flow[:,:,1] + y_fy_tilde
				//hz = np.multiply((yp_tilde.astype(float)/255-self.y_tilde), self.z)
				//hzx = np.multiply(yp_fx_tilde-self.y_fx_tilde, self.zfx)
				//hzy = -np.multiply(yp_fy_tilde-self.y_fy_tilde, self.zfy)

			    if ((s + t) < len)
			    {
			        partialSum[t] = ((float)(yp_im_t[s+t]-y_im_t[s+t]))*((float)(y_im[s+t]-y_im_t[s+t]))/255.0/255.0;
			        partialSum_fx[t] = (yp_fx_t[s+t]-y_fx_t[s+t])*(y_fx[s+t]-y_fx_t[s+t]);
			        partialSum_fy[t] = -(yp_fy_t[s+t]-y_fy_t[s+t])*(y_fy[s+t]+y_fy_t[s+t]);
			    }
			    else
			    {       
			        partialSum[t] = 0.0;
			        partialSum_fx[t] = 0.0;
			        partialSum_fy[t] = 0.0;
			    }
			    if ((s + blockDim.x + t) < len)
			    {   
			        partialSum[blockDim.x + t] = ((float)(yp_im_t[s+blockDim.x+t]-y_im_t[s+blockDim.x+t]))*((float)(y_im[s+blockDim.x+t]-y_im_t[s+blockDim.x+t]))/255.0/255.0;
			        partialSum_fx[blockDim.x + t] = (yp_fx_t[s+blockDim.x+t]-y_fx_t[s+blockDim.x+t])*(y_fx[s+blockDim.x+t]-y_fx_t[s+blockDim.x+t]);
			        partialSum_fy[blockDim.x + t] = -(yp_fy_t[s+blockDim.x+t]-y_fy_t[s+blockDim.x+t])*(y_fy[s+blockDim.x+t]+y_fy_t[s+blockDim.x+t]);
			    }
			    else
			    {
			        partialSum[blockDim.x + t] = 0.0;
			        partialSum_fx[blockDim.x + t] = 0.0;
			        partialSum_fy[blockDim.x + t] = 0.0;
			    }
			    // Traverse reduction tree
			    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
			    {
			      __syncthreads();
			        if (t < stride)
			            partialSum[t] += partialSum[t + stride];
			            partialSum_fx[t] += partialSum_fx[t + stride];
			            partialSum_fy[t] += partialSum_fy[t + stride];
			    }
			    __syncthreads();
			    // Write the computed sum of the block to the output vector at correct index
			    if (t == 0 && (globalThreadId*2) < len)
			    {
			        output[blockIdx.x] = partialSum[t];
			        output_fx[blockIdx.x] = partialSum_fx[t];
			        output_fy[blockIdx.x] = partialSum_fy[t];
			    }
			}

			__global__ void j(unsigned char *y_im_t, float *y_fx_t, float *y_fy_t,
								unsigned char *yp_im_t, float *yp_fx_t, float *yp_fy_t, 
								unsigned char *ypp_im_t, float *ypp_fx_t, float *ypp_fy_t, 
								float *output, float *output_fx, float *output_fy, 
								int len) 
			{
			    // Load a segment of the input vector into shared memory
			    __shared__ float partialSum[2*{{ block_size }}];
			    __shared__ float partialSum_fx[2*{{ block_size }}];
			    __shared__ float partialSum_fy[2*{{ block_size }}];
			    const unsigned int scale = 1;

			    int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
			    unsigned int t = threadIdx.x;
			    unsigned int s = 2*blockIdx.x*blockDim.x;

			    //CPU code for reference
				//hz = np.multiply((yp_tilde.astype(float)/255-self.y_tilde), (ypp_tilde.astype(float)/255-self.y_tilde))
				//hzx = np.multiply(yp_fx_tilde-self.y_fx_tilde, ypp_fx_tilde-self.y_fx_tilde)
				//hzy = np.multiply(yp_fy_tilde-self.y_fy_tilde, ypp_fy_tilde-self.y_fy_tilde)

			    if ((s + t) < len)
			    {
			        partialSum[t] = ((float)(yp_im_t[s+t]-y_im_t[s+t]))*((float)(ypp_im_t[s+t]-y_im_t[s+t]))/255.0/255.0;
			        partialSum_fx[t] = ((yp_fx_t[s+t]-y_fx_t[s+t])*scale)*((ypp_fx_t[s+t]-y_fx_t[s+t])*scale);
			        partialSum_fy[t] = ((yp_fy_t[s+t]-y_fy_t[s+t])*scale)*((ypp_fy_t[s+t]-y_fy_t[s+t])*scale);
			        //partialSum[t] = ((float)(yp_im_t[s+t]-y_im_t[s+t]))/255.0;
			        //partialSum_fx[t] = (ypp_fx_t[s+t]-y_fx_t[s+t]);
			        //partialSum_fy[t] = (ypp_fy_t[s+t]-y_fy_t[s+t]);
			    }
			    else
			    {       
			        partialSum[t] = 0.0;
			        partialSum_fx[t] = 0.0;
			        partialSum_fy[t] = 0.0;
			    }
			    if ((s + blockDim.x + t) < len)
			    {   
			        partialSum[blockDim.x + t] = ((float)(yp_im_t[s+blockDim.x+t]-y_im_t[s+blockDim.x+t]))*((float)(ypp_im_t[s+blockDim.x+t]-y_im_t[s+blockDim.x+t]))/255.0/255.0;
			        partialSum_fx[blockDim.x + t] = ((yp_fx_t[s+blockDim.x+t]-y_fx_t[s+blockDim.x+t])*scale)*((ypp_fx_t[s+blockDim.x+t]-y_fx_t[s+blockDim.x+t])*scale);
			        partialSum_fy[blockDim.x + t] = ((yp_fy_t[s+blockDim.x+t]-y_fy_t[s+blockDim.x+t])*scale)*((ypp_fy_t[s+blockDim.x+t]-y_fy_t[s+blockDim.x+t])*scale);
			        //partialSum[blockDim.x + t] = ((float)(yp_im_t[s+blockDim.x+t]-y_im_t[s+blockDim.x+t]))/255.0;
			        //partialSum_fx[blockDim.x + t] = (ypp_fx_t[s+blockDim.x+t]-y_fx_t[s+blockDim.x+t]);
			        //partialSum_fy[blockDim.x + t] = (ypp_fy_t[s+blockDim.x+t]-y_fy_t[s+blockDim.x+t]);
			    }
			    else
			    {
			        partialSum[blockDim.x + t] = 0.0;
			        partialSum_fx[blockDim.x + t] = 0.0;
			        partialSum_fy[blockDim.x + t] = 0.0;
			    }
			    // Traverse reduction tree
			    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
			    {
			      __syncthreads();
			        if (t < stride)
			            partialSum[t] += partialSum[t + stride];
			            partialSum_fx[t] += partialSum_fx[t + stride];
			            partialSum_fy[t] += partialSum_fy[t + stride];
			    }
			    __syncthreads();
			    // Write the computed sum of the block to the output vector at correct index
			    if (t == 0 && (globalThreadId*2) < len)
			    {
			        output[blockIdx.x] = partialSum[t];
			        output_fx[blockIdx.x] = partialSum_fx[t];
			        output_fy[blockIdx.x] = partialSum_fy[t];
			    }
			}

			//Sum all elements of an input array of floats...
			__global__ void initjac(unsigned char *y_tilde, unsigned char *y_im, float *output, int len) 
			{
			    // Load a segment of the input vector into shared memory
			    __shared__ float partialSum[2*{{ block_size }}];
			    int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
			    unsigned int t = threadIdx.x;
			    unsigned int start = 2*blockIdx.x*blockDim.x;

			    //pointwise multiplication here
			    //may need to take a stride of 4, here... will experiment and see
			    if ((start + t) < len)
			    {
			        partialSum[t] = (float)y_tilde[start + t]*(float)y_im[start + t]/255.0/255.0;
			        //partialSum[t] = y_im[start + t]/255.0;
			        //partialSum[t] = y_tilde[(start + t)]/255.0;
			    }
			    else
			    {       
			        partialSum[t] = 0;
			    }
			    if ((start + blockDim.x + t) < len)
			    {   
			        partialSum[blockDim.x + t] = (float)y_tilde[start + blockDim.x + t]*(float)y_im[start + blockDim.x + t]/255.0/255.0;
			        //partialSum[blockDim.x + t] = y_im[start + blockDim.x + t]/255.0;
			        //partialSum[blockDim.x + t] = y_tilde[(start + blockDim.x + t)]/255.0;
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

			    //pointwise multiplication here
			    //may need to take a stride of 4, here... will experiment and see
			    if ((start + t) < len)
			    {
			        partialSum[t] = y_tilde[start + t]*y_im[start + t];
			        //partialSum[t] = y_im[start + t];
			        //partialSum[t] = y_tilde[(start + t)];
			    }
			    else
			    {
			        partialSum[t] = 0.0;
			    }
			    if ((start + blockDim.x + t) < len)
			    {
			        partialSum[blockDim.x + t] = y_tilde[start + blockDim.x + t]*y_im[start + blockDim.x + t];
			        //partialSum[blockDim.x + t] = y_im[start + blockDim.x + t];
			        //partialSum[blockDim.x + t] = y_tilde[(start + blockDim.x + t)];
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
			self.cuda_jz.prepare("PPPPPPPPPPPPi")
			self.cuda_j = cuda_module.get_function("j")
			self.cuda_j.prepare("PPPPPPPPPPPPi")
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

	def _pack_texture_into_PBO(self, pbo, texid, bytesize, texformat, usage = GL_STREAM_DRAW):
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(pbo))
		#Needed? is according to http://stackoverflow.com/questions/10507215/how-to-copy-a-texture-into-a-pbo-in-pyopengl
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, bytesize, None, usage)
		glEnable(GL_TEXTURE_2D)
		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_2D, texid)
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, texformat, ctypes.c_void_p(0))
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)

	def initjacobian(self, y_im_flip, y_flow, test = False):
		y_im = np.flipud(y_im_flip)
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
		bytesize = self.height*self.width
		self._pack_texture_into_PBO(y_tilde_pbo, self.texid, bytesize, GL_UNSIGNED_BYTE)

		#Load y_im (current frame) info from CPU memory
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(y_im_pbo))
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, self.width*self.height, y_im, GL_STREAM_DRAW)
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)

		########################################################################
		#y_fx###################################################################
		########################################################################
		floatsize = 4 #32-bit
		self._pack_texture_into_PBO(y_fx_tilde_pbo, self.tex_fx_id, bytesize*floatsize, GL_FLOAT)

		#Load y_fx (current frame) info from CPU memory
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(y_fx_pbo))
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, self.width*self.height*floatsize, y_flow[:,:,0], GL_STREAM_DRAW)
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)

		########################################################################
		#y_fy###################################################################
		########################################################################

		self._pack_texture_into_PBO(y_fy_tilde_pbo, self.tex_fy_id, bytesize*floatsize, GL_FLOAT)

		#Load y_fx (current frame) info from CPU memory
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(y_fy_pbo))
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, self.width*self.height*floatsize, y_flow[:,:,1], GL_STREAM_DRAW)
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)

		pycuda_y_tilde_pbo = cuda_gl.BufferObject(long(y_tilde_pbo))
		pycuda_y_fx_tilde_pbo = cuda_gl.BufferObject(long(y_fx_tilde_pbo))
		pycuda_y_fy_tilde_pbo = cuda_gl.BufferObject(long(y_fy_tilde_pbo))
		pycuda_y_im_pbo = cuda_gl.BufferObject(long(y_im_pbo))
		pycuda_y_fx_pbo = cuda_gl.BufferObject(long(y_fx_pbo))
		pycuda_y_fy_pbo = cuda_gl.BufferObject(long(y_fy_pbo))

		#Loaded all into CUDA accessible memory, can test loaded with the following
		if test:
			return self._process_initjac_test(TEST_IMAGE)
		else:
			return None 

	def total_test(self):
		return self._process_total_test()

	def _process_initjac_test(self, mode=TEST_IMAGE):
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
			#dtype = np.uint32
			dtype = np.float32
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

	def _process_total_test(self):
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

	def jz(self):
		global pycuda_yp_tilde_pbo, yp_tilde_pbo,\
		 pycuda_yp_fx_tilde_pbo, yp_fx_tilde_pbo,\
		 pycuda_yp_fy_tilde_pbo, yp_fy_tilde_pbo

		assert yp_tilde_pbo is not None
		floatsize = 4 #32bit precision...
		bytesize = self.height*self.width

		pycuda_yp_tilde_pbo.unregister()
		pycuda_yp_fx_tilde_pbo.unregister()
		pycuda_yp_fy_tilde_pbo.unregister()

		self._pack_texture_into_PBO(yp_tilde_pbo, self.texid, bytesize, GL_UNSIGNED_BYTE)
		self._pack_texture_into_PBO(yp_fx_tilde_pbo, self.tex_fx_id, bytesize*floatsize, GL_FLOAT)
		self._pack_texture_into_PBO(yp_fy_tilde_pbo, self.tex_fy_id, bytesize*floatsize, GL_FLOAT)

		pycuda_yp_tilde_pbo = cuda_gl.BufferObject(long(yp_tilde_pbo))
		pycuda_yp_fx_tilde_pbo = cuda_gl.BufferObject(long(yp_fx_tilde_pbo))
		pycuda_yp_fy_tilde_pbo = cuda_gl.BufferObject(long(yp_fy_tilde_pbo))

		#Copied perturbed image data to CUDA accessible memory, run the Cuda kernel
		return self._process_jz()

	def _process_jz(self):
		""" Use PyCuda """
		nElements = self.width*self.height
		nBlocks = nElements/BLOCK_SIZE + 1
		#print 'No. elements:', nElements
		#print 'No. blocks:', nBlocks
		grid_dimensions = (nBlocks, 1)
		block_dimensions = (BLOCK_SIZE, 1, 1)

		im_mapping = pycuda_y_im_pbo.map()
		fx_mapping = pycuda_y_fx_pbo.map()
		fy_mapping = pycuda_y_fy_pbo.map()

		tilde_im_mapping = pycuda_y_tilde_pbo.map()
		tilde_fx_mapping = pycuda_y_fx_tilde_pbo.map()
		tilde_fy_mapping = pycuda_y_fy_tilde_pbo.map()

		p_tilde_mapping = pycuda_yp_tilde_pbo.map()
		p_tilde_fx_mapping = pycuda_yp_fx_tilde_pbo.map()
		p_tilde_fy_mapping = pycuda_yp_fy_tilde_pbo.map()
		
		partialsum = np.zeros((nBlocks,1), dtype=np.float32)
		partialsum_gpu = gpuarray.to_gpu(partialsum)
		partialsum_fx = np.zeros((nBlocks,1), dtype=np.float32)
		partialsum_fx_gpu = gpuarray.to_gpu(partialsum_fx)
		partialsum_fy = np.zeros((nBlocks,1), dtype=np.float32)
		partialsum_fy_gpu = gpuarray.to_gpu(partialsum_fy)

		#Make the call...
		self.cuda_jz.prepared_call(grid_dimensions, block_dimensions,\
			 im_mapping.device_ptr(),fx_mapping.device_ptr(),fy_mapping.device_ptr(),\
			 tilde_im_mapping.device_ptr(),tilde_fx_mapping.device_ptr(),\
			 tilde_fy_mapping.device_ptr(),p_tilde_mapping.device_ptr(),\
			 p_tilde_fx_mapping.device_ptr(),p_tilde_fy_mapping.device_ptr(),\
			 partialsum_gpu.gpudata, partialsum_fx_gpu.gpudata,\
			 partialsum_fy_gpu.gpudata, np.uint32(nElements))
		cuda_driver.Context.synchronize()

		im_mapping.unmap()
		fx_mapping.unmap()
		fy_mapping.unmap()

		tilde_im_mapping.unmap()
		tilde_fx_mapping.unmap()
		tilde_fy_mapping.unmap()

		p_tilde_mapping.unmap()
		p_tilde_fx_mapping.unmap()
		p_tilde_fy_mapping.unmap()

		#Read out the answer...
		partialsum = partialsum_gpu.get()
		partialsum_fx = partialsum_fx_gpu.get()
		partialsum_fy = partialsum_fy_gpu.get()
		sum_gpu = np.sum(partialsum[0:np.ceil(nBlocks/2.)])
		sum_fx_gpu = np.sum(partialsum_fx[0:np.ceil(nBlocks/2.)])
		sum_fy_gpu = np.sum(partialsum_fy[0:np.ceil(nBlocks/2.)])
		#print sum_gpu, sum_fx_gpu, sum_fy_gpu 
		return sum_gpu+sum_fx_gpu+sum_fy_gpu

	def j(self, state, deltaX, i, j):
		global pycuda_yp_tilde_pbo, yp_tilde_pbo,\
		 pycuda_yp_fx_tilde_pbo, yp_fx_tilde_pbo,\
		 pycuda_yp_fy_tilde_pbo, yp_fy_tilde_pbo,\
		 pycuda_ypp_tilde_pbo, ypp_tilde_pbo,\
		 pycuda_ypp_fx_tilde_pbo, ypp_fx_tilde_pbo,\
		 pycuda_ypp_fy_tilde_pbo, ypp_fy_tilde_pbo

		assert yp_tilde_pbo is not None
		floatsize = 4
		bytesize = self.height*self.width

		#Perturb first
		state.X[i,0] += deltaX
		state.refresh()
		state.render()
		state.X[i,0] -= deltaX

		#Load pixel buffers
		pycuda_yp_tilde_pbo.unregister()
		pycuda_yp_fx_tilde_pbo.unregister()
		pycuda_yp_fy_tilde_pbo.unregister()
		self._pack_texture_into_PBO(yp_tilde_pbo, self.texid, bytesize, GL_UNSIGNED_BYTE)
		self._pack_texture_into_PBO(yp_fx_tilde_pbo, self.tex_fx_id, bytesize*floatsize, GL_FLOAT)
		self._pack_texture_into_PBO(yp_fy_tilde_pbo, self.tex_fy_id, bytesize*floatsize, GL_FLOAT)
		pycuda_yp_tilde_pbo = cuda_gl.BufferObject(long(yp_tilde_pbo))
		pycuda_yp_fx_tilde_pbo = cuda_gl.BufferObject(long(yp_fx_tilde_pbo))
		pycuda_yp_fy_tilde_pbo = cuda_gl.BufferObject(long(yp_fy_tilde_pbo))

		#Perturb second
		state.X[j,0] += deltaX
		state.refresh()
		state.render()
		state.X[j,0] -= deltaX

		#Load pixel buffers
		pycuda_ypp_tilde_pbo.unregister()
		pycuda_ypp_fx_tilde_pbo.unregister()
		pycuda_ypp_fy_tilde_pbo.unregister()
		self._pack_texture_into_PBO(ypp_tilde_pbo, self.texid, bytesize, GL_UNSIGNED_BYTE)
		self._pack_texture_into_PBO(ypp_fx_tilde_pbo, self.tex_fx_id, bytesize*floatsize, GL_FLOAT)
		self._pack_texture_into_PBO(ypp_fy_tilde_pbo, self.tex_fy_id, bytesize*floatsize, GL_FLOAT)
		pycuda_ypp_tilde_pbo = cuda_gl.BufferObject(long(ypp_tilde_pbo))
		pycuda_ypp_fx_tilde_pbo = cuda_gl.BufferObject(long(ypp_fx_tilde_pbo))
		pycuda_ypp_fy_tilde_pbo = cuda_gl.BufferObject(long(ypp_fy_tilde_pbo))

		#Send to CUDA!
		return self._process_j()

	def _process_j(self):
		""" Use PyCuda """
		nElements = self.width*self.height
		nBlocks = nElements/BLOCK_SIZE + 1
		#print 'No. elements:', nElements
		#print 'No. blocks:', nBlocks
		grid_dimensions = (nBlocks, 1)
		block_dimensions = (BLOCK_SIZE, 1, 1)

		tilde_im_mapping = pycuda_y_tilde_pbo.map()
		tilde_fx_mapping = pycuda_y_fx_tilde_pbo.map()
		tilde_fy_mapping = pycuda_y_fy_tilde_pbo.map()

		p_tilde_mapping = pycuda_yp_tilde_pbo.map()
		p_tilde_fx_mapping = pycuda_yp_fx_tilde_pbo.map()
		p_tilde_fy_mapping = pycuda_yp_fy_tilde_pbo.map()
		
		pp_tilde_mapping = pycuda_ypp_tilde_pbo.map()
		pp_tilde_fx_mapping = pycuda_ypp_fx_tilde_pbo.map()
		pp_tilde_fy_mapping = pycuda_ypp_fy_tilde_pbo.map()

		partialsum = np.zeros((nBlocks,1), dtype=np.float32)
		partialsum_gpu = gpuarray.to_gpu(partialsum)
		partialsum_fx = np.zeros((nBlocks,1), dtype=np.float32)
		partialsum_fx_gpu = gpuarray.to_gpu(partialsum_fx)
		partialsum_fy = np.zeros((nBlocks,1), dtype=np.float32)
		partialsum_fy_gpu = gpuarray.to_gpu(partialsum_fy)

		#Make the call...
		self.cuda_j.prepared_call(grid_dimensions, block_dimensions,\
			 tilde_im_mapping.device_ptr(),tilde_fx_mapping.device_ptr(),\
			 tilde_fy_mapping.device_ptr(),p_tilde_mapping.device_ptr(),\
			 p_tilde_fx_mapping.device_ptr(),p_tilde_fy_mapping.device_ptr(),\
			 pp_tilde_mapping.device_ptr(),pp_tilde_fx_mapping.device_ptr(),\
			 pp_tilde_fy_mapping.device_ptr(),\
			 partialsum_gpu.gpudata, partialsum_fx_gpu.gpudata,\
			 partialsum_fy_gpu.gpudata, np.uint32(nElements))
		cuda_driver.Context.synchronize()

		tilde_im_mapping.unmap()
		tilde_fx_mapping.unmap()
		tilde_fy_mapping.unmap()

		p_tilde_mapping.unmap()
		p_tilde_fx_mapping.unmap()
		p_tilde_fy_mapping.unmap()

		pp_tilde_mapping.unmap()
		pp_tilde_fx_mapping.unmap()
		pp_tilde_fy_mapping.unmap()

		#Read out the answer...
		partialsum = partialsum_gpu.get()
		partialsum_fx = partialsum_fx_gpu.get()
		partialsum_fy = partialsum_fy_gpu.get()
		sum_gpu = np.sum(partialsum[0:np.ceil(nBlocks/2.)])
		sum_fx_gpu = np.sum(partialsum_fx[0:np.ceil(nBlocks/2.)])
		sum_fy_gpu = np.sum(partialsum_fy[0:np.ceil(nBlocks/2.)])
		scale = 1;
		#print 'j (GPU) components'
		#print sum_gpu, sum_fx_gpu/scale/scale, sum_fy_gpu/scale/scale
		return sum_gpu+sum_fx_gpu/scale/scale+sum_fy_gpu/scale/scale

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
			return np.sum(np.multiply(y_im/255.,y_tilde/255., dtype=np.float32), dtype=np.float32)
			#return np.sum(y_tilde/255., dtype=np.float32)
			#return np.sum(y_im/255., dtype=np.float32)
			#Flow X
			#return np.sum(np.multiply(y_flow[:,:,0], y_fx_tilde, dtype=np.float32), dtype=np.float32)
			#return np.sum(y_flow[:,:,0], dtype = np.float32)
			#return np.sum(y_fx_tilde, dtype = np.float32)
			#Flow Y
			#return np.sum(np.multiply(y_flow[:,:,1], y_fy_tilde, dtype=np.float32), dtype=np.float32)
			#return np.sum(y_flow[:,:,1], dtype = np.float32)
			#return np.sum(y_fy_tilde, dtype = np.float32)
		else:
			return None

	def jz_CPU(self):
		(yp_tilde, yp_fx_tilde, yp_fy_tilde) = self.get_pixel_data()
		hz = np.multiply((yp_tilde.astype(float)/255-self.y_tilde), self.z)
		hzx = np.multiply(yp_fx_tilde-self.y_fx_tilde, self.zfx)
		hzy = -np.multiply(yp_fy_tilde-self.y_fy_tilde, self.zfy)
		#print np.sum(hz), np.sum(hzx), np.sum(hzy)
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

		#Test
		hz_t = np.multiply((yp_tilde.astype(float)/255-self.y_tilde), (ypp_tilde.astype(float)/255-self.y_tilde))
		hzx_t = np.multiply(yp_fx_tilde-self.y_fx_tilde, ypp_fx_tilde-self.y_fx_tilde, dtype=np.float32)
		hzy_t = np.multiply(yp_fy_tilde-self.y_fy_tilde, ypp_fy_tilde-self.y_fy_tilde, dtype=np.float32)

		#print 'j_CPU components'
		#print np.sum(hz_t),np.sum(hzx_t),np.sum(hzy_t)
		#print np.sum(hz_t)+np.sum(hzx_t)+np.sum(hzy_t)
		#print 'No. non zero components'
		#print 'x:', np.mean(np.abs(hzx_t))
		#print 'y:', np.mean(np.abs(hzy_t))
		return np.sum(hz)+np.sum(hzx)+np.sum(hzy)

		#(yp_fx_t[s+blockDim.x+t]-y_fx_t[s+blockDim.x+t])