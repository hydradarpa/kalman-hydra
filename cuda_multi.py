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

from cuda import * 

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
import logging  
import cv2
from matplotlib import pyplot as plt

TEST_IMAGE = 1 
TEST_FLOWX = 2 
TEST_FLOWY = 3 
TEST_MASK = 4

class CUDAGL_multi(CUDAGL):
	def __init__(self, texture, texture_fx, texture_fy, texture_m, fbo, fbo_fx, fbo_fy, fbo_m, texid, tex_fx_id, tex_fy_id, tex_m_id, eps_Z, eps_J, eps_M, cuda, n, len_Q):
		self.len_Q = len_Q
		self.n = n 
		self.cuda = cuda 
		self.eps_J = eps_J 
		self.eps_Z = eps_Z
		self.eps_M = eps_M
		self.texid = texid
		self.tex_fx_id = tex_fx_id
		self.tex_fy_id = tex_fy_id
		self.tex_m_id = tex_m_id
		self._fbo1 = fbo 
		self._fbo2 = fbo_fx 
		self._fbo3 = fbo_fy 
		self._fbo4 = fbo_m 
		self.texture = texture
		self.texture_fx = texture_fx
		self.texture_fy = texture_fy
		self.texture_m = texture_m
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

			#include <stdio.h>

			//Argument below:
			//PPPPPPPPPPPPPPPPi

			__global__ void histogram_jz(unsigned char *y_im, float *y_fx, float *y_fy, unsigned char *y_m, 
								unsigned char *y_im_t, float *y_fx_t, float *y_fy_t, unsigned char *y_m_t,
								unsigned char *yp_im_t, float *yp_fx_t, float *yp_fy_t, unsigned char *yp_m_t,
								float *output, float *output_fx, float *output_fy, float *output_m,
								int len) {

				int stride = 4;
				int m, mp, fp1, fp2, fp, f1, f2, f, face;

			    float ps;
			    float ps_fx;
			    float ps_fy;
			    float ps_m;

			    float eps_J = {{ eps_J }};
			    float eps_Z = {{ eps_Z }};
			    float eps_M = {{ eps_M }};

				// pixel coordinates
				int x = blockIdx.x * blockDim.x + threadIdx.x;		
				// grid dimensions
				int nx = blockDim.x * gridDim.x; 
				// linear thread index within 2D block
				int t = threadIdx.x; 
				// total threads in 2D block
				int nt = blockDim.x; 
				// linear block index within 2D grid
				int g = blockIdx.x;

				// initialize temporary accumulation array in global memory
				float *gmem = output + g * {{ num_vertices }};
				float *gmem_fx = output_fx + g * {{ num_vertices }};
				float *gmem_fy = output_fy + g * {{ num_vertices }};
				float *gmem_m = output_m + g * {{ num_vertices }};

				for (int i = t; i < {{ num_vertices }}; i += nt) {
					gmem[i] = 0;
					gmem_fx[i] = 0;
					gmem_fy[i] = 0;
					gmem_m[i] = 0;
				}

				// process pixels
				// updates our block's partial histogram in global memory
				for (int col = x; col < len; col += nx) {
					ps = ((float)(yp_im_t[col]-y_im_t[col]))*((float)(y_im[col]-y_im_t[col]))/255.0/255.0/eps_Z;
					ps_fx = (yp_fx_t[col]-y_fx_t[col])*(y_fx[col]-y_fx_t[col])/eps_J;
					ps_fy = -(yp_fy_t[col]-y_fy_t[col])*(y_fy[col]+y_fy_t[col])/eps_J;
					ps_m = ((float)(yp_m_t[stride*col]-y_m_t[stride*col]))*((float)(y_m[col]-y_m_t[stride*col]))/255.0/255.0/eps_M;

			        //Face components from mask render
			        mp = yp_m_t[stride*col] == 255;
			        m = y_m_t[stride*col] == 255;

			        fp1 = yp_m_t[stride*col+1];
			        fp2 = yp_m_t[stride*col+2];
			        fp = 256*fp1+fp2;

			        f1 = y_m_t[stride*col+1];
			        f2 = y_m_t[stride*col+2];
			        f = 256*f1+f2;

			        face = m * f + (!m && mp) * fp;

			        if ((m || mp) && (face < (256*256-1))) {
				        //if (face > {{ num_vertices }})
						//	printf("histogram jz: Block: %d, Thread: %d, Face: %d. ",  blockIdx.x, threadIdx.x, face);
						//if (face == 34) {
						//	printf("Found face 34. ");
						//}
						atomicAdd(&gmem[face], ps);
						atomicAdd(&gmem_fx[face], ps_fx);
						atomicAdd(&gmem_fy[face], ps_fy);
						atomicAdd(&gmem_m[face], ps_m);
					}
				}
			}

			__global__ void histogram_j(unsigned char *y_im_t, float *y_fx_t, float *y_fy_t, unsigned char *y_m_t, 
								unsigned char *yp_im_t, float *yp_fx_t, float *yp_fy_t, unsigned char *yp_m_t, 
								unsigned char *ypp_im_t, float *ypp_fx_t, float *ypp_fy_t, unsigned char *ypp_m_t, 
								float *output, float *output_fx, float *output_fy, float *output_m, 
								float *output_nz, int len) {

				int stride = 4;
				int m, mp, mpp, fpp1, fpp2, fpp, fp1, fp2, fp, f1, f2, f, face;
				int scale = 1;

			    float ps;
			    float ps_fx;
			    float ps_fy;
			    float ps_m;

			    float eps_J = {{ eps_J }};
			    float eps_Z = {{ eps_Z }};
			    float eps_M = {{ eps_M }};

				// pixel coordinates
				int x = blockIdx.x * blockDim.x + threadIdx.x;		
				// grid dimensions
				int nx = blockDim.x * gridDim.x; 
				// linear thread index within 2D block
				int t = threadIdx.x; 
				// total threads in 2D block
				int nt = blockDim.x; 
				// linear block index within 2D grid
				int g = blockIdx.x;

				// initialize temporary accumulation array in global memory
				float *gmem = output + g * {{ num_q }};
				float *gmem_fx = output_fx + g * {{ num_q }};
				float *gmem_fy = output_fy + g * {{ num_q }};
				float *gmem_m = output_m + g * {{ num_q }};
				float *gmem_nz = output_nz + g * {{ num_q }};

				for (int i = t; i < {{ num_vertices }}; i += nt) {
					gmem[i] = 0;
					gmem_fx[i] = 0;
					gmem_fy[i] = 0;
					gmem_m[i] = 0;
					gmem_nz[i] = 0;
				}

				// process pixels
				// updates our block's partial histogram in global memory
				for (int col = x; col < len; col += nx) {
			        ps = ((float)(yp_im_t[col]-y_im_t[col]))*((float)(ypp_im_t[col]-y_im_t[col]))/255.0/255.0/eps_Z;
			        ps_fx = ((yp_fx_t[col]-y_fx_t[col])*scale)*((ypp_fx_t[col]-y_fx_t[col])*scale)/eps_J;
			        ps_fy = ((yp_fy_t[col]-y_fy_t[col])*scale)*((ypp_fy_t[col]-y_fy_t[col])*scale)/eps_J;

			        //Mask component
			        ps_m = ((float)(yp_m_t[stride*col]-y_m_t[stride*col]))*((float)(ypp_m_t[col*stride]-y_m_t[stride*col]))/255.0/255.0/eps_M;

			        //Face components from mask render
			        mpp = ypp_m_t[stride*col] == 255;
			        mp = yp_m_t[stride*col] == 255;
			        m = y_m_t[stride*col] == 255;

			        fpp1 = ypp_m_t[stride*col+1];
			        fpp2 = ypp_m_t[stride*col+2];
			        fpp = 256*fpp1+fpp2;

			        fp1 = yp_m_t[stride*col+1];
			        fp2 = yp_m_t[stride*col+2];
			        fp = 256*fp1+fp2;

			        f1 = y_m_t[stride*col+1];
			        f2 = y_m_t[stride*col+2];
			        f = 256*f1+f2;

			        //if (f == 110 || fp == 110 || fpp == 100) {
					//		printf("f = %d, fp = %d, fpp = %d. ", f, fp, fpp);
					//}

			        face = m*f + (!m)*(mp*fp + (!mp && mpp)*fpp);

			        if ((m || mp || mpp) && (face < (256*256-1))) {
				        //if (face > {{ num_q }})
						//	printf("Block: %d, Thread: %d, Face: %d. ",  blockIdx.x, threadIdx.x, face);

						atomicAdd(&gmem[face], ps);
						atomicAdd(&gmem_fx[face], ps_fx);
						atomicAdd(&gmem_fy[face], ps_fy);
						atomicAdd(&gmem_m[face], ps_m);
						atomicAdd(&gmem_nz[face], 1);
					}
				}
			}

			__global__ void j(unsigned char *y_im_t, float *y_fx_t, float *y_fy_t, unsigned char *y_m_t, 
								unsigned char *yp_im_t, float *yp_fx_t, float *yp_fy_t, unsigned char *yp_m_t, 
								unsigned char *ypp_im_t, float *ypp_fx_t, float *ypp_fy_t, unsigned char *ypp_m_t, 
								float *output, float *output_fx, float *output_fy, float *output_m, 
								int len) 
			{
			    // Load a segment of the input vector into shared memory
			    __shared__ float partialSum[2*{{ block_size }}];
			    __shared__ float partialSum_fx[2*{{ block_size }}];
			    __shared__ float partialSum_fy[2*{{ block_size }}];
			    __shared__ float partialSum_m[2*{{ block_size }}];
			    const unsigned int scale = 1;
			    const unsigned int stride = 4;
			    int idx;

			    float eps_J = {{ eps_J }};
			    float eps_Z = {{ eps_Z }};
			    float eps_M = {{ eps_M }};

			    int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
			    unsigned int t = threadIdx.x;
			    unsigned int s = 2*blockIdx.x*blockDim.x;

			    //CPU code for reference
				//hz = np.multiply((yp_tilde.astype(float)/255-self.y_tilde), (ypp_tilde.astype(float)/255-self.y_tilde))
				//hzx = np.multiply(yp_fx_tilde-self.y_fx_tilde, ypp_fx_tilde-self.y_fx_tilde)
				//hzy = np.multiply(yp_fy_tilde-self.y_fy_tilde, ypp_fy_tilde-self.y_fy_tilde)

			    if ((s + t) < len)
			    {
			    	idx = stride*(s+t);
			        partialSum[t] = ((float)(yp_im_t[s+t]-y_im_t[s+t]))*((float)(ypp_im_t[s+t]-y_im_t[s+t]))/255.0/255.0/eps_Z;
			        partialSum_fx[t] = ((yp_fx_t[s+t]-y_fx_t[s+t])*scale)*((ypp_fx_t[s+t]-y_fx_t[s+t])*scale)/eps_J;
			        partialSum_fy[t] = ((yp_fy_t[s+t]-y_fy_t[s+t])*scale)*((ypp_fy_t[s+t]-y_fy_t[s+t])*scale)/eps_J;
			        partialSum_m[t] = ((float)(yp_m_t[idx]-y_m_t[idx]))*((float)(ypp_m_t[idx]-y_m_t[idx]))/255.0/255.0/eps_M;

			        //partialSum[t] = ((float)(yp_im_t[s+t]-y_im_t[s+t]))/255.0;
			        //partialSum_fx[t] = (ypp_fx_t[s+t]-y_fx_t[s+t]);
			        //partialSum_fy[t] = (ypp_fy_t[s+t]-y_fy_t[s+t]);
			    }
			    else
			    {       
			        partialSum[t] = 0.0;
			        partialSum_fx[t] = 0.0;
			        partialSum_fy[t] = 0.0;
			        partialSum_m[t] = 0.0;
			    }
			    if ((s + blockDim.x + t) < len)
			    {   
			    	idx = stride*(s+t+blockDim.x);
			        partialSum[blockDim.x + t] = ((float)(yp_im_t[s+blockDim.x+t]-y_im_t[s+blockDim.x+t]))*((float)(ypp_im_t[s+blockDim.x+t]-y_im_t[s+blockDim.x+t]))/255.0/255.0/eps_Z;
			        partialSum_fx[blockDim.x + t] = ((yp_fx_t[s+blockDim.x+t]-y_fx_t[s+blockDim.x+t])*scale)*((ypp_fx_t[s+blockDim.x+t]-y_fx_t[s+blockDim.x+t])*scale)/eps_J;
			        partialSum_fy[blockDim.x + t] = ((yp_fy_t[s+blockDim.x+t]-y_fy_t[s+blockDim.x+t])*scale)*((ypp_fy_t[s+blockDim.x+t]-y_fy_t[s+blockDim.x+t])*scale)/eps_J;
			        partialSum_m[blockDim.x + t] = ((float)(yp_m_t[idx]-y_m_t[idx]))*((float)(ypp_m_t[idx]-y_m_t[idx]))/255.0/255.0/eps_M;

			        //partialSum[blockDim.x + t] = ((float)(yp_im_t[s+blockDim.x+t]-y_im_t[s+blockDim.x+t]))/255.0;
			        //partialSum_fx[blockDim.x + t] = (ypp_fx_t[s+blockDim.x+t]-y_fx_t[s+blockDim.x+t]);
			        //partialSum_fy[blockDim.x + t] = (ypp_fy_t[s+blockDim.x+t]-y_fy_t[s+blockDim.x+t]);
			    }
			    else
			    {
			        partialSum[blockDim.x + t] = 0.0;
			        partialSum_fx[blockDim.x + t] = 0.0;
			        partialSum_fy[blockDim.x + t] = 0.0;
			        partialSum_m[blockDim.x + t] = 0.0;
			    }
			    // Traverse reduction tree
			    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
			    {
			      __syncthreads();
			        if (t < stride)
			            partialSum[t] += partialSum[t + stride];
			            partialSum_fx[t] += partialSum_fx[t + stride];
			            partialSum_fy[t] += partialSum_fy[t + stride];
			            partialSum_m[t] += partialSum_m[t + stride];
			    }
			    __syncthreads();
			    // Write the computed sum of the block to the output vector at correct index
			    if (t == 0 && (globalThreadId*2) < len)
			    {
			        output[blockIdx.x] = partialSum[t];
			        output_fx[blockIdx.x] = partialSum_fx[t];
			        output_fy[blockIdx.x] = partialSum_fy[t];
			        output_m[blockIdx.x] = partialSum_m[t];
			    }
			}

			__global__ void jz(unsigned char *y_im, float *y_fx, float *y_fy, unsigned char *y_m, 
								unsigned char *y_im_t, float *y_fx_t, float *y_fy_t, unsigned char *y_m_t,
								unsigned char *yp_im_t, float *yp_fx_t, float *yp_fy_t, unsigned char *yp_m_t,
								float *output, float *output_fx, float *output_fy, float *output_m,
								int len) 
			{
			    // Load a segment of the input vector into shared memory
			    __shared__ float partialSum[2*{{ block_size }}];
			    __shared__ float partialSum_fx[2*{{ block_size }}];
			    __shared__ float partialSum_fy[2*{{ block_size }}];
			    __shared__ float partialSum_m[2*{{ block_size }}];

			    float eps_J = {{ eps_J }};
			    float eps_Z = {{ eps_Z }};
			    float eps_M = {{ eps_M }};

			    unsigned int idx;
			    unsigned const int stride = 4;

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
			    	idx = stride*(s+t);
			        partialSum[t] = ((float)(yp_im_t[s+t]-y_im_t[s+t]))*((float)(y_im[s+t]-y_im_t[s+t]))/255.0/255.0/eps_Z;
			        partialSum_fx[t] = (yp_fx_t[s+t]-y_fx_t[s+t])*(y_fx[s+t]-y_fx_t[s+t])/eps_J;
			        partialSum_fy[t] = -(yp_fy_t[s+t]-y_fy_t[s+t])*(y_fy[s+t]+y_fy_t[s+t])/eps_J;
			        partialSum_m[t] = ((float)(yp_m_t[idx]-y_m_t[idx]))*((float)(y_m[s+t]-y_m_t[idx]))/255.0/255.0/eps_M;
			    }
			    else
			    {       
			        partialSum[t] = 0.0;
			        partialSum_fx[t] = 0.0;
			        partialSum_fy[t] = 0.0;
			        partialSum_m[t] = 0.0;
			    }
			    if ((s + blockDim.x + t) < len)
			    {   
			    	idx = stride*(s+t+blockDim.x);
			        partialSum[blockDim.x + t] = ((float)(yp_im_t[s+blockDim.x+t]-y_im_t[s+blockDim.x+t]))*((float)(y_im[s+blockDim.x+t]-y_im_t[s+blockDim.x+t]))/255.0/255.0/eps_Z;
			        partialSum_fx[blockDim.x + t] = (yp_fx_t[s+blockDim.x+t]-y_fx_t[s+blockDim.x+t])*(y_fx[s+blockDim.x+t]-y_fx_t[s+blockDim.x+t])/eps_J;
			        partialSum_fy[blockDim.x + t] = -(yp_fy_t[s+blockDim.x+t]-y_fy_t[s+blockDim.x+t])*(y_fy[s+blockDim.x+t]+y_fy_t[s+blockDim.x+t])/eps_J;
			        partialSum_m[blockDim.x + t] = ((float)(yp_m_t[idx]-y_m_t[idx]))*((float)(y_m[s+blockDim.x+t]-y_m_t[idx]))/255.0/255.0/eps_M;
			    }
			    else
			    {
			        partialSum[blockDim.x + t] = 0.0;
			        partialSum_fx[blockDim.x + t] = 0.0;
			        partialSum_fy[blockDim.x + t] = 0.0;
			        partialSum_m[blockDim.x + t] = 0.0;
			    }
			    // Traverse reduction tree
			    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
			    {
			      __syncthreads();
			        if (t < stride)
			            partialSum[t] += partialSum[t + stride];
			            partialSum_fx[t] += partialSum_fx[t + stride];
			            partialSum_fy[t] += partialSum_fy[t + stride];
			            partialSum_m[t] += partialSum_m[t + stride];
			    }
			    __syncthreads();
			    // Write the computed sum of the block to the output vector at correct index
			    if (t == 0 && (globalThreadId*2) < len)
			    {
			        output[blockIdx.x] = partialSum[t];
			        output_fx[blockIdx.x] = partialSum_fx[t];
			        output_fy[blockIdx.x] = partialSum_fy[t];
			        output_m[blockIdx.x] = partialSum_m[t];
			    }
			}
			}
			""")
			cuda_source = cuda_tpl.render(block_size=BLOCK_SIZE, eps_J = self.eps_J, eps_Z = self.eps_Z, eps_M = self.eps_M, num_vertices = self.n, num_q = self.len_Q)
			cuda_module = SourceModule(cuda_source, no_extern_c=1)
			# The argument "PPP" indicates that the zscore function will take three PBOs as arguments
			self.cuda_histjz = cuda_module.get_function("histogram_jz")
			self.cuda_histjz.prepare("PPPPPPPPPPPPPPPPi")
			self.cuda_histj = cuda_module.get_function("histogram_j")
			self.cuda_histj.prepare("PPPPPPPPPPPPPPPPPi")
			self.cuda_j = cuda_module.get_function("j")
			self.cuda_j.prepare("PPPPPPPPPPPPPPPPi")			
			self.cuda_jz = cuda_module.get_function("jz")
			self.cuda_jz.prepare("PPPPPPPPPPPPPPPPi")			
			self._createPBOs()

	def _createPBOs(self):
		num_texels = self.width*self.height
		###########
		#y_im data#
		###########
		data = np.zeros((num_texels,1),np.uint8)
		(self.y_tilde_pbo, self.pycuda_y_tilde_pbo) = self._initializePBO(data)
		(self.yp_tilde_pbo, self.pycuda_yp_tilde_pbo) = self._initializePBO(data)
		(self.ypp_tilde_pbo, self.pycuda_ypp_tilde_pbo) = self._initializePBO(data)
		(self.y_im_pbo, self.pycuda_y_im_pbo) = self._initializePBO(data)

		#############
		#Flow x data#
		#############
		data = np.zeros((num_texels,1),np.float32)
		(self.y_fx_tilde_pbo, self.pycuda_y_fx_tilde_pbo) = self._initializePBO(data)
		(self.yp_fx_tilde_pbo, self.pycuda_yp_fx_tilde_pbo) = self._initializePBO(data)
		(self.ypp_fx_tilde_pbo, self.pycuda_ypp_fx_tilde_pbo) = self._initializePBO(data)
		(self.y_fx_pbo, self.pycuda_y_fx_pbo) = self._initializePBO(data)

		#############
		#Flow y data#
		#############
		data = np.zeros((num_texels,1),np.float32)
		(self.y_fy_tilde_pbo, self.pycuda_y_fy_tilde_pbo) = self._initializePBO(data)
		(self.yp_fy_tilde_pbo, self.pycuda_yp_fy_tilde_pbo) = self._initializePBO(data)
		(self.ypp_fy_tilde_pbo, self.pycuda_ypp_fy_tilde_pbo) = self._initializePBO(data)
		(self.y_fy_pbo, self.pycuda_y_fy_pbo) = self._initializePBO(data)

		###########
		#Mask data#
		###########
		data = np.zeros((num_texels,1),np.uint8)
		(self.y_m_tilde_pbo, self.pycuda_y_m_tilde_pbo) = self._initializePBO(data)
		(self.yp_m_tilde_pbo, self.pycuda_yp_m_tilde_pbo) = self._initializePBO(data)
		(self.ypp_m_tilde_pbo, self.pycuda_ypp_m_tilde_pbo) = self._initializePBO(data)
		(self.y_m_pbo, self.pycuda_y_m_pbo) = self._initializePBO(data)

	#def _del_PBO(self, pbo):
	#	glBindBuffer(GL_ARRAY_BUFFER, long(pbo))
	#	glDeleteBuffers(1, long(pbo))
	#	glBindBuffer(GL_ARRAY_BUFFER, 0)

	#def _destroy_PBOs(self):
	#	global pycuda_y_tilde_pbo, y_tilde_pbo,\
	#	 pycuda_y_fx_tilde_pbo, y_fx_tilde_pbo,\
	#	 pycuda_y_fy_tilde_pbo, y_fy_tilde_pbo,\
	#	 pycuda_y_m_tilde_pbo, y_m_tilde_pbo,\
	#	 pycuda_yp_tilde_pbo, yp_tilde_pbo,\
	#	 pycuda_yp_fx_tilde_pbo, yp_fx_tilde_pbo,\
	#	 pycuda_yp_fy_tilde_pbo, yp_fy_tilde_pbo,\
	#	 pycuda_yp_m_tilde_pbo, yp_m_tilde_pbo,\
	#	 pycuda_ypp_tilde_pbo, ypp_tilde_pbo,\
	#	 pycuda_ypp_fx_tilde_pbo, ypp_fx_tilde_pbo,\
	#	 pycuda_ypp_fy_tilde_pbo, ypp_fy_tilde_pbo,\
	#	 pycuda_ypp_m_tilde_pbo, ypp_m_tilde_pbo,\
	#	 pycuda_y_im_pbo, y_im_pbo,\
	#	 pycuda_y_fx_pbo, y_fx_pbo,\
	#	 pycuda_y_fy_pbo, y_fy_pbo,\
	#	 pycuda_y_m_pbo, y_m_pbo

	#	print 'Deleting PBOs'

	#	for pbo in [y_tilde_pbo, yp_tilde_pbo, ypp_tilde_pbo,\
	#	 y_fx_tilde_pbo, yp_fx_tilde_pbo, ypp_fx_tilde_pbo,\
	#	 y_fy_tilde_pbo, yp_fy_tilde_pbo, ypp_fy_tilde_pbo,\
	#	 y_m_tilde_pbo, yp_m_tilde_pbo, ypp_m_tilde_pbo]:
	#		try:
	#			self._del_PBO(pbo)
	#		except TypeError:
	#			print 'Passing' 

	#	pycuda_y_tilde_pbo, y_tilde_pbo, \
	#	 pycuda_y_fx_tilde_pbo, y_fx_tilde_pbo,\
	#	 pycuda_y_fy_tilde_pbo, y_fy_tilde_pbo,\
	#	 pycuda_y_m_tilde_pbo, y_m_tilde_pbo,\
	#	 pycuda_yp_tilde_pbo, yp_tilde_pbo,\
	#	 pycuda_yp_fx_tilde_pbo, yp_fx_tilde_pbo,\
	#	 pycuda_yp_fy_tilde_pbo, yp_fy_tilde_pbo,\
	#	 pycuda_yp_m_tilde_pbo, yp_m_tilde_pbo,\
	#	 pycuda_ypp_tilde_pbo, ypp_tilde_pbo,\
	#	 pycuda_ypp_fx_tilde_pbo, ypp_fx_tilde_pbo,\
	#	 pycuda_ypp_fy_tilde_pbo, ypp_fy_tilde_pbo,\
	#	 pycuda_ypp_m_tilde_pbo, ypp_m_tilde_pbo,\
	#	 pycuda_y_im_pbo, y_im_pbo,\
	#	 pycuda_y_fx_pbo, y_fx_pbo,\
	#	 pycuda_y_fy_pbo, y_fy_pbo,\
	#	 pycuda_y_m_pbo, y_m_pbo = [None]*32

	def initjacobian(self, y_im_flip, y_flow, y_m_flip, test = False):
		#Copy y_im, y_fx, y_fy to GPU and copy y_tilde, y_fx_tilde, y_fy_tilde to GPU 
		#print pycuda_y_tilde_pbo 

		y_im = np.flipud(y_im_flip)
		y_m = np.flipud(y_m_flip)
		yfx = np.flipud(y_flow[:,:,0])
		yfy = np.flipud(y_flow[:,:,0])
		y_flow = np.dstack((yfx,yfy))

		#Tell cuda we are going to get into these buffers
		self.pycuda_y_tilde_pbo.unregister()
		self.pycuda_y_fx_tilde_pbo.unregister()
		self.pycuda_y_fy_tilde_pbo.unregister()
		self.pycuda_y_m_tilde_pbo.unregister()
		self.pycuda_y_im_pbo.unregister()
		self.pycuda_y_fx_pbo.unregister()
		self.pycuda_y_fy_pbo.unregister()
		self.pycuda_y_m_pbo.unregister()

		########################################################################
		#y_im###################################################################
		########################################################################

		#Load buffer for packing
		bytesize = self.height*self.width
		self._pack_texture_into_PBO(self.y_tilde_pbo, self.texid, bytesize, GL_UNSIGNED_BYTE)

		#Load y_im (current frame) info from CPU memory
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(self.y_im_pbo))
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, self.width*self.height, y_im, GL_STREAM_DRAW)
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)

		########################################################################
		#y_fx###################################################################
		########################################################################
		floatsize = 4 #32-bit
		self._pack_texture_into_PBO(self.y_fx_tilde_pbo, self.tex_fx_id, bytesize*floatsize, GL_FLOAT)

		#Load y_fx (current frame) info from CPU memory
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(self.y_fx_pbo))
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, self.width*self.height*floatsize, y_flow[:,:,0], GL_STREAM_DRAW)
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)

		########################################################################
		#y_fy###################################################################
		########################################################################

		self._pack_texture_into_PBO(self.y_fy_tilde_pbo, self.tex_fy_id, bytesize*floatsize, GL_FLOAT)

		#Load y_fx (current frame) info from CPU memory
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(self.y_fy_pbo))
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, self.width*self.height*floatsize, y_flow[:,:,1], GL_STREAM_DRAW)
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)

		########################################################################
		#y_m####################################################################
		########################################################################

		#Load buffer for packing
		rgbsize = 4 #32-bit
		bytesize = self.height*self.width
		self._pack_texture_into_PBO(self.y_m_tilde_pbo, self.tex_m_id, bytesize*rgbsize, GL_UNSIGNED_BYTE, imageformat = GL_RGBA)

		#Load y_m (current frame) info from CPU memory
		glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, long(self.y_m_pbo))
		glBufferData(GL_PIXEL_PACK_BUFFER_ARB, self.width*self.height, y_m, GL_STREAM_DRAW)
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0)
		glDisable(GL_TEXTURE_2D)

		self.pycuda_y_tilde_pbo = cuda_gl.BufferObject(long(self.y_tilde_pbo))
		self.pycuda_y_fx_tilde_pbo = cuda_gl.BufferObject(long(self.y_fx_tilde_pbo))
		self.pycuda_y_fy_tilde_pbo = cuda_gl.BufferObject(long(self.y_fy_tilde_pbo))
		self.pycuda_y_m_tilde_pbo = cuda_gl.BufferObject(long(self.y_m_tilde_pbo))
		self.pycuda_y_im_pbo = cuda_gl.BufferObject(long(self.y_im_pbo))
		self.pycuda_y_fx_pbo = cuda_gl.BufferObject(long(self.y_fx_pbo))
		self.pycuda_y_fy_pbo = cuda_gl.BufferObject(long(self.y_fy_pbo))
		self.pycuda_y_m_pbo = cuda_gl.BufferObject(long(self.y_m_pbo))

		#Loaded all into CUDA accessible memory, can test loaded with the following
		if test:
			return self._process_initjac_test(TEST_IMAGE)
		else:
			return None 

	def jz(self, state):
		assert self.yp_tilde_pbo is not None
		floatsize = 4 #number of bytes, 32bit precision...
		rgbsize = 4 #32-bit
		bytesize = self.height*self.width

		#state.refresh()
		#state.render()

		self.pycuda_yp_tilde_pbo.unregister()
		self.pycuda_yp_fx_tilde_pbo.unregister()
		self.pycuda_yp_fy_tilde_pbo.unregister()
		self.pycuda_yp_m_tilde_pbo.unregister()

		self._pack_texture_into_PBO(self.yp_tilde_pbo, self.texid, bytesize, GL_UNSIGNED_BYTE)
		self._pack_texture_into_PBO(self.yp_fx_tilde_pbo, self.tex_fx_id, bytesize*floatsize, GL_FLOAT)
		self._pack_texture_into_PBO(self.yp_fy_tilde_pbo, self.tex_fy_id, bytesize*floatsize, GL_FLOAT)
		self._pack_texture_into_PBO(self.yp_m_tilde_pbo, self.tex_m_id, bytesize*rgbsize, GL_UNSIGNED_BYTE, imageformat = GL_RGBA)

		self.pycuda_yp_tilde_pbo = cuda_gl.BufferObject(long(self.yp_tilde_pbo))
		self.pycuda_yp_fx_tilde_pbo = cuda_gl.BufferObject(long(self.yp_fx_tilde_pbo))
		self.pycuda_yp_fy_tilde_pbo = cuda_gl.BufferObject(long(self.yp_fy_tilde_pbo))
		self.pycuda_yp_m_tilde_pbo = cuda_gl.BufferObject(long(self.yp_m_tilde_pbo))

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

		im_mapping = self.pycuda_y_im_pbo.map()
		fx_mapping = self.pycuda_y_fx_pbo.map()
		fy_mapping = self.pycuda_y_fy_pbo.map()
		m_mapping = self.pycuda_y_m_pbo.map()

		tilde_im_mapping = self.pycuda_y_tilde_pbo.map()
		tilde_fx_mapping = self.pycuda_y_fx_tilde_pbo.map()
		tilde_fy_mapping = self.pycuda_y_fy_tilde_pbo.map()
		tilde_m_mapping = self.pycuda_y_m_tilde_pbo.map()

		p_tilde_mapping = self.pycuda_yp_tilde_pbo.map()
		p_tilde_fx_mapping = self.pycuda_yp_fx_tilde_pbo.map()
		p_tilde_fy_mapping = self.pycuda_yp_fy_tilde_pbo.map()
		p_tilde_m_mapping = self.pycuda_yp_m_tilde_pbo.map()
		
		partialsum = np.zeros((nBlocks,1), dtype=np.float32)
		partialsum_gpu = gpuarray.to_gpu(partialsum)
		partialsum_fx = np.zeros((nBlocks,1), dtype=np.float32)
		partialsum_fx_gpu = gpuarray.to_gpu(partialsum_fx)
		partialsum_fy = np.zeros((nBlocks,1), dtype=np.float32)
		partialsum_fy_gpu = gpuarray.to_gpu(partialsum_fy)
		partialsum_m = np.zeros((nBlocks,1), dtype=np.float32)
		partialsum_m_gpu = gpuarray.to_gpu(partialsum_m)


		#CUDA definition:
		#__global__ void jz(unsigned char *y_im, float *y_fx, float *y_fy, unsigned char *y_m, 
		#						unsigned char *y_im_t, float *y_fx_t, float *y_fy_t, unsigned char *y_m_t,
		#						unsigned char *yp_im_t, float *yp_fx_t, float *yp_fy_t, unsigned char *yp_m_t,
		#						float *output, float *output_fx, float *output_fy, float *output_m,
		#						int len) 

		#Make the call...
		cuda_driver.Context.synchronize()
		self.cuda_jz.prepared_call(grid_dimensions, block_dimensions,\
			 im_mapping.device_ptr(),fx_mapping.device_ptr(),\
			 fy_mapping.device_ptr(),m_mapping.device_ptr(),\
			 tilde_im_mapping.device_ptr(),tilde_fx_mapping.device_ptr(),\
			 tilde_fy_mapping.device_ptr(),tilde_m_mapping.device_ptr(),\
			 p_tilde_mapping.device_ptr(),p_tilde_fx_mapping.device_ptr(),\
			 p_tilde_fy_mapping.device_ptr(),p_tilde_m_mapping.device_ptr(),\
			 partialsum_gpu.gpudata, partialsum_fx_gpu.gpudata,\
			 partialsum_fy_gpu.gpudata, partialsum_m_gpu.gpudata,\
			 np.uint32(nElements))
		cuda_driver.Context.synchronize()

		im_mapping.unmap()
		fx_mapping.unmap()
		fy_mapping.unmap()
		m_mapping.unmap()

		tilde_im_mapping.unmap()
		tilde_fx_mapping.unmap()
		tilde_fy_mapping.unmap()
		tilde_m_mapping.unmap()

		p_tilde_mapping.unmap()
		p_tilde_fx_mapping.unmap()
		p_tilde_fy_mapping.unmap()
		p_tilde_m_mapping.unmap()

		#Read out the answer...
		partialsum = partialsum_gpu.get()
		partialsum_fx = partialsum_fx_gpu.get()
		partialsum_fy = partialsum_fy_gpu.get()
		partialsum_m = partialsum_m_gpu.get()
		sum_gpu = np.sum(partialsum[0:np.ceil(nBlocks/2.)])
		sum_fx_gpu = np.sum(partialsum_fx[0:np.ceil(nBlocks/2.)])
		sum_fy_gpu = np.sum(partialsum_fy[0:np.ceil(nBlocks/2.)])
		sum_m_gpu = np.sum(partialsum_m[0:np.ceil(nBlocks/2.)])
		#print 'GPU', sum_gpu, sum_fx_gpu, sum_fy_gpu 
		#return sum_gpu+sum_fx_gpu+sum_fy_gpu+sum_m_gpu
		jzc = np.array([sum_gpu,sum_fx_gpu,sum_fy_gpu,sum_m_gpu])
		return (sum_gpu+sum_fx_gpu+sum_fy_gpu+sum_m_gpu, jzc)

	def jz_multi(self, state):
		#print 'jz_multi: pycuda_yp_tilde_pbo', pycuda_yp_tilde_pbo 
		#print dir(pycuda_yp_tilde_pbo)

		assert self.yp_tilde_pbo is not None
		floatsize = 4 #number of bytes, 32bit precision...
		rgbsize = 4 #32-bit
		bytesize = self.height*self.width

		#state.refresh()
		#state.render()

		self.pycuda_yp_tilde_pbo.unregister()
		self.pycuda_yp_fx_tilde_pbo.unregister()
		self.pycuda_yp_fy_tilde_pbo.unregister()
		self.pycuda_yp_m_tilde_pbo.unregister()

		self._pack_texture_into_PBO(self.yp_tilde_pbo, self.texid, bytesize, GL_UNSIGNED_BYTE)
		self._pack_texture_into_PBO(self.yp_fx_tilde_pbo, self.tex_fx_id, bytesize*floatsize, GL_FLOAT)
		self._pack_texture_into_PBO(self.yp_fy_tilde_pbo, self.tex_fy_id, bytesize*floatsize, GL_FLOAT)
		self._pack_texture_into_PBO(self.yp_m_tilde_pbo, self.tex_m_id, bytesize*rgbsize, GL_UNSIGNED_BYTE, imageformat = GL_RGBA)

		self.pycuda_yp_tilde_pbo = cuda_gl.BufferObject(long(self.yp_tilde_pbo))
		self.pycuda_yp_fx_tilde_pbo = cuda_gl.BufferObject(long(self.yp_fx_tilde_pbo))
		self.pycuda_yp_fy_tilde_pbo = cuda_gl.BufferObject(long(self.yp_fy_tilde_pbo))
		self.pycuda_yp_m_tilde_pbo = cuda_gl.BufferObject(long(self.yp_m_tilde_pbo))

		#Copied perturbed image data to CUDA accessible memory, run the Cuda kernel
		return self._process_jz_multi()

	def _process_jz_multi(self):
		""" Use PyCuda """
		nElements = self.width*self.height
		nBlocks = nElements/BLOCK_SIZE + 1
		#print 'No. elements:', nElements
		#print 'No. blocks:', nBlocks
		grid_dimensions = (nBlocks, 1)
		block_dimensions = (BLOCK_SIZE, 1, 1)

		im_mapping = self.pycuda_y_im_pbo.map()
		fx_mapping = self.pycuda_y_fx_pbo.map()
		fy_mapping = self.pycuda_y_fy_pbo.map()
		m_mapping = self.pycuda_y_m_pbo.map()

		tilde_im_mapping = self.pycuda_y_tilde_pbo.map()
		tilde_fx_mapping = self.pycuda_y_fx_tilde_pbo.map()
		tilde_fy_mapping = self.pycuda_y_fy_tilde_pbo.map()
		tilde_m_mapping = self.pycuda_y_m_tilde_pbo.map()

		p_tilde_mapping = self.pycuda_yp_tilde_pbo.map()
		p_tilde_fx_mapping = self.pycuda_yp_fx_tilde_pbo.map()
		p_tilde_fy_mapping = self.pycuda_yp_fy_tilde_pbo.map()
		p_tilde_m_mapping = self.pycuda_yp_m_tilde_pbo.map()
		
		partialsum = np.zeros((nBlocks*self.n,1), dtype=np.float32)
		partialsum_gpu = gpuarray.to_gpu(partialsum)
		partialsum_fx = np.zeros((nBlocks*self.n,1), dtype=np.float32)
		partialsum_fx_gpu = gpuarray.to_gpu(partialsum_fx)
		partialsum_fy = np.zeros((nBlocks*self.n,1), dtype=np.float32)
		partialsum_fy_gpu = gpuarray.to_gpu(partialsum_fy)
		partialsum_m = np.zeros((nBlocks*self.n,1), dtype=np.float32)
		partialsum_m_gpu = gpuarray.to_gpu(partialsum_m)


		#CUDA definition:
		#__global__ void jz(unsigned char *y_im, float *y_fx, float *y_fy, unsigned char *y_m, 
		#						unsigned char *y_im_t, float *y_fx_t, float *y_fy_t, unsigned char *y_m_t,
		#						unsigned char *yp_im_t, float *yp_fx_t, float *yp_fy_t, unsigned char *yp_m_t,
		#						float *output, float *output_fx, float *output_fy, float *output_m,
		#						int len) 

		#Make the call...
		cuda_driver.Context.synchronize()
		self.cuda_histjz.prepared_call(grid_dimensions, block_dimensions,\
			 im_mapping.device_ptr(),fx_mapping.device_ptr(),\
			 fy_mapping.device_ptr(),m_mapping.device_ptr(),\
			 tilde_im_mapping.device_ptr(),tilde_fx_mapping.device_ptr(),\
			 tilde_fy_mapping.device_ptr(),tilde_m_mapping.device_ptr(),\
			 p_tilde_mapping.device_ptr(),p_tilde_fx_mapping.device_ptr(),\
			 p_tilde_fy_mapping.device_ptr(),p_tilde_m_mapping.device_ptr(),\
			 partialsum_gpu.gpudata, partialsum_fx_gpu.gpudata,\
			 partialsum_fy_gpu.gpudata, partialsum_m_gpu.gpudata,\
			 np.uint32(nElements))
		cuda_driver.Context.synchronize()

		im_mapping.unmap()
		fx_mapping.unmap()
		fy_mapping.unmap()
		m_mapping.unmap()

		tilde_im_mapping.unmap()
		tilde_fx_mapping.unmap()
		tilde_fy_mapping.unmap()
		tilde_m_mapping.unmap()

		p_tilde_mapping.unmap()
		p_tilde_fx_mapping.unmap()
		p_tilde_fy_mapping.unmap()
		p_tilde_m_mapping.unmap()

		#Read out the answer...
		partialsum = partialsum_gpu.get()
		partialsum_fx = partialsum_fx_gpu.get()
		partialsum_fy = partialsum_fy_gpu.get()
		partialsum_m = partialsum_m_gpu.get()

		#Reshape arrays
		partialsum = np.reshape(partialsum, (nBlocks,self.n))
		partialsum_fx = np.reshape(partialsum_fx, (nBlocks,self.n))
		partialsum_fy = np.reshape(partialsum_fy, (nBlocks,self.n))
		partialsum_m = np.reshape(partialsum_m, (nBlocks,self.n))

		#print partialsum.shape 

		sum_gpu = np.matrix(np.sum(partialsum,0)).T
		sum_fx_gpu = np.matrix(np.sum(partialsum_fx,0)).T
		sum_fy_gpu = np.matrix(np.sum(partialsum_fy,0)).T
		sum_m_gpu = np.matrix(np.sum(partialsum_m,0)).T

		#print sum_m_gpu.shape 

		#print 'GPU', sum_gpu, sum_fx_gpu, sum_fy_gpu 
		#return sum_gpu+sum_fx_gpu+sum_fy_gpu+sum_m_gpu
		jzc = np.bmat([[sum_gpu,sum_fx_gpu,sum_fy_gpu,sum_m_gpu]])
		return (sum_gpu+sum_fx_gpu+sum_fy_gpu+sum_m_gpu, jzc)

	def j(self, state, deltaX, i, j):
		assert self.yp_tilde_pbo is not None
		floatsize = 4
		rgbsize = 4 #32-bit
		bytesize = self.height*self.width

		#Perturb first
		state.X[i,0] += deltaX
		state.refresh()
		state.render()
		state.X[i,0] -= deltaX

		#Load pixel buffers
		self.pycuda_yp_tilde_pbo.unregister()
		self.pycuda_yp_fx_tilde_pbo.unregister()
		self.pycuda_yp_fy_tilde_pbo.unregister()
		self.pycuda_yp_m_tilde_pbo.unregister()
		self._pack_texture_into_PBO(self.yp_tilde_pbo, self.texid, bytesize, GL_UNSIGNED_BYTE)
		self._pack_texture_into_PBO(self.yp_fx_tilde_pbo, self.tex_fx_id, bytesize*floatsize, GL_FLOAT)
		self._pack_texture_into_PBO(self.yp_fy_tilde_pbo, self.tex_fy_id, bytesize*floatsize, GL_FLOAT)
		self._pack_texture_into_PBO(self.yp_m_tilde_pbo, self.tex_m_id, bytesize*rgbsize, GL_UNSIGNED_BYTE, imageformat = GL_RGBA)
		self.pycuda_yp_tilde_pbo = cuda_gl.BufferObject(long(self.yp_tilde_pbo))
		self.pycuda_yp_fx_tilde_pbo = cuda_gl.BufferObject(long(self.yp_fx_tilde_pbo))
		self.pycuda_yp_fy_tilde_pbo = cuda_gl.BufferObject(long(self.yp_fy_tilde_pbo))
		self.pycuda_yp_m_tilde_pbo = cuda_gl.BufferObject(long(self.yp_m_tilde_pbo))

		#Perturb second
		state.X[j,0] += deltaX
		state.refresh()
		state.render()
		state.X[j,0] -= deltaX

		#Load pixel buffers
		self.pycuda_ypp_tilde_pbo.unregister()
		self.pycuda_ypp_fx_tilde_pbo.unregister()
		self.pycuda_ypp_fy_tilde_pbo.unregister()
		self.pycuda_ypp_m_tilde_pbo.unregister()
		self._pack_texture_into_PBO(self.ypp_tilde_pbo, self.texid, bytesize, GL_UNSIGNED_BYTE)
		self._pack_texture_into_PBO(self.ypp_fx_tilde_pbo, self.tex_fx_id, bytesize*floatsize, GL_FLOAT)
		self._pack_texture_into_PBO(self.ypp_fy_tilde_pbo, self.tex_fy_id, bytesize*floatsize, GL_FLOAT)
		self._pack_texture_into_PBO(self.ypp_m_tilde_pbo, self.tex_m_id, bytesize*rgbsize, GL_UNSIGNED_BYTE, imageformat = GL_RGBA)
		self.pycuda_ypp_tilde_pbo = cuda_gl.BufferObject(long(self.ypp_tilde_pbo))
		self.pycuda_ypp_fx_tilde_pbo = cuda_gl.BufferObject(long(self.ypp_fx_tilde_pbo))
		self.pycuda_ypp_fy_tilde_pbo = cuda_gl.BufferObject(long(self.ypp_fy_tilde_pbo))
		self.pycuda_ypp_m_tilde_pbo = cuda_gl.BufferObject(long(self.ypp_m_tilde_pbo))

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

		tilde_im_mapping = self.pycuda_y_tilde_pbo.map()
		tilde_fx_mapping = self.pycuda_y_fx_tilde_pbo.map()
		tilde_fy_mapping = self.pycuda_y_fy_tilde_pbo.map()
		tilde_m_mapping = self.pycuda_y_m_tilde_pbo.map()

		p_tilde_mapping = self.pycuda_yp_tilde_pbo.map()
		p_tilde_fx_mapping = self.pycuda_yp_fx_tilde_pbo.map()
		p_tilde_fy_mapping = self.pycuda_yp_fy_tilde_pbo.map()
		p_tilde_m_mapping = self.pycuda_yp_m_tilde_pbo.map()
		
		pp_tilde_mapping = self.pycuda_ypp_tilde_pbo.map()
		pp_tilde_fx_mapping = self.pycuda_ypp_fx_tilde_pbo.map()
		pp_tilde_fy_mapping = self.pycuda_ypp_fy_tilde_pbo.map()
		pp_tilde_m_mapping = self.pycuda_ypp_m_tilde_pbo.map()

		partialsum = np.zeros((nBlocks,1), dtype=np.float32)
		partialsum_gpu = gpuarray.to_gpu(partialsum)
		partialsum_fx = np.zeros((nBlocks,1), dtype=np.float32)
		partialsum_fx_gpu = gpuarray.to_gpu(partialsum_fx)
		partialsum_fy = np.zeros((nBlocks,1), dtype=np.float32)
		partialsum_fy_gpu = gpuarray.to_gpu(partialsum_fy)
		partialsum_m = np.zeros((nBlocks,1), dtype=np.float32)
		partialsum_m_gpu = gpuarray.to_gpu(partialsum_m)

		#CUDA definition
		#__global__ void j(unsigned char *y_im_t, float *y_fx_t, float *y_fy_t, unsigned char *y_m_t, 
		#					unsigned char *yp_im_t, float *yp_fx_t, float *yp_fy_t, unsigned char *yp_m_t, 
		#					unsigned char *ypp_im_t, float *ypp_fx_t, float *ypp_fy_t, unsigned char *ypp_m_t, 
		#					float *output, float *output_fx, float *output_fy, float *output_m, 
		#					int len) 

		#Make the call...
		self.cuda_j.prepared_call(grid_dimensions, block_dimensions,\
			 tilde_im_mapping.device_ptr(),tilde_fx_mapping.device_ptr(),\
			 tilde_fy_mapping.device_ptr(),tilde_m_mapping.device_ptr(),\
			 p_tilde_mapping.device_ptr(),p_tilde_fx_mapping.device_ptr(),\
			 p_tilde_fy_mapping.device_ptr(),p_tilde_m_mapping.device_ptr(),\
			 pp_tilde_mapping.device_ptr(),pp_tilde_fx_mapping.device_ptr(),\
			 pp_tilde_fy_mapping.device_ptr(),pp_tilde_m_mapping.device_ptr(),\
			 partialsum_gpu.gpudata, partialsum_fx_gpu.gpudata,\
			 partialsum_fy_gpu.gpudata, partialsum_m_gpu.gpudata,\
			 np.uint32(nElements))
		cuda_driver.Context.synchronize()

		tilde_im_mapping.unmap()
		tilde_fx_mapping.unmap()
		tilde_fy_mapping.unmap()
		tilde_m_mapping.unmap()

		p_tilde_mapping.unmap()
		p_tilde_fx_mapping.unmap()
		p_tilde_fy_mapping.unmap()
		p_tilde_m_mapping.unmap()

		pp_tilde_mapping.unmap()
		pp_tilde_fx_mapping.unmap()
		pp_tilde_fy_mapping.unmap()
		pp_tilde_m_mapping.unmap()

		#Read out the answer...
		partialsum = partialsum_gpu.get()
		partialsum_fx = partialsum_fx_gpu.get()
		partialsum_fy = partialsum_fy_gpu.get()
		partialsum_m = partialsum_m_gpu.get()
		sum_gpu = np.sum(partialsum[0:np.ceil(nBlocks/2.)])
		sum_fx_gpu = np.sum(partialsum_fx[0:np.ceil(nBlocks/2.)])
		sum_fy_gpu = np.sum(partialsum_fy[0:np.ceil(nBlocks/2.)])
		sum_m_gpu = np.sum(partialsum_m[0:np.ceil(nBlocks/2.)])
		scale = 1;
		#print 'j (GPU) components'
		#print sum_gpu, sum_fx_gpu/scale/scale, sum_fy_gpu/scale/scale
		return sum_gpu+sum_fx_gpu/scale/scale+sum_fy_gpu/scale/scale+sum_m_gpu

	def j_multi(self, state, deltaX, ee, label):

		assert self.yp_tilde_pbo is not None
		floatsize = 4
		rgbsize = 4 #32-bit
		bytesize = self.height*self.width

		#Perturb first
		#logging.debug('---------- Perturb and render first point')
		state.X[ee[:,0],0] = np.squeeze(state.X[ee[:,0],0] + deltaX)
		state.refresh(label, hess = True)
		state.render()
		state.X[ee[:,0],0] = np.squeeze(state.X[ee[:,0],0] - deltaX)

		#Load pixel buffers
		#logging.debug('---------- map into CUDA accessible memory')
		self.pycuda_yp_tilde_pbo.unregister()
		self.pycuda_yp_fx_tilde_pbo.unregister()
		self.pycuda_yp_fy_tilde_pbo.unregister()
		self.pycuda_yp_m_tilde_pbo.unregister()
		self._pack_texture_into_PBO(self.yp_tilde_pbo, self.texid, bytesize, GL_UNSIGNED_BYTE)
		self._pack_texture_into_PBO(self.yp_fx_tilde_pbo, self.tex_fx_id, bytesize*floatsize, GL_FLOAT)
		self._pack_texture_into_PBO(self.yp_fy_tilde_pbo, self.tex_fy_id, bytesize*floatsize, GL_FLOAT)
		self._pack_texture_into_PBO(self.yp_m_tilde_pbo, self.tex_m_id, bytesize*rgbsize, GL_UNSIGNED_BYTE, imageformat = GL_RGBA)
		self.pycuda_yp_tilde_pbo = cuda_gl.BufferObject(long(self.yp_tilde_pbo))
		self.pycuda_yp_fx_tilde_pbo = cuda_gl.BufferObject(long(self.yp_fx_tilde_pbo))
		self.pycuda_yp_fy_tilde_pbo = cuda_gl.BufferObject(long(self.yp_fy_tilde_pbo))
		self.pycuda_yp_m_tilde_pbo = cuda_gl.BufferObject(long(self.yp_m_tilde_pbo))

		#Perturb second
		#logging.debug('---------- Perturb and render second point')
		state.X[ee[:,1],0] = np.squeeze(state.X[ee[:,1],0] + deltaX)
		state.refresh(label, hess = True)
		state.render()
		state.X[ee[:,1],0] = np.squeeze(state.X[ee[:,1],0] - deltaX)

		#Load pixel buffers
		#logging.debug('---------- 2nd map into CUDA accessible memory. Unregister')
		self.pycuda_ypp_tilde_pbo.unregister()
		self.pycuda_ypp_fx_tilde_pbo.unregister()
		self.pycuda_ypp_fy_tilde_pbo.unregister()
		self.pycuda_ypp_m_tilde_pbo.unregister()
		logging.debug('---------- Pack texture into PBO')
		self._pack_texture_into_PBO(self.ypp_tilde_pbo, self.texid, bytesize, GL_UNSIGNED_BYTE)
		self._pack_texture_into_PBO(self.ypp_fx_tilde_pbo, self.tex_fx_id, bytesize*floatsize, GL_FLOAT)
		self._pack_texture_into_PBO(self.ypp_fy_tilde_pbo, self.tex_fy_id, bytesize*floatsize, GL_FLOAT)
		self._pack_texture_into_PBO(self.ypp_m_tilde_pbo, self.tex_m_id, bytesize*rgbsize, GL_UNSIGNED_BYTE, imageformat = GL_RGBA)
		#logging.debug('---------- Create cuda_gl BufferObjects')
		self.pycuda_ypp_tilde_pbo = cuda_gl.BufferObject(long(self.ypp_tilde_pbo))
		self.pycuda_ypp_fx_tilde_pbo = cuda_gl.BufferObject(long(self.ypp_fx_tilde_pbo))
		self.pycuda_ypp_fy_tilde_pbo = cuda_gl.BufferObject(long(self.ypp_fy_tilde_pbo))
		self.pycuda_ypp_m_tilde_pbo = cuda_gl.BufferObject(long(self.ypp_m_tilde_pbo))

		#Send to CUDA!
		logging.debug('---------- Perform reduction with CUDA')
		return self._process_j_multi()

	def _process_j_multi(self):
		""" Use PyCuda """

		nElements = self.width*self.height
		nBlocks = nElements/BLOCK_SIZE + 1
		#print 'No. elements:', nElements
		#print 'No. blocks:', nBlocks
		grid_dimensions = (nBlocks, 1)
		block_dimensions = (BLOCK_SIZE, 1, 1)

		tilde_im_mapping = self.pycuda_y_tilde_pbo.map()
		tilde_fx_mapping = self.pycuda_y_fx_tilde_pbo.map()
		tilde_fy_mapping = self.pycuda_y_fy_tilde_pbo.map()
		tilde_m_mapping = self.pycuda_y_m_tilde_pbo.map()

		p_tilde_mapping = self.pycuda_yp_tilde_pbo.map()
		p_tilde_fx_mapping = self.pycuda_yp_fx_tilde_pbo.map()
		p_tilde_fy_mapping = self.pycuda_yp_fy_tilde_pbo.map()
		p_tilde_m_mapping = self.pycuda_yp_m_tilde_pbo.map()
		
		pp_tilde_mapping = self.pycuda_ypp_tilde_pbo.map()
		pp_tilde_fx_mapping = self.pycuda_ypp_fx_tilde_pbo.map()
		pp_tilde_fy_mapping = self.pycuda_ypp_fy_tilde_pbo.map()
		pp_tilde_m_mapping = self.pycuda_ypp_m_tilde_pbo.map()

		partialsum = np.zeros((nBlocks*self.len_Q,1), dtype=np.float32)
		partialsum_gpu = gpuarray.to_gpu(partialsum)
		partialsum_fx = np.zeros((nBlocks*self.len_Q,1), dtype=np.float32)
		partialsum_fx_gpu = gpuarray.to_gpu(partialsum_fx)
		partialsum_fy = np.zeros((nBlocks*self.len_Q,1), dtype=np.float32)
		partialsum_fy_gpu = gpuarray.to_gpu(partialsum_fy)
		partialsum_m = np.zeros((nBlocks*self.len_Q,1), dtype=np.float32)
		partialsum_m_gpu = gpuarray.to_gpu(partialsum_m)
		partialsum_nz = np.zeros((nBlocks*self.len_Q,1), dtype=np.float32)
		partialsum_nz_gpu = gpuarray.to_gpu(partialsum_nz)

		#CUDA definition
		#__global__ void histj(unsigned char *y_im_t, float *y_fx_t, float *y_fy_t, unsigned char *y_m_t, 
		#					unsigned char *yp_im_t, float *yp_fx_t, float *yp_fy_t, unsigned char *yp_m_t, 
		#					unsigned char *ypp_im_t, float *ypp_fx_t, float *ypp_fy_t, unsigned char *ypp_m_t, 
		#					float *output, float *output_fx, float *output_fy, float *output_m, 
		#					float *output_nz, int len) 

		#Make the call...
		self.cuda_histj.prepared_call(grid_dimensions, block_dimensions,\
			 tilde_im_mapping.device_ptr(),tilde_fx_mapping.device_ptr(),\
			 tilde_fy_mapping.device_ptr(),tilde_m_mapping.device_ptr(),\
			 p_tilde_mapping.device_ptr(),p_tilde_fx_mapping.device_ptr(),\
			 p_tilde_fy_mapping.device_ptr(),p_tilde_m_mapping.device_ptr(),\
			 pp_tilde_mapping.device_ptr(),pp_tilde_fx_mapping.device_ptr(),\
			 pp_tilde_fy_mapping.device_ptr(),pp_tilde_m_mapping.device_ptr(),\
			 partialsum_gpu.gpudata, partialsum_fx_gpu.gpudata,\
			 partialsum_fy_gpu.gpudata, partialsum_m_gpu.gpudata,\
			 partialsum_nz_gpu.gpudata, np.uint32(nElements))
		cuda_driver.Context.synchronize()

		tilde_im_mapping.unmap()
		tilde_fx_mapping.unmap()
		tilde_fy_mapping.unmap()
		tilde_m_mapping.unmap()

		p_tilde_mapping.unmap()
		p_tilde_fx_mapping.unmap()
		p_tilde_fy_mapping.unmap()
		p_tilde_m_mapping.unmap()

		pp_tilde_mapping.unmap()
		pp_tilde_fx_mapping.unmap()
		pp_tilde_fy_mapping.unmap()
		pp_tilde_m_mapping.unmap()

		#Read out the answer...
		partialsum = partialsum_gpu.get()
		partialsum_fx = partialsum_fx_gpu.get()
		partialsum_fy = partialsum_fy_gpu.get()
		partialsum_m = partialsum_m_gpu.get()
		partialsum_nz = partialsum_nz_gpu.get()

		#Reshape arrays
		partialsum = np.reshape(partialsum, (nBlocks,self.len_Q))
		partialsum_fx = np.reshape(partialsum_fx, (nBlocks,self.len_Q))
		partialsum_fy = np.reshape(partialsum_fy, (nBlocks,self.len_Q))
		partialsum_m = np.reshape(partialsum_m, (nBlocks,self.len_Q))
		partialsum_nz = np.reshape(partialsum_nz, (nBlocks,self.len_Q))

		#print partialsum.shape 

		sum_gpu = np.matrix(np.sum(partialsum,0)).T
		sum_fx_gpu = np.matrix(np.sum(partialsum_fx,0)).T
		sum_fy_gpu = np.matrix(np.sum(partialsum_fy,0)).T
		sum_m_gpu = np.matrix(np.sum(partialsum_m,0)).T

		sum_nz_gpu = np.matrix(np.sum(partialsum_nz,0)).T
		j_nz = sum_nz_gpu > 0

		#print sum_m_gpu.shape 

		#print 'GPU', sum_gpu, sum_fx_gpu, sum_fy_gpu 
		#return sum_gpu+sum_fx_gpu+sum_fy_gpu+sum_m_gpu
		jc = np.bmat([[sum_gpu,sum_fx_gpu,sum_fy_gpu,sum_m_gpu]])
		return (sum_gpu+sum_fx_gpu+sum_fy_gpu+sum_m_gpu, j_nz, jc)