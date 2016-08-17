#################################
#gloo renderer for Kalman filter#
#################################

#lansdell. Feb 11th 2016

import numpy as np 
from vispy import gloo
from vispy import app
from imgproc import findObjectThreshold
import cv2 
from time import gmtime, strftime
from matplotlib import pyplot as plt

from cuda_multi import CUDAGL, CUDAGL_multi

from cvtools import * 

CV_MAT_DEPTH_MASK = 7
CV_CN_SHIFT = 3

VERT_SHADER = """ // simple vertex shader

attribute vec3 a_position;
attribute vec3 a_velocity;
attribute vec2 a_texcoord;
uniform vec4 u_color;
varying vec4 v_color;
varying vec3 v_vel;

void main (void) {
	v_vel = a_velocity;
	v_color = u_color;
	// Pass tex coords
	gl_TexCoord[0] = vec4(a_texcoord.x, a_texcoord.y, 0.0, 0.0);
	// Calculate position
	gl_Position = vec4(a_position.x, a_position.y, a_position.z, 1.0);
}
"""

FRAG_SHADER = """ // simple fragment shader
uniform sampler2D texture1;

void main()
{
	gl_FragColor = texture2D(texture1, gl_TexCoord[0].st);
}
"""

FRAG_SHADER_RED = """ // simple fragment shader
uniform sampler2D texture1;

void main()
{
	vec4 tex1 = texture2D(texture1, gl_TexCoord[0].st);
	gl_FragColor = vec4(tex1.r, 0, 0, tex1.a);
}
"""

FRAG_SHADER_GREEN = """ // simple fragment shader
uniform sampler2D texture1;

void main()
{
	vec4 tex1 = texture2D(texture1, gl_TexCoord[0].st);
	gl_FragColor = vec4(0, tex1.g, 0, tex1.a);
}
"""

FRAG_SHADER_FLOWX = """ // simple fragment shader
varying vec3 v_vel;
varying vec4 v_color;
float c_zero = 0.0;

void main()
{
	gl_FragColor = vec4(c_zero, c_zero, c_zero, 1.0);
	gl_FragColor.r = v_vel.x;

	//vec3 pix;
	//pix.r = v_vel.x;//+5)/10;
	//pix.g = v_vel.y;//+5)/10;
	//pix.b = 0.0;
	//gl_FragColor.rgb = pix;
	//gl_FragColor.a = 1.0;
}
"""

#Need to make this paint the colors of the faces
def frag_shader_mask(F):
	FRAG_SHADER_MASK = """
	#version 330 
	uniform vec3 u_colors[%d];
	out vec4 fragmentColor;
	void main()
	{
		fragmentColor.rgb = u_colors[gl_PrimitiveID]/255;
		fragmentColor.a = 1.0;
	}
	""" % F
	return FRAG_SHADER_MASK

FRAG_SHADER_FLOWY = """ // simple fragment shader
varying vec3 v_vel;
varying vec4 v_color;
const float c_zero = 0.0;

void main()
{
	gl_FragColor = vec4(c_zero, c_zero, c_zero, 1.0);
	gl_FragColor.r = v_vel.y;
}
"""

FRAG_SHADER_FLOW = """ // fragment shader to convert flow to hue/saturation
varying vec3 v_vel;
varying vec4 v_color;

// relative lengths of color transitions:
// these are chosen based on perceptual similarity
// (e.g. one can distinguish more shades between red and yellow
//  than between yellow and green)
const int RY = 15;
const int YG = 6;
const int GC = 4;
const int CB = 11;
const int BM = 13;
const int MR = 6;
const int NCOLS = RY + YG + GC + CB + BM + MR;
vec3 colorWheel[NCOLS];

const float maxrad = 5;
const float PI = 3.14159265;

void main() {
	float fx = v_vel.x;
	float fy = v_vel.y;
	int k = 0;
	for (int i = 0; i < RY; ++i, ++k)
		colorWheel[k] = vec3(255, 255 * i / RY, 0);
	for (int i = 0; i < YG; ++i, ++k)
		colorWheel[k] = vec3(255 - 255 * i / YG, 255, 0);
	for (int i = 0; i < GC; ++i, ++k)
		colorWheel[k] = vec3(0, 255, 255 * i / GC);
	for (int i = 0; i < CB; ++i, ++k)
		colorWheel[k] = vec3(0, 255 - 255 * i / CB, 255);
	for (int i = 0; i < BM; ++i, ++k)
		colorWheel[k] = vec3(255 * i / BM, 0, 255);
	for (int i = 0; i < MR; ++i, ++k)
		colorWheel[k] = vec3(255, 0, 255 - 255 * i / MR);
	float rad = sqrt(fx * fx + fy * fy);
	float a = atan(-fy, -fx) / PI;
	float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
	int k0 = int(fk);
	int k1 = int(mod(k0 + 1, NCOLS));
	float f = fk - k0;
	vec3 pix;
	for (int b = 0; b < 3; b++)
	{
		float col0 = colorWheel[k0][b] / 255.0f;
		float col1 = colorWheel[k1][b] / 255.0f;
		float col = (1 - f) * col0 + f * col1;
		if (rad <= maxrad)
			col = rad * col / maxrad; // increase saturation with radius
		//else
		//	col *= .75; // out of range
		pix[2 - b] = col;
	}

	gl_FragColor.rgb = pix;
	gl_FragColor.a = 1.0;

	/* //Simple shader
	vec3 pix;
	pix.r = (fx+5)/10;
	pix.g = (fy+5)/10;
	pix.b = 0.0;
	gl_FragColor.rgb = pix;
	gl_FragColor.a = 1.0;
	*/
}
"""

def frag_shader_lines(I):
	FRAG_SHADER_LINES = """
	#version 330
	uniform sampler1D texture1;
	float nComp = %d;
	out vec4 fragmentColor;
	void main()
	{
		fragmentColor = texture(texture1, (gl_PrimitiveID/3)/nComp);
	}
	""" % I
	return FRAG_SHADER_LINES

class Renderer(app.Canvas):

	def __init__(self, distmesh, vel, flow, nx, im1, cuda, eps_Z, eps_J, eps_M, labels, labels_hess, Q, showtracking = False, force = None, multi = True):

		self.Q = Q
		self.cuda = cuda
		self.showtracking = showtracking 
		self.force = force
		self.fmin = -.5
		self.fmax = .5
		self.activeface = 0 
		self.state = 'texture'
		self.tri = distmesh.t 
		self.I = len(self.tri)
		self.F = len(self.tri)
		self.n = distmesh.p.shape[0]
		title = 'Hydra tracker. Displaying %s state (space to toggle)' % self.state
		size = (nx, nx)
		app.Canvas.__init__(self, keys='interactive', title = title, show = showtracking, size=size, resizable=False)
		self.indices_buffer, self.outline_buffer, self.vertex_data, self.quad_data, self.quad_buffer, self.l0 = self.loadMesh(distmesh.p, vel, distmesh.t, nx, labels, labels_hess)
		self.l = self.l0.copy()

		self._vbo = gloo.VertexBuffer(self.vertex_data)
		self._quad = gloo.VertexBuffer(self.quad_data)
		self.current_frame = im1
		self.current_texture = gloo.Texture2D(im1)
		self.init_texture = gloo.Texture2D(im1)

		self.current_flowx = flow[:,:,0]
		self.current_flowy = flow[:,:,1]
		self.current_fx_texture = gloo.Texture2D(flow[:,:,0], format="luminance", internalformat="r32f")
		self.current_fy_texture = gloo.Texture2D(flow[:,:,1], format="luminance", internalformat="r32f")

		#Setup programs
		self._program = gloo.Program(VERT_SHADER, FRAG_SHADER)
		self._program['texture1'] = self.init_texture
		self._program.bind(self._vbo)
		#self._program_lines = gloo.Program(VERT_SHADER, FRAG_SHADER_LINES)

		self._program_lines = gloo.Program(VERT_SHADER, frag_shader_lines(self.I))
		#for i in range(self.I):
		#	self._program_lines[u'u_colors[%d]'%i] = tuple(self.linecolors[i,:])
		self._program_lines['texture1'] = self.linecolors.astype(np.uint8)
		self._program_lines.bind(self._vbo)
		self._program_flowx = gloo.Program(VERT_SHADER, FRAG_SHADER_FLOWX)
		self._program_flowx['u_color'] = 1, 0, 0, 1
		self._program_flowx.bind(self._vbo)
		self._program_flowy = gloo.Program(VERT_SHADER, FRAG_SHADER_FLOWY)
		self._program_flowy['u_color'] = 1, 0, 0, 1
		self._program_flowy.bind(self._vbo)
		self._program_flow = gloo.Program(VERT_SHADER, FRAG_SHADER_FLOW)
		self._program_flow['u_color'] = 0, 1, 0, 1
		self._program_flow.bind(self._vbo)
		self._program_red = gloo.Program(VERT_SHADER, FRAG_SHADER_RED)
		self._program_green = gloo.Program(VERT_SHADER, FRAG_SHADER_GREEN)
		self._program_mask = gloo.Program(VERT_SHADER, frag_shader_mask(self.F))
		self._program_mask['u_colors'] = np.squeeze(self.facecolors[:,:,0])
		self._program_mask.bind(self._vbo)

		self._program_outline = gloo.Program(VERT_SHADER, frag_shader_lines(self.I))
		self._program_outline.bind(self._vbo)
		self._setoutlinecolors()

		#Create FBOs, attach the color buffer and depth buffer
		self.shape = (nx, nx)
		self._rendertex1 = gloo.Texture2D((self.shape + (1,)), format="luminance", internalformat="r8")
		self._rendertex2 = gloo.Texture2D((self.shape + (1,)), format="luminance", internalformat="r32f")
		self._rendertex3 = gloo.Texture2D((self.shape + (1,)), format="luminance", internalformat="r32f")
		#No straightforward 1 bit texture support... possibly can do with some tricks, but also 
		#possibly not worth it.
		self._rendertex4 = gloo.Texture2D((self.shape + (4,)), format="rgba", internalformat="rgba")
		self._fbo1 = gloo.FrameBuffer(self._rendertex1, gloo.RenderBuffer(self.shape))
		self._fbo2 = gloo.FrameBuffer(self._rendertex2, gloo.RenderBuffer(self.shape))
		self._fbo3 = gloo.FrameBuffer(self._rendertex3, gloo.RenderBuffer(self.shape))
		self._fbo4 = gloo.FrameBuffer(self._rendertex4, gloo.RenderBuffer(self.shape))

		#import rpdb2 
		#rpdb2.start_embedded_debugger("asdf")
		#gloo.set_viewport(0, 0, nx, nx)
		#tm.kf.state.renderer._backend._vispy_set_size(1000,1000)
		#self._backend._vispy_set_size(*size)
		#gloo.set_viewport(0, 0, *size)
		gloo.set_clear_color('black')
		self._timer = app.Timer('auto', connect=self.update, start=True)
		if showtracking:
			self.show()
		gloo.set_viewport(0, 0, *size)
		self.render()
		self.draw(None)
		#print self._rendertex1.id
		#print self.context.shared._parser._objects
		a=self.context.shared._parser.get_object(self._rendertex1.id)._handle
		b=self.context.shared._parser.get_object(self._rendertex2.id)._handle
		c=self.context.shared._parser.get_object(self._rendertex3.id)._handle
		d=self.context.shared._parser.get_object(self._rendertex4.id)._handle
		if multi:
			self.cudagl = CUDAGL_multi(self._rendertex1, self._rendertex2, self._rendertex3, self._rendertex4, self._fbo1, self._fbo2, self._fbo3, self._fbo4, a, b, c, d, eps_Z, eps_J, eps_M, cuda, self.n, len(self.Q))
		else:
			self.cudagl = CUDAGL(self._rendertex1, self._rendertex2, self._rendertex3, self._rendertex4, self._fbo1, self._fbo2, self._fbo3, self._fbo4, a, b, c, d, eps_Z, eps_J, eps_M, cuda)

		#print self.size
		#self._backend._vispy_set_size(*size)
		#print self.size

	def __del__(self):
		print 'Deleting renderer'
		del self.cudagl 
		self.close()

	def on_resize(self, event):
		width, height = event.physical_size
		#gloo.set_viewport(0, 0, width, height)

	def render(self):
		#Render the current positions to FBO1 
		with self._fbo1:
			gloo.clear()
			self._program.draw('triangles', self.indices_buffer)
		#Render the current velocities to FBO2
		with self._fbo2:
			gloo.clear()
			self._program_flowx.draw('triangles', self.indices_buffer)
		with self._fbo3:
			gloo.clear()
			self._program_flowy.draw('triangles', self.indices_buffer)
		#Render mask 
		with self._fbo4:
			gloo.clear()
			self._program_mask.draw('triangles', self.indices_buffer)

	def on_draw(self, event):
		self.draw(event)

	def _updatemaskpalette(self, colors):
		for i in range(len(colors)):
			self._program_mask['u_colors[%d]'%i] = colors[i,:]

	def _updatelinecolors(self, colors):
		self._program_lines['texture1'] = colors.astype(np.uint8)
		#for i in range(len(colors)):
		#	self._program_lines['u_colors[%d]'%i] = colors[i,:]

	def _updateoutlinecolors(self, colors):
		self._program_outline['texture1'] = colors.astype(np.uint8)
		#for i in range(len(colors)):
		#	self._program_outline['u_colors[%d]'%i] = colors[i,:]

	def draw(self, event):
		#Turn on additive blending
		gloo.set_state('additive')
		gloo.clear()
		#Summary render to main screen
		if self.state == 'texture' or self.state == 'raw':
			self._program.draw('triangles', self.indices_buffer)
		elif self.state == 'flow':
			self._program_flow.bind(self._vbo)
			self._program_flow.draw('triangles', self.indices_buffer)
		elif self.state == 'mask':
			self._program_mask.bind(self._vbo)
			#self._updatemaskpalette(np.squeeze(self.randfacecolors[:,:,0]))
			#self._updatemaskpalette(np.squeeze(self.randhessfacecolors[:,:,1]))
			self._updatemaskpalette(np.squeeze(self.hessfacecolors[:,:,1]))
			#print self._program_mask['u_colors']#self.randfacecolors[:,:,0]
			self._program_mask.draw('triangles', self.indices_buffer)			
		elif self.state == 'overlay':
			self._program_red['texture1'] = self.current_texture
			self._program_red.bind(self._quad)
			self._program_red.draw('triangles', self.quad_buffer)
			self._program_green.bind(self._vbo)
			self._program_green['texture1'] = self.init_texture
			self._program_green.draw('triangles', self.indices_buffer)
		else:
			self._program_outline.bind(self._vbo)
			self._program_outline.draw('lines', self.outline_buffer)
		#Draw wireframe, too
		if self.state != 'raw' and self.state != 'outline':
			self._program_lines.draw('lines', self.outline_buffer)

	def on_key_press(self, event):
		if event.key in [' ']:
			if self.state == 'flow':
				self.state = 'raw'
			elif self.state == 'raw':
				self.state = 'overlay'
			elif self.state == 'overlay':
				self.state = 'texture'
			elif self.state == 'texture':
				self.state = 'mask'
			elif self.state == 'mask':
				self.state = 'outline'
			else:
				self.state = 'flow'
			self.title = 'Hydra tracker. Displaying %s state (space to toggle, q to quit).' % self.state
			if self.state == 'outline':
				self.title += ' Face: %d' % self.activeface 
			self.update()

		#Shift the active face around a bit...
		if event.key in ['a', 's', 'd', 'w']:
			vertices = self.vertices 
			[t1, t2, t3] = self.tri[self.activeface,:]
			if event.key in ['a']:
				vertices[t1,0] -= 2
				vertices[t2,0] -= 2
				vertices[t3,0] -= 2
			if event.key in ['d']:
				vertices[t1,0] += 2
				vertices[t2,0] += 2
				vertices[t3,0] += 2
			if event.key in ['w']:
				vertices[t1,1] -= 2
				vertices[t2,1] -= 2
				vertices[t3,1] -= 2
			if event.key in ['s']:
				vertices[t1,1] += 2
				vertices[t2,1] += 2
				vertices[t3,1] += 2
			self.update_vertex_buffer(vertices, self.velocities, 1)

		if event.key in ['h']:
			self.screenshot()
		if event.key in ['q']:
			self.cudagl._destroy_PBOs()
			self.close()
		if event.key in ['t']:
			self.activeface += 1 
			if self.activeface >= len(self.tri):
				self.activeface = 0 
			self._setoutlinecolors()

	def _setoutlinecolors(self):
		for idx in range(len(self.tri)):
			if idx == self.activeface:
				o = [255, 0, 0, 255]
			else:
				o = [0, 255, 0, 128]
			self.outlinecolors[idx,:] = o
		self._updateoutlinecolors(self.outlinecolors)

	def screenshot(self, saveall = False, basename = 'screenshot'):
		with self._fbo2:
			print 'saving flowx'
			pixels = gloo.read_pixels(out_type = np.float32)[:,:,0]
			fn = './' + basename + '_flowx_' + strftime("%Y-%m-%d_%H:%M:%S", gmtime()) + '.png'
			#print np.max(pixels)
			cv2.imwrite(fn, (255.*(pixels-np.min(pixels))/(np.max(pixels)-np.min(pixels))).astype(int))
		with self._fbo3:
			print 'saving flowy'
			pixels = gloo.read_pixels(out_type = np.float32)[:,:,0]
			fn = './' + basename + '_flowy_' + strftime("%Y-%m-%d_%H:%M:%S", gmtime()) + '.png'
			cv2.imwrite(fn, (255.*(pixels-np.min(pixels))/(np.max(pixels)-np.min(pixels))).astype(int))
		if not saveall:
			pixels = gloo.read_pixels()
			pixels = cv2.cvtColor(pixels, cv2.COLOR_BGRA2RGBA)
			fn = './' + basename + '_' + self.state + '_' + strftime("%Y-%m-%d_%H:%M:%S", gmtime()) + '.png'
			print 'Saving screenshot to ' + fn
			cv2.imwrite(fn, pixels)
			return None
		else:
			oldstate = self.state
			#self._updatemaskpalette(np.squeeze(self.randfacecolors[:,:,0]))
			self._updatemaskpalette(np.squeeze(self.hessfacecolors[:,:,1]))
			#change render mode, rerender, and save
			#for state in ['flow', 'raw', 'overlay', 'texture']:
			for state in ['raw', 'overlay', 'texture', 'mask']:
				print 'saving', state
				self.state = state
				self._updatemaskpalette(np.squeeze(self.hessfacecolors[:,:,1]))
				self.draw(None)
				pixels = gloo.read_pixels()
				pixels = cv2.cvtColor(pixels, cv2.COLOR_BGRA2RGBA)
				if state == 'overlay':
					overlay = pixels
				fn = './' + basename + '_' + state + '_' + strftime("%Y-%m-%d_%H:%M:%S", gmtime()) + '.png'
				#print 'Saving screenshot to ' + fn
				cv2.imwrite(fn, pixels)
			self.state = oldstate
			self.update()
			return overlay

	def getpredimg(self):
		oldstate = self.state
		self.state = 'raw'
		self.draw(None)
		pixels = gloo.read_pixels()
		self.state = oldstate
		return pixels

	def error(self, state, y_im, y_flow, y_m):
		state.refresh()
		state.render()
		with self._fbo1:
			pixels = gloo.read_pixels()
		with self._fbo2:
			fx = gloo.read_pixels(out_type = np.float32)
		with self._fbo3:
			fy = gloo.read_pixels(out_type = np.float32)
		with self._fbo4:
			m = gloo.read_pixels()

		e_im = np.sum(np.multiply(y_im-pixels[:,:,0], y_im-pixels[:,:,0]))
		e_fx = np.sum(np.multiply(y_flow[:,:,0]-fx[:,:,0], y_flow[:,:,0]-fx[:,:,0]))
		e_fy = np.sum(np.multiply(y_flow[:,:,1]+fy[:,:,0], y_flow[:,:,1]+fy[:,:,0]))
		e_m = np.sum(np.multiply(255*y_m-m[:,:,0], 255*y_m-m[:,:,0]))
		return e_im, e_fx, e_fy, e_m, fx, fy

	def update_vertex_buffer(self, vertices, velocities, multi_idx = -1, hess = False):
		self.vertices = vertices 
		self.velocities = velocities 
		verdata = np.zeros((self.nP,3))
		veldata = np.zeros((self.nP,3))
		#rescale
		verdata[:,0:2] = 2*vertices/self.nx-1
		verdata[:,1] = -verdata[:,1]
		#veldata[:,0:2] = 2*velocities/self.nx
		veldata[:,0:2] = velocities
		veldata[:,1] = -veldata[:,1]
		self.vertex_data['a_position'] = verdata
		self.vertex_data['a_velocity'] = veldata 
		self._vbo = gloo.VertexBuffer(self.vertex_data)
		self._program.bind(self._vbo)
		self._program_lines.bind(self._vbo)
		self._program_flowx.bind(self._vbo)
		self._program_flowy.bind(self._vbo)
		self._program_flow.bind(self._vbo)
		self._program_mask.bind(self._vbo)

		#Update color of bars if update based on force available
		for idx, t in enumerate(self.tri):
			if self.force is not None:
				#First edge
				pt = vertices[t[0],:]-vertices[t[1],:]
				self.l[3*idx,0] = np.linalg.norm(pt)
				f = self.force(self.l[3*idx,0], self.l0[3*idx,0])
				c = (f-self.fmin)/(self.fmax-self.fmin)
				c = 255*min(max(c, 0), 1)
				self.linecolors[3*idx,:] = [0, 0, c, 255]
				#Second edge
				pt = vertices[t[1],:]-vertices[t[2],:]
				self.l[3*idx+1,0] = np.linalg.norm(pt)
				f = self.force(self.l[3*idx+1,0], self.l0[3*idx+1,0])
				c = (f-self.fmin)/(self.fmax-self.fmin)
				c = min(max(c, 0), 1)
				self.linecolors[3*idx+1,:] = [0, 0, c, 255]
				#Third edge
				pt = vertices[t[2],:]-vertices[t[0],:]
				self.l[3*idx+2,0] = np.linalg.norm(pt)
				f = self.force(self.l[3*idx+2,0], self.l0[3*idx+2,0])
				c = (f-self.fmin)/(self.fmax-self.fmin)
				c = min(max(c, 0), 1)
				self.linecolors[3*idx+2,:] = [0, 0, c, 255]
	
		#Update uniform data
		self._updatelinecolors(self.linecolors)

		#self._program_mask['u_colors'] = np.squeeze(self.facecolors[:,:,multi_idx])
		if not hess:
			self._updatemaskpalette(np.squeeze(self.facecolors[:,:,multi_idx]))
		else:
			self._updatemaskpalette(np.squeeze(self.hessfacecolors[:,:,multi_idx]))


	#Load mesh data
	def loadMesh(self, vertices, velocities, triangles, nx, labels, labels_hess):
		self.vertices = vertices 
		self.velocities = velocities 
		# Create vetices and texture coords, combined in one array for high performance
		self.nP = vertices.shape[0]
		self.nT = triangles.shape[0]
		self.nx = nx
		vertex_data = np.zeros(self.nP, dtype=[('a_position', np.float32, 3),
			('a_texcoord', np.float32, 2), ('a_velocity', np.float32, 3)])
		verdata = np.zeros((self.nP,3))
		veldata = np.zeros((self.nP,3))
		uvdata = np.zeros((self.nP,2))
		outlinedata = np.zeros((self.nT, 6))
		#rescale
		verdata[:,0:2] = 2*vertices/nx-1
		verdata[:,1] = -verdata[:,1]
		#veldata[:,0:2] = 2*velocities/nx-1
		veldata[:,0:2] = velocities
		veldata[:,1] = -veldata[:,1]
		uvdata = vertices/nx
		vertex_data['a_position'] = verdata
		vertex_data['a_texcoord'] = uvdata 
		vertex_data['a_velocity'] = veldata 
		indices = triangles.reshape((1,-1)).astype(np.uint16)
		indices_buffer = gloo.IndexBuffer(indices)
	
		l0 = np.zeros((3*len(triangles),1))
		for idx, t in enumerate(triangles):
			outlinedata[idx,0] = t[0]
			outlinedata[idx,1] = t[1]
			pt = vertices[t[0],:]-vertices[t[1],:]
			l0[3*idx,0] = np.linalg.norm(pt)

			outlinedata[idx,2] = t[1]
			outlinedata[idx,3] = t[2]
			pt = vertices[t[1],:]-vertices[t[2],:]
			l0[3*idx+1,0] = np.linalg.norm(pt)
			
			outlinedata[idx,4] = t[2]
			outlinedata[idx,5] = t[0]
			pt = vertices[t[2],:]-vertices[t[0],:]
			l0[3*idx+2,0] = np.linalg.norm(pt)

		self.linecolors = np.zeros((3*len(triangles),4))		
		self.linecolors[:,2] = 128
		self.linecolors[:,3] = 255

		self.outlinecolors = np.zeros((len(triangles),4))		

		#Setup multiperturbation face colors
		self.facecolors = 255*np.ones((len(self.tri), 3, labels.shape[1]+1), dtype=np.uint8)
		for i in range(labels.shape[1]):
			for idx,t in enumerate(self.tri):
				self.facecolors[idx, 1, i] = np.floor_divide(labels[idx,i], 256)
				self.facecolors[idx, 2, i] = np.remainder(labels[idx,i], 256)

		#Setup multiperturbation face colors, randomized for clearer visualization
		randcolors = np.random.rand(len(vertices), 3)
		self.randfacecolors = 255*np.ones((len(self.tri), 3, labels.shape[1]+1), dtype=np.uint8)
		for i in range(labels.shape[1]):
			for idx,t in enumerate(self.tri):
				self.randfacecolors[idx, :, i] = 255*randcolors[labels[idx,i],:]

		#Setup multiperturbation face colors
		self.hessfacecolors = 255*np.ones((len(self.tri), 3, labels_hess.shape[1]+1), dtype=np.uint8)
		for i in range(labels_hess.shape[1]):
			for idx,t in enumerate(self.tri):
				self.hessfacecolors[idx, 1, i] = np.floor_divide(labels_hess[idx,i], 256)
				self.hessfacecolors[idx, 2, i] = np.remainder(labels_hess[idx,i], 256)

		#Setup multiperturbation face colors
		randcolors = np.random.rand(np.max(labels_hess)+1, 3)
		self.randhessfacecolors = 255*np.ones((len(self.tri), 3, labels_hess.shape[1]+1), dtype=np.uint8)
		for i in range(labels_hess.shape[1]):
			for idx,t in enumerate(self.tri):
				self.randhessfacecolors[idx, :, i] = 255*randcolors[labels_hess[idx,i],:]


		outline = outlinedata.reshape((1,-1)).astype(np.uint16)
		outline_buffer = gloo.IndexBuffer(outline)

		quad_data = np.zeros(4, dtype=[('a_position', np.float32, 3),
				('a_texcoord', np.float32, 2), ('a_velocity', np.float32, 3)])
		quad_ver = np.array([[-1.0, 1.0, 0.0],  [-1.0, -1.0, 0.0],
							 [ 1.0, 1.0, 0.0], [ 1.0, -1.0, 0.0,] ], np.float32)
		quad_coord = np.array([  [0.0, 0.0], [0.0, 1.0],
								[1.0, 0.0], [1.0, 1.0] ], np.float32)
		quad_vel = np.zeros((4,3))
		quad_data['a_texcoord'] = quad_coord 
		quad_data['a_position'] = quad_ver 
		quad_data['a_velocity'] = quad_vel 
		quad_triangles = np.array([[0, 1, 2], [1, 2, 3]])
		quad_indices = quad_triangles.reshape((1,-1)).astype(np.uint16)
		quad_buffer = gloo.IndexBuffer(quad_indices)
		return indices_buffer, outline_buffer, vertex_data, quad_data, quad_buffer, l0

	def update_frame(self, y_im, y_flow, y_m):
		self.current_frame = y_im 
		self.current_texture = gloo.Texture2D(y_im)
		self.current_flowx = y_flow[:,:,0] 
		self.current_fx_texture = gloo.Texture2D(y_flow[:,:,0], format="luminance", internalformat="r32f")
		self.current_flowy = y_flow[:,:,1] 
		self.current_fy_texture = gloo.Texture2D(y_flow[:,:,1], format="luminance", internalformat="r32f")
		self.current_mask = 255*y_m
		self.current_mask_texture = gloo.Texture2D(255*y_m)

	def get_flow(self):
		self.render()
		with self._fbo2:
			flowx = gloo.read_pixels(out_type = np.float32)
		with self._fbo3:
			flowy = gloo.read_pixels(out_type = np.float32)
		return (flowx, flowy)

	def initjacobian(self, y_im, y_flow, y_m):
		if self.cuda:
			self.cudagl.initjacobian(y_im, y_flow, 255*y_m)
			#self.cudagl.initjacobian_CPU(y_im, y_flow, 255*y_m)
		else:
			self.cudagl.initjacobian_CPU(y_im, y_flow, 255*y_m)

	def jz(self, state):
		#Compare both and see if they're always off, or just sometimes...

		if self.cuda:
			#print 'jz(). Using GPU (CUDA)'
			jz_GPU = self.cudagl.jz(state)
			#jz_CPU = self.cudagl.jz_CPU()
			#print 'GPU:', jz_GPU, 'CPU:', jz_CPU
			return jz_GPU

			#return self.cudagl.jz()
		else:
			#print 'Using CPU'
			return self.cudagl.jz_CPU(state)

	def jz_multi(self, state):
		#Compare both and see if they're always off, or just sometimes...

		if self.cuda:
			#print 'jz(). Using GPU (CUDA)'
			jz_GPU = self.cudagl.jz_multi(state)
			#jz_CPU = self.cudagl.jz_CPU()
			#print 'GPU:', jz_GPU, 'CPU:', jz_CPU
			return jz_GPU

			#return self.cudagl.jz()
		else:
			#print 'Using CPU'
			return self.cudagl.jz_CPU(state)

	def j(self, state, deltaX, i, j):
		if self.cuda:
			#print 'j(). Using GPU (CUDA)'
			j_GPU = self.cudagl.j(state, deltaX, i, j)
			#j_CPU = self.cudagl.j_CPU(state, deltaX, i, j)
			#print 'GPU:', j_GPU, 'CPU:', j_CPU
			return j_GPU

			#return self.cudagl.j(state, deltaX, i, j)
		else:
			return self.cudagl.j_CPU(state, deltaX, i, j)

	def j_multi(self, state, deltaX, ee, labelidx, ee_idx):
		if self.cuda:
			#print 'j(). Using GPU (CUDA)'
			j_GPU = self.cudagl.j_multi(state, deltaX, ee, labelidx)
			#j_CPU = self.cudagl.j_CPU(state, deltaX, ee)
			#print 'GPU:', j_GPU, 'CPU:', j_CPU
			return j_GPU
		else:
			h = np.zeros((len(self.Q), 1))
			h_hist = np.zeros((len(self.Q), 1))
			for idx, eidx in enumerate(ee_idx):
				e = ee[idx]
				h[eidx] = self.cudagl.j_CPU(state, deltaX, e[0], e[1]) 
				h_hist[eidx] = 1
			return (h, h_hist)

class VideoStream:
	def __init__(self, fn, threshold):
		print("Creating video stream from " + fn)
		print("Background subtraction below intensity " + str(threshold))
		self.threshold = threshold
		self.cap = cv2.VideoCapture(fn)
		ret, frame = self.cap.read()
		assert ret, "Cannot open %s" % fn 
		nx,ny = frame.shape[0:2]
		self.nx = nx 
		self.ny = ny
		self.frame = frame
		self.frame_orig = frame.copy()
		self.grayframe = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)

	def read(self, backsub = True):
		try:
			ret, frame = self.cap.read()
			self.frame = frame 
			self.grayframe = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
		except:
			ret = False
			frame = None
			grayframe = None 
			mask = None 
			return ret, frame, grayframe, mask

		if not backsub:
			return ret, self.frame, self.grayframe
		else:
			(mask, ctrs, fd) = findObjectThreshold(self.frame, threshold = self.threshold)
			#Apply mask
			backframe = np.multiply(np.dstack((mask, mask, mask)), self.frame)
			backgrayframe = np.multiply(mask, self.grayframe)		
			return ret, backframe, backgrayframe, mask

	def current_frame(self, backsub = True):
		if backsub:
			(mask, ctrs, fd) = findObjectThreshold(self.frame, threshold = self.threshold)
			frame = np.multiply(np.dstack((mask, mask, mask)), self.frame)
		else:
			frame = self.frame 
		return frame 

	def gray_frame(self, backsub = True):
		if backsub:
			(mask, ctrs, fd) = findObjectThreshold(self.frame, threshold = self.threshold)
			frame = np.multiply(mask, self.grayframe)
		else:
			frame = self.grayframe 
		return frame 

	def backsub(self, im = None):
		(mask, ctrs, fd) = findObjectThreshold(self.frame, threshold = self.threshold)
		if im is None:
			return mask, ctrs, fd
		else:
			if len(im.shape) == 2:
				return np.multiply(mask, im)
			#Probably a better way to do this...
			elif im.shape[2] == 2:
				return np.multiply(np.dstack((mask, mask)), im)
			elif im.shape[2] == 3:
				return np.multiply(np.dstack((mask, mask, mask)), im)

	def isOpened(self):
		return self.cap.isOpened()

class FlowStream:
	def __init__(self, path):
		print("Creating optic flow stream from " + path + "* ...")
		self.path = path 
		self.frame = 0
		if self.isOpened():
			print("Opened successfully.")
		else:
			print("Cannot find flow data at " + path + "*.")

	def read(self):
		flowframe = None 
		fn_x = self.path + ("_%03d"%self.frame) + "_x.mat"
		fn_y = self.path + ("_%03d"%self.frame) + "_y.mat"
		self.frame += 1
		try:
			self.flowx = readFileToMat(fn_x)
			self.flowy = readFileToMat(fn_y)
			nx = self.flowx.shape[0]
			ny = self.flowx.shape[1]
			flowframe = np.zeros((nx, ny, 2), dtype = np.float32)
			flowframe[:,:,0] = self.flowx 
			flowframe[:,:,1] = self.flowy
			ret = True
		except IOError:
			ret = False 
		return ret, flowframe

	def draw(self):
		"""Draw current flow field"""
		#gray_flowx = 255.*(self.flowx-np.min(self.flowx))/(np.max(self.flowx)-np.min(self.flowx))
		#gray_flowy = 255.*(self.flowy-np.min(self.flowy))/(np.max(self.flowy)-np.min(self.flowy))
		plt.imshow(self.flowx, cmap = 'gray', interpolation = 'bicubic')
		plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
		plt.show()
		print "Waiting for user to close plot window"
		plt.imshow(self.flowy, cmap = 'gray', interpolation = 'bicubic')
		plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
		plt.show()
		print "Waiting for user to close plot window"

	def isOpened(self):
		"""Check if file to be read exists"""
		from os.path import isfile
		fn_x = self.path + ("_%03d"%self.frame) + "_x.mat"
		fn_y = self.path + ("_%03d"%self.frame) + "_y.mat"
		if isfile(fn_x) and isfile(fn_y):
			return True 
		else:
			return False 
