# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test rendering mesh on top of video
"""
import numpy as np
import cv2
import time 

from kalman import *
from imgproc import * 
from distmesh_dyn import *

from vispy import gloo
from vispy import app

threshold = 9
fn = "./video/GCaMP_local_prop.avi"

cap = cv2.VideoCapture(fn)
ret, frame = cap.read()
frame_orig = frame.copy()
(mask, ctrs, fd) = findObjectThreshold(frame, threshold = threshold)
nx,ny = frame.shape[0:2]

distmesh = DistMesh(frame)
distmesh.createMesh(ctrs, fd, frame, True)

# Create a texture
im1 = frame 

#Load mesh data
def loadMesh(vertices, triangles, nx):
	# Create vetices and texture coords, combined in one array for high performance
	nP = vertices.shape[0]
	nT = triangles.shape[0]
	vertex_data = np.zeros(nP, dtype=[('a_position', np.float32, 3),
		('a_texcoord', np.float32, 2)])
	verdata = np.zeros((nP,3))
	uvdata = np.zeros((nP,2))
	outlinedata = np.zeros((nT, 6))
	#rescale
	verdata[:,0:2] = 2*vertices/nx-1
	verdata[:,1] = -verdata[:,1]
	uvdata = vertices/nx
	vertex_data['a_position'] = verdata
	vertex_data['a_texcoord'] = uvdata 
	indices = triangles.reshape((1,-1)).astype(np.uint16)
	indices_buffer = gloo.IndexBuffer(indices)

	for idx, t in enumerate(triangles):
		outlinedata[idx,0] = t[0]
		outlinedata[idx,1] = t[1]
		outlinedata[idx,2] = t[1]
		outlinedata[idx,3] = t[2]
		outlinedata[idx,4] = t[2]
		outlinedata[idx,5] = t[0]
	outline = outlinedata.reshape((1,-1)).astype(np.uint16)
	outline_buffer = gloo.IndexBuffer(outline)
	return indices_buffer, outline_buffer, vertex_data

indices_buffer, outline_buffer, vertex_data = loadMesh(distmesh.p, distmesh.t, nx)

VERT_SHADER = """ // simple vertex shader

attribute vec3 a_position;
attribute vec2 a_texcoord;
uniform float sizeFactor;
uniform vec4 u_color;
varying vec4 v_color;

void main (void) {
	v_color = u_color;
	// Pass tex coords
	gl_TexCoord[0] = vec4(a_texcoord.x, a_texcoord.y, 0.0, 0.0);
	// Calculate position
	gl_Position = sizeFactor*vec4(a_position.x, a_position.y, a_position.z,
														1.0/sizeFactor);
}
"""

FRAG_SHADER = """ // simple fragment shader
uniform sampler2D texture1;

void main()
{
	gl_FragColor = texture2D(texture1, gl_TexCoord[0].st);
}
"""

FRAG_SHADER_LINES = """ // simple fragment shader
varying vec4 v_color;

void main()
{
	gl_FragColor = v_color;
}
"""

class Canvas(app.Canvas):

	def __init__(self):
		app.Canvas.__init__(self, keys='interactive', title = 'Hydra tracker')
		self._program = gloo.Program(VERT_SHADER, FRAG_SHADER)
		self._vbo = gloo.VertexBuffer(vertex_data)
		self._program['texture1'] = gloo.Texture2D(im1)
		self._program.bind(self._vbo)

		self._program_lines = gloo.Program(VERT_SHADER, FRAG_SHADER_LINES)
		self._program_lines.bind(self._vbo)

		gloo.set_clear_color('black')
		self._timer = app.Timer('auto', connect=self.update, start=True)
		self.show()

	def on_resize(self, event):
		width, height = event.physical_size
		gloo.set_viewport(0, 0, width, height)

	def on_draw(self, event):
		gloo.clear()
		self._program['sizeFactor'] = 1# + np.sin(time.time() * 3) * 0.2
		self._program.draw('triangles', indices_buffer)
		#Draw wireframe, too
		self._program_lines['sizeFactor'] = 1# + np.sin(time.time() * 3) * 0.2
		self._program_lines['u_color'] = 1, 1, 0, 1
		self._program_lines.draw('lines', outline_buffer)

if __name__ == '__main__':
	c = Canvas()
	app.run()