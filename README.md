# HydraMesh
Python and C++ code for Hydra optical flow, behavior and neural analysis

Dependencies:
-Vispy*
-Numpy
-PyCuda**
-DistMesh 
-HDF
-OpenCV2
-cvtools (https://github.com/benlansdell/cvtools)

Notes:
*  Uses OpenGL rendering. If using remotely, you'll need to set up a VirtualGL server
** If have a CUDA compatible graphics card

# HydraGL. 

State space model using an extended Kalman filter to track Hydra in video

Example: 
./run_kalmanfilter.py ./video/johntest_brightcontrast_short.tif -s 15

## Algorithm

Implements Kalman filter 
