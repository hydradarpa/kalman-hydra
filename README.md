# HydraMesh
Python and C++ code for Hydra optical flow, behavior and neural analysis.

lansdell 2016

Dependencies:
* Vispy*
* Numpy
* PyCuda**
* DistMesh 
* OpenCV2
* cvtools (https://github.com/benlansdell/cvtools)
* matplotlib

Notes:
* *Uses OpenGL rendering. If using remotely, you'll need to set up a VirtualGL server
* **If have a CUDA compatible graphics card

## HydraGL

A state space model using an extended Kalman filter to track Hydra in video. 

![alt tag](https://github.com/benlansdell/hydra/blob/master/hydra_wireframe_inverted.png)

How to use:
```
run_kalmanfilter.py <input_video> <optic_flow_path> <output_video> [...]  
run_kalmanfilter.py -h for more information
```
Optic flow must be precomputed and in .mat binary files created using writeMatToFile(). Files must be named:
```
[optic_flow_path]_%03d_x.mat
[optic_flow_path]_%03d_y.mat
```
for the X,Y components of the optic flow, respectively. The easiest way to do this is to run bin/optical_flow_ext, which uses a GPU implementation of Brox optic flow [1].

Example: Run with mesh length 15
```
./run_kalmanfilter.py ./video/johntest_brightcontrast_short.tif ./video/johntest_brightcontrast_short/flow ./video/output.avi -s 15
```

## Algorithm

Implements Kalman filter in which underlying states are  
Similar to [2].
(Write-up coming soon)

# References
[1] Thomas Brox, Andres Bruhn, Nils Papenberg, and Joachim Weickert (2004). "High Accuracy Optical Flow Estimation Based on a Theory for Warping". Proc. 8th European Conference on Computer Vision  
[2] Dellaert, F., Thrun, S., Thorpe, C. (1998). "Jacobian Images of Super-Resolved TextureMaps forModel-BasedMotion Estimation and Tracking". IEEE Workshop on Applications of Computer Vision (WACV'98).
