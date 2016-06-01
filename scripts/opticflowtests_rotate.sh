#!/bin/sh

VGPY="/opt/VirtualGL/bin/vglrun ipython"

runGenSynthetic()
{
	echo "runGenSynthetic()"
	#$VGPY gen_synthetic.py $WARP $GEOM
}

runGenSyntheticNeurons()
{
	echo "runGenSyntheticNeurons()"
	#$VGPY gen_synthetic_neurons.py $WARP $GEOM
}

analyseFlow()
{
	echo "analyseFlow()"
	echo "Running for $WARP and $GEOM";

	#Brox flow
	ipython test_flow.py $WARP $GEOM

	ipython test_flow.py $WARP $GEOM  _solver_it_20
	ipython test_flow.py $WARP $GEOM  _solver_it_5
	ipython test_flow.py $WARP $GEOM  _inner_it_20
	ipython test_flow.py $WARP $GEOM  _inner_it_5
	ipython test_flow.py $WARP $GEOM  _alpha_0.4
	ipython test_flow.py $WARP $GEOM  _alpha_0.1
	ipython test_flow.py $WARP $GEOM  _gamma_100
	ipython test_flow.py $WARP $GEOM  _gamma_25

	#Deep flow
	ipython test_flow.py $WARP $GEOM _deep

	#Simple flow
	#ipython test_flow.py $WARP $GEOM _simple
}

runFlow()
{
	echo "Running for $WARP and $GEOM";
	#./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/${WARP}_flow_solver_it_20 0.197 50.0 0.8 10 77 20
	#./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/${WARP}_flow_solver_it_5 0.197 50.0 0.8 10 77 5
	#./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/${WARP}_flow_inner_it_20 0.197 50.0 0.8 20 77 10
	#./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/${WARP}_flow_inner_it_5 0.197 50.0 0.8 5 77 10
	#./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/${WARP}_flow_alpha_0.4 0.4 50.0 0.8 10 77 10
	#./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/${WARP}_flow_alpha_0.1 0.1 50.0 0.8 10 77 10
	#./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/${WARP}_flow_gamma_100 0.197 100.0 0.8 10 77 10
	#./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/${WARP}_flow_gamma_25 0.197 25.0 0.8 10 77 10
	
	#Deep flow
	#./bin/deep_optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/${WARP}_flow_deep

#	#Simple flow
#	#./bin/simple_optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/${WARP}_flow_simple	

#	echo "./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/$WARP_flow_solver_it_20 0.197 50.0 0.8 10 77 20"
#	echo "./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/$WARP_flow_solver_it_5 0.197 50.0 0.8 10 77 5"
#	echo "./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/$WARP_flow_inner_it_20 0.197 50.0 0.8 20 77 10"
#	echo "./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/$WARP_flow_inner_it_5 0.197 50.0 0.8 5 77 10"
#	echo "./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/$WARP_flow_alpha_0.4 0.4 50.0 0.8 10 77 10"
#	echo "./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/$WARP_flow_alpha_0.1 0.1 50.0 0.8 10 77 10"
#	echo "./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/$WARP_flow_gamma_100 0.197 100.0 0.8 10 77 10"
#	echo "./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/$WARP_flow_gamma_25 0.197 25.0 0.8 10 77 10"
#	
#	#Deep flow
#	echo "./bin/deep_optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/$WARP_flow_deep"
#
#	#Simple flow
#	#./bin/simple_optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/$WARP_flow_simple	
}

######################################################
#Generate the warped image data and compute the flows#
######################################################

#runFlow 
#analyseFlow 

WARP='rotate'

GEOM='square1'
runGenSynthetic
runFlow 
analyseFlow
GEOM='square2_gradient'
runGenSynthetic
runFlow 
analyseFlow
GEOM='square3_gradient_texture'
runGenSynthetic
runFlow 
analyseFlow
GEOM='square4_gradient_texture_rot1'
runGenSynthetic
runFlow 
analyseFlow
GEOM='square5_gradient_texture_rot2'
runGenSynthetic
runFlow 
analyseFlow
GEOM='hydra_neurons1'
runGenSyntheticNeurons
runFlow 
analyseFlow
GEOM='hydra1'
runGenSynthetic
runFlow 
analyseFlow