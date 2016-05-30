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

runFlow()
{
	./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/$WARP_flow_solver_it_20 0.197 50.0 0.8 10 77 20
	./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/$WARP_flow_solver_it_5 0.197 50.0 0.8 10 77 5
	./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/$WARP_flow_inner_it_20 0.197 50.0 0.8 20 77 10
	./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/$WARP_flow_inner_it_5 0.197 50.0 0.8 5 77 10
	./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/$WARP_flow_alpha_0.4 0.4 50.0 0.8 10 77 10
	./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/$WARP_flow_alpha_0.1 0.1 50.0 0.8 10 77 10
	./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/$WARP_flow_gamma_100 0.197 100.0 0.8 10 77 10
	./bin/optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/$WARP_flow_gamma_25 0.197 25.0 0.8 10 77 10
	
	#Deep flow
	./bin/deep_optical_flow_ext ./synthetictests/$GEOM/$WARP/$WARP.avi ./synthetictests/$GEOM/$WARP/$WARP_flow_deep

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

WARP='translate_leftup'

GEOM='square1'
runGenSynthetic
runFlow
GEOM='square2_gradient'
runGenSynthetic
runFlow
GEOM='square3_gradient_texture'
runGenSynthetic
runFlow
GEOM='square4_gradient_texture_rot1'
runGenSynthetic
runFlow
GEOM='square5_gradient_texture_rot2'
runGenSynthetic
runFlow
GEOM='hydra_neurons1'
runGenSyntheticNeurons
runFlow

WARP='translate_leftup_stretch'

GEOM='square1'
runGenSynthetic
runFlow
GEOM='square2_gradient'
runGenSynthetic
runFlow
GEOM='square3_gradient_texture'
runGenSynthetic
runFlow
GEOM='square4_gradient_texture_rot1'
runGenSynthetic
runFlow
GEOM='square5_gradient_texture_rot2'
runGenSynthetic
runFlow
GEOM='hydra_neurons1'
runGenSyntheticNeurons
runFlow

WARP='warp'

GEOM='square1'
runGenSynthetic
runFlow
GEOM='square2_gradient'
runGenSynthetic
runFlow
GEOM='square3_gradient_texture'
runGenSynthetic
runFlow
GEOM='square4_gradient_texture_rot1'
runGenSynthetic
runFlow
GEOM='square5_gradient_texture_rot2'
runGenSynthetic
runFlow
GEOM='hydra_neurons1'
runGenSyntheticNeurons
runFlow

WARP='rotate'

GEOM='square1'
runGenSynthetic
runFlow
GEOM='square2_gradient'
runGenSynthetic
runFlow
GEOM='square3_gradient_texture'
runGenSynthetic
runFlow
GEOM='square4_gradient_texture_rot1'
runGenSynthetic
runFlow
GEOM='square5_gradient_texture_rot2'
runGenSynthetic
runFlow
GEOM='hydra_neurons1'
runGenSyntheticNeurons
runFlow

###################
#Analyze the flows#
###################
