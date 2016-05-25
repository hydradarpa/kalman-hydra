#!/bin/sh
#Hydra1 translate_leftup_stretch
./bin/optical_flow_ext ./synthetictests/hydra1/translate_leftup_stretch/translate_leftup_stretch.avi ./synthetictests/hydra1/translate_leftup_stretch/translate_leftup_stretch_flow_solver_it_20 0.197 50.0 0.8 10 77 20
./bin/optical_flow_ext ./synthetictests/hydra1/translate_leftup_stretch/translate_leftup_stretch.avi ./synthetictests/hydra1/translate_leftup_stretch/translate_leftup_stretch_flow_solver_it_5 0.197 50.0 0.8 10 77 5
./bin/optical_flow_ext ./synthetictests/hydra1/translate_leftup_stretch/translate_leftup_stretch.avi ./synthetictests/hydra1/translate_leftup_stretch/translate_leftup_stretch_flow_inner_it_20 0.197 50.0 0.8 20 77 10
./bin/optical_flow_ext ./synthetictests/hydra1/translate_leftup_stretch/translate_leftup_stretch.avi ./synthetictests/hydra1/translate_leftup_stretch/translate_leftup_stretch_flow_inner_it_5 0.197 50.0 0.8 5 77 10
./bin/optical_flow_ext ./synthetictests/hydra1/translate_leftup_stretch/translate_leftup_stretch.avi ./synthetictests/hydra1/translate_leftup_stretch/translate_leftup_stretch_flow_alpha_0.4 0.4 50.0 0.8 10 77 10
./bin/optical_flow_ext ./synthetictests/hydra1/translate_leftup_stretch/translate_leftup_stretch.avi ./synthetictests/hydra1/translate_leftup_stretch/translate_leftup_stretch_flow_alpha_0.1 0.1 50.0 0.8 10 77 10
./bin/optical_flow_ext ./synthetictests/hydra1/translate_leftup_stretch/translate_leftup_stretch.avi ./synthetictests/hydra1/translate_leftup_stretch/translate_leftup_stretch_flow_gamma_100 0.197 100.0 0.8 10 77 10
./bin/optical_flow_ext ./synthetictests/hydra1/translate_leftup_stretch/translate_leftup_stretch.avi ./synthetictests/hydra1/translate_leftup_stretch/translate_leftup_stretch_flow_gamma_25 0.197 25.0 0.8 10 77 10
