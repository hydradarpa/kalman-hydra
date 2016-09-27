path_in = './video/20160412/stk_0001';
path_res = './video/20160412/stk_0001_mfsf_nref100'; 

% Run MFSF (same command as in case 1, but we have deleted the
% specification of MaxPL,W), with default parameters that will create good
% results (limiting the coarse-to-fine strategy to process pyramid levels with at most
% MaxPIXpyr=20000 pixels):
[u,v,parmsOF,info] = runMFSF('path_in',path_in,'frname_frmt','frame_%03d.tif', 'nref', 100, 'sframe', 1, ...
 'nframe', 200, 'MaxPIXpyr', 20000);

% Save the result:
mkdir(path_res);
save(fullfile(path_res,'result.mat'), 'u', 'v', 'parmsOF','info');

% Visualise the result (note that the current visualisation code is
% slow, mainly due to grid visualisation, so for faster runtimes
% set type_vis='colcode,warp' or avoid visualisations completely):
path_figs = fullfile(path_res,'figures');
% use an image with a mask for reference frame in order to
% visualise the grid only for the foreground:
tic; visualizeMFSF(path_in,u,v,parmsOF, 'path_figs',path_figs, 'Nrows_grid', 60,...
    'file_mask_grid','in_test/mask_f89.png'); 
fprintf('\nRuntime of MFSF algorithm: %g sec (%g sec per frame)\nRuntime of visualisation code: %g sec\n', ...
    info.runtime,info.runtime/parmsOF.nframe,toc);
