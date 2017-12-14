%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NAME:	populate_inputs.m
%% PROJECT: 	Survival Curves for Different Cancer Types
%% AUTHOR:	Yugarshi Mondal
%% DESCRIPTION: This script takes CNN outputs which have already been thresholded and creates csvs
%%		that detect lymp infiltration.
%% INPUTS:	
%% OUTPUTS:

%addpath('/data02/shared/yoshi/u24_lymphocyte/lym_outputs')
%% Relative Path to get_patch_til_svs_wrap
addpath('../lym_outputs')

%%%%%%%%%%%%%%%%%% Edit these paths %%%%%%%%%%%%%%%%%%%%%%
processedImages = '/data08/shared/lehhou/active_learning_osprey';
outputs = '/data02/shared/yoshi/u24_lymphocyte/cluster_indices/inputs';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


cancer_types = {'blca','brca','cesc','coad','luad','lusc','paad','prad','read','skcm','stad','ucec','uvm'};

for i = 1:length(cancer_types)
	i
	mkdir(strcat('./inputs/', cancer_types{i}));
	get_patch_til_svs_file_wrap(cancer_types{i},outputs,processedImages);
end
