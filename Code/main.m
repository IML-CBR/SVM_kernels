%% Reset all
clear all;
close all;
clc;
%% Move to working directory
tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));
%% BLOCK 1 - Juli
% 1)
Dataset = load('../Data/example_dataset_1');
labels = Dataset.labels;
data = Dataset.data;
% 2)
sigma = 1;
K = exp( -L2_distance(data,data)/(2*sigma^2));

% 3)
% 4)

%% BLOCK 2 - Juli

%% BLOCK 3 - Juli

%% BLOCK 4 - Juli

%% BLOCK 5 - Xavi

%% BLOCK 6 - Xavi

%% BLOCK 7 - Xavi

%% BLOCK 8 - Xavi



