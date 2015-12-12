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
data = (Dataset.data)';
% 2)
sigma = 1;
K = exp( -L2_distance(data',data')/(2*sigma^2));
figure; imagesc(K); title('Gram matrix for sigma = 1');
% 3)
minvals = (K == min(min(K)));
maxvals = (K == max(max(K)));
figure;
subplot(1,2,1); imagesc(maxvals); title(strcat('Maximum values of K, with value: ',num2str(max(max(K)))));
subplot(1,2,2); imagesc(minvals); title(strcat('Minimum values of K, with value: ',num2str(min(min(K)))));
positivedefinite = all(eig(K) > 0);

% 4) & 5)
lambda = 1;
sigma = 1;
[model, v] = train_rbfSVM( labels, data, lambda, sigma );

% 6)
name = strcat('linear SVM soft with lambda ',num2str(lambda),' and sigma ',num2str(sigma));
plotRbfSVM2( data, labels, model, name );


%% BLOCK 2 - Juli


%% BLOCK 3 - Juli

%% BLOCK 4 - Juli

%% BLOCK 5 - Xavi

%% BLOCK 6 - Xavi

%% BLOCK 7 - Xavi

%% BLOCK 8 - Xavi



