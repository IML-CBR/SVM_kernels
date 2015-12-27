%% Reset all
clear all;
close all;
close(findall(0,'Tag','tree viewer'));
clc;
%% Move to working directory
tmp = matlab.desktop.editor.getActive;
cd(fileparts(tmp.Filename));
%% BLOCK 1
% 1)
Dataset = load('../Data/example_dataset_1');
labels = Dataset.labels;
data = (Dataset.data)';
% 2)
sigma = 1;
% Gram matrix
K = exp( -L2_distance(data',data')/(2*sigma^2));
figure; imagesc(K); title('Gram matrix for sigma = 1');
% 3)
% Maximum and minimum values
minvals = (K == min(min(K)));
maxvals = (K == max(max(K)));
figure;
subplot(1,2,1); imagesc(maxvals); title(strcat('Maximum values of K, with value: ',num2str(max(max(K)))));
subplot(1,2,2); imagesc(minvals); title(strcat('Minimum values of K, with value: ',num2str(min(min(K)))));
% Positive definite
positivedefinite = all(eig(K) > 0)

% 4) & 5)
lambda = 1;
sigma = 1;
% RBF SVM model
[model, v] = train_rbfSVM( labels, data, lambda, sigma );
Nsvs=size(model.svs,1)
Nsvm=size(model.margin,1)
Mdist=model.m
% Train Error
K_dense = exp( -L2_distance(data(model.svs,:)',data')/(2*model.sigma^2)); 
y_pred = model.vy(model.svs)' * K_dense;
trainErr=1-mean(sign(y_pred)==labels')
% 6)
name = strcat('rbf SVM soft with lambda ',num2str(lambda),' and sigma ',num2str(sigma));
% RBF plot on training data
plotRbfSVM( data, labels, model, name );

%% BLOCK 2
% 1)
% 2)
Dataset = load('../Data/example_dataset_1');
labels = Dataset.labels;
data = Dataset.data';

% 3)
t1 = classregtree(data, labels);
view(t1);
sfit = eval(t1,data);
ACC = mean(sfit==labels)
ERR = 1-ACC


% 4)
t2 = classregtree(data, labels, 'minparent', 1);
view(t2);
sfit = eval(t2,data);
ACC = mean(sfit==labels)
ERR = 1-ACC

%% BLOCK 3
% 1)
Dataset = load('../Data/example_dataset_1');
labels = Dataset.labels;
data = Dataset.data';
% 2)
k = 10;
kfolds = kfoldIndexer(data,k);
% 3)
freqPos = mean(labels == 1)
freqNeg = mean(labels == -1)
freqPosk = zeros(1,k);
freqNegk = freqPosk;
for i=1:k
    freqPosk(i) = mean(labels(kfolds{i}) == 1);
    freqNegk(i) = mean(labels(kfolds{i}) == -1);
end
freqPosk
freqNegk


%% BLOCK 4
% 1)
Dataset = load('../Data/example_dataset_1');
labels = Dataset.labels;
data = Dataset.data';
% 2)
k = 5;
kfolds = kfoldIndexer(data,k);
% Class Frequencies
freqPos = mean(labels == 1)
freqNeg = mean(labels == -1)
freqPosk = zeros(1,k);
freqNegk = freqPosk;
for i=1:k
    freqPosk(i) = mean(labels(kfolds{i}) == 1);
    freqNegk(i) = mean(labels(kfolds{i}) == -1);
end
freqPosk
freqNegk
%% 3)
lambdas = [0.01,0.1,1,10];
sigmas = [0.1,0.25,0.5,0.75,1,2.5,5,7.5,10];
errParamMat = zeros(size(lambdas,2),size(sigmas,2));
bestValidationsErrSVM = ones(1,k);
for i=1:1:size(lambdas,2)
    for j=1:1:size(sigmas,2) 
        auxACC = zeros(1,k);
        for n=1:1:k
            trainI = cell2mat(kfolds(setdiff((1:1:k),n)));
            testI = kfolds{n};

            trainX = data(trainI,:);
            trainY = labels(trainI);
            testX = data(testI,:);
            testY = labels(testI);

            model = train_rbfSVM( trainY, trainX, lambdas(i), sigmas(j) );
            K_dense = exp( -L2_distance(trainX(model.svs,:)',testX')/(2*model.sigma^2));
            predY = model.vy(model.svs)' * K_dense;
            auxACC(n) = mean(testY==sign(predY)');
        end
        errParamMat(i,j) = 1-mean(auxACC);
        if (1-mean(auxACC)) < mean(bestValidationsErrSVM) 
            bestValidationsErrSVM = 1-auxACC;
        end
    end
end

%% Plot best combination found of lambda and sigma
[bestRow,bestCol] = find(min(min(errParamMat))==errParamMat,1,'first');
model = train_rbfSVM( labels, data, lambdas(bestRow), sigmas(bestCol) );
name = strcat('rbf SVM soft with lambda ',num2str(lambdas(bestRow)),...
    ' and sigma ',num2str(sigmas(bestCol)));
plotRbfSVM( data, labels, model, name );

% Plot the cross-validation error obtained for each pair lambda sigma choosen 
% 3D surface
figure;
surf(sigmas,lambdas,errParamMat);
xlabel('Sigma')
ylabel('Lambda')
zlabel('Error')
% 3D interpolated surface
figure;
surf(sigmas,lambdas,errParamMat);shading interp;
xlabel('Sigma')
ylabel('Lambda')
zlabel('Error')
% 2D colormap
figure;
imagesc(sigmas,lambdas,errParamMat);
xlabel('Sigma')
ylabel('Lambda')
zlabel('Error')
% Error obtained from cross-validation table for lambda as rows and sigma
% as columns
errParamMat

%% 4)

minparents = (1:1:100);
errParamMat = zeros(1,size(minparents,2));
bestValidationsErrTree = ones(1,k);
for i=1:1:size(minparents,2)
    auxACC = zeros(1,k);
    for n=1:1:k
        trainI = cell2mat(kfolds(setdiff((1:1:k),n)));
        testI = kfolds{n};

        trainX = data(trainI,:);
        trainY = labels(trainI);
        testX = data(testI,:);
        testY = labels(testI);

        model = classregtree(trainX, trainY, 'minparent', minparents(i));
        predY = eval(model,testX);
        auxACC(n) = mean(testY==sign(predY));
    end
    errParamMat(i) = 1-mean(auxACC);
    if (1-mean(auxACC)) < mean(bestValidationsErrTree) 
        bestValidationsErrTree = 1-auxACC;
    end
end

% Plot best minparent found with cross-validation
bestMinparent = minparents(find(min(errParamMat)==errParamMat,1,'last'));
model = classregtree(data, labels, 'minparent',bestMinparent);
view(model);
    
% Plot the cross-validation error obtained for each pair minparent choosen
% 2D plot of the cross-validation mean error 
figure;
plot(minparents,errParamMat,'-');
% Error obtained from cross-validation table for lambda as rows and sigma
% as columns
errParamMat


% 5)
bestValidationsErrSVM
bestValidationsErrTree

%% BLOCK 5
% close all
% clear
% clc
data = load('../Data/example_dataset_1.mat');
data = [data.data' data.labels];

[ train_data, test_data ] = splitTrainTest(data,0.8);

k = 5;
kfolds = kfoldIndexer(train_data,k);

%% RBF-SVM parameters
lambdas = [0.01,0.1,1,10];
sigmas = [0.1,0.25,0.5,0.75,1,2.5,5,7.5,10];
errParamMat = zeros(size(lambdas,2),size(sigmas,2));
bestValidationsErrSVM = ones(1,k);
for i=1:size(lambdas,2)
    lambda = lambdas(i);
    for j=1:size(sigmas,2)
        sigma = sigmas(j);
        auxACC = zeros(1,k);
        for n=1:k
            trainI = cell2mat(kfolds(setdiff((1:1:k),n)));
            testI = kfolds{n};

            trainX = train_data(trainI,1:end-1);
            trainY = train_data(trainI,end);
            testX = train_data(testI,1:end-1);
            testY = train_data(testI,end);

            model = train_rbfSVM( trainY, trainX, lambda, sigma );
            K_dense = exp( -L2_distance(trainX(model.svs,:)',testX')/(2*model.sigma^2));
            predY = model.vy(model.svs)' * K_dense;
            auxACC(n) = mean(testY==sign(predY)');
        end
        errParamMat(i,j) = 1-mean(auxACC);
        if (1-mean(auxACC)) < mean(bestValidationsErrSVM) 
            bestValidationsErrSVM = 1-auxACC;
        end
    end
end
%% Find best SVM parameters
[bestRow,bestCol] = find(min(min(errParamMat))==errParamMat,1,'first');
bestLambda = lambdas(bestRow);
bestSigma = sigmas(bestCol);
%% Desicion-Tree parameters
minparents = (1:1:100);
errParamMat = zeros(1,size(minparents,2));
bestValidationsErrTree = ones(1,k);
for i=1:1:size(minparents,2)
    auxACC = zeros(1,k);
    for n=1:1:k
        trainI = cell2mat(kfolds(setdiff((1:1:k),n)));
        testI = kfolds{n};

        trainX = train_data(trainI,1:end-1);
        trainY = train_data(trainI,end);
        testX = train_data(testI,1:end-1);
        testY = train_data(testI,end);

        model = classregtree(trainX, trainY, 'minparent', minparents(i));
        predY = eval(model,testX);
        auxACC(n) = mean(testY==sign(predY));
    end
    errParamMat(i) = 1-mean(auxACC);
    if (1-mean(auxACC)) < mean(bestValidationsErrTree) 
        bestValidationsErrTree = 1-auxACC;
    end
end
%% Find best Tree parameters
bestMinparent = minparents(find(min(errParamMat)==errParamMat,1,'last'));
%% Prepare test and train sets
train_data_X = train_data(:,1:end-1);
train_data_Y = train_data(:,end);
test_data_X = test_data(:,1:end-1);
test_data_Y = test_data(:,end);
%% SVM-Out-Of-Sample-Error
modelSVM = train_rbfSVM( train_data_Y, train_data_X, bestLambda, bestSigma );
K_dense = exp( -L2_distance(train_data_X(modelSVM.svs,:)',test_data_X')/(2*modelSVM.sigma^2));
predictionSVM = modelSVM.vy(modelSVM.svs)' * K_dense;
accuracy_SVM = mean(test_data_Y==sign(predictionSVM)');
out_sampleErrSVM = 1-accuracy_SVM
%% Tree-Out-Of-Sample-Error
modelTree = classregtree(train_data_X, train_data_Y, 'minparent',bestMinparent);
predictionTree = eval(modelTree,test_data_X);
accuracy_Tree = mean(test_data_Y==sign(predictionTree));
out_sampleErrTree = 1-accuracy_Tree
%% Parameters for Validation with all data
data_X = data(:,1:end-1);
data_Y = data(:,end);
%% SVM Validation Error
modelSVM = train_rbfSVM( data_Y, data_X, bestLambda, bestSigma );
K_dense = exp( -L2_distance(data_X(modelSVM.svs,:)',data_X')/(2*modelSVM.sigma^2));
predictionSVM = modelSVM.vy(modelSVM.svs)' * K_dense;
accuracy_SVM = mean(data_Y==sign(predictionSVM)');
ValidationErrSVM = 1-accuracy_SVM
%% Tree Validation Error
modelTree = classregtree(data_X, data_Y, 'minparent',bestMinparent);
predictionTree = eval(modelTree,data_X);
accuracy_Tree = mean(data_Y==sign(predictionTree));
ValidationErrTree = 1-accuracy_Tree


%% BLOCK 6
close all
clear
clc
data = load('../Data/example_dataset_1.mat');
data = [data.data' data.labels];

k = 5;
kfolds = kfoldIndexer(data,k);

k_inner = 5;

lambdas = [0.01,0.1,1,10];
sigmas = [0.1,0.25,0.5,0.75,1,2.5,5];
best_lambdas = zeros(1,k);
best_sigmas = zeros(1,k);
accuracies_rbf_nested = zeros(1,k);

minparents = [1, 5, 10, 15];
best_miniparents = zeros(1,k);
accuracies_trees_nested = zeros(1,k);
for n=1:k
    train_indexes = cell2mat(kfolds(setdiff((1:k_inner),n)));
    train_data = data(train_indexes,1:end);
    test_data = data(kfolds{n},:);
    
    train_data_X = train_data(:,1:end-1);
    train_data_Y = train_data(:,end);
    test_data_X = test_data(:,1:end-1);
    test_data_Y = test_data(:,end);
    
    kfolds_inner = kfoldIndexer(train_data,k_inner);
    
    % RBF-SVM parameters
    errParamMat = zeros(size(lambdas,2),size(sigmas,2));
    bestValidationsErrSVM = ones(1,k);
    for i=1:size(lambdas,2)
        lambda = lambdas(i);
        for j=1:size(sigmas,2)
            sigma = sigmas(j);
            auxACC = zeros(1,k_inner);
            for m=1:k
                trainI = cell2mat(kfolds_inner(setdiff((1:k_inner),m)));
                testI = kfolds_inner{m};

                trainX = train_data(trainI,1:end-1);
                trainY = train_data(trainI,end);
                testX = train_data(testI,1:end-1);
                testY = train_data(testI,end);

                model = train_rbfSVM( trainY, trainX, lambda, sigma );
                K_dense = exp( -L2_distance(trainX(model.svs,:)',testX')/(2*model.sigma^2));
                predY = model.vy(model.svs)' * K_dense;
                auxACC(m) = mean(testY==sign(predY)');
            end
            errParamMat(i,j) = 1-mean(auxACC);
            if (1-mean(auxACC)) < mean(bestValidationsErrSVM)
                bestValidationsErrSVM = 1-auxACC;
                best_lambdas(n) = lambda;
                best_sigmas(n) = sigma;
            end
        end
    end
    model = train_rbfSVM( train_data_Y, train_data_X, best_lambdas(n), best_sigmas(n) );
    K_dense = exp( -L2_distance(train_data_X(model.svs,:)',test_data_X')/(2*model.sigma^2));
    prediction = model.vy(model.svs)' * K_dense;
    accuracies_rbf_nested(n) = mean(test_data_Y==sign(prediction)');
    
    % Decision trees parameters
    errParamMat = zeros(size(lambdas,2),size(sigmas,2));
    bestValidationsErrTree = ones(1,k);
    for i=1:size(minparents,2)
        minparent = minparents(i);
        auxACC = zeros(1,k_inner);
        for m=1:k
            trainI = cell2mat(kfolds_inner(setdiff((1:k_inner),m)));
            testI = kfolds_inner{m};

            trainX = train_data(trainI,1:end-1);
            trainY = train_data(trainI,end);
            testX = train_data(testI,1:end-1);
            testY = train_data(testI,end);

            model = classregtree(trainX, trainY, 'minparent', minparent);
            predY = eval(model,testX);
            auxACC(m) = mean(testY==sign(predY));
        end
        errParamMat(i,j) = 1-mean(auxACC);
        if (1-mean(auxACC)) < mean(bestValidationsErrTree) 
            bestValidationsErrTree = 1-auxACC;
            best_miniparents(n) = minparent;
        end
    end
    model = classregtree(trainX, trainY, 'minparent', best_miniparents(n));
    predY = eval(model,testX);
    accuracies_trees_nested(n) = mean(testY==sign(predY));
end
total_accuracy_rbf_nested = mean(accuracies_rbf_nested);
total_accuracy_trees_nested = mean(accuracies_trees_nested);


%% BLOCK 7

%% BLOCK 8
% close all
% clear
% clc

dataset_names = {'aus','bcw','bid','bre','car','cmc','ech','fac','ger','hec'};

k = 5;
k_inner = 5;

lambdas = [0.01,0.1,1,10];
lambdas = [0.1,1];
% sigmas = [0.1,0.25,0.5,0.75,1];
sigmas = [0.1,0.25,0.5];
best_lambdas = zeros(1,k);
best_sigmas = zeros(1,k);
accuracies_rbf_nested = zeros(1,k);

minparents = [1, 5, 10, 15];
best_miniparents = zeros(1,k);
accuracies_trees_nested = zeros(1,k);

% total_accuracies = zeros(2,size(dataset_names,2));
for d = 1:size(dataset_names,2)
    dataset_name = dataset_names{d};
    data_aux = load(['../datasets/' dataset_name]);
    
    data_x = data_aux.data_aux_v2.Data';
    labels = data_aux.data_aux_v2.labels';
    labels(labels==0)=-1;
    data = [data_x labels];
    kfolds = kfoldIndexer(data,k);
    
    for n=1:k
        train_indexes = cell2mat(kfolds(setdiff((1:k_inner),n)));
        train_data = data(train_indexes,1:end);
        test_data = data(kfolds{n},:);

        train_data_X = train_data(:,1:end-1);
        train_data_Y = train_data(:,end);
        test_data_X = test_data(:,1:end-1);
        test_data_Y = test_data(:,end);

        kfolds_inner = kfoldIndexer(train_data,k_inner);

        % RBF-SVM parameters
        errParamMat = zeros(size(lambdas,2),size(sigmas,2));
        bestValidationsErrSVM = ones(1,k);
        for i=1:size(lambdas,2)
            lambda = lambdas(i);
            for j=1:size(sigmas,2)
                sigma = sigmas(j);
                auxACC = zeros(1,k_inner);
                for m=1:k_inner
                    trainI = cell2mat(kfolds_inner(setdiff((1:k_inner),m)));
                    testI = kfolds_inner{m};

                    trainX = train_data_X(trainI,:);
                    trainY = train_data_Y(trainI);
                    testX = train_data_X(testI,:);
                    testY = train_data_Y(testI);
                    
                    fprintf('Dataset: %d\n',d);
                    model = train_rbfSVM( trainY, trainX, lambda, sigma );
                    K_dense = exp( -L2_distance(trainX(model.svs,:)',testX')/(2*model.sigma^2));
                    predY = model.vy(model.svs)' * K_dense;
                    auxACC(m) = mean(testY==sign(predY)');
                end
                errParamMat(i,j) = 1-mean(auxACC);
                if (1-mean(auxACC)) < mean(bestValidationsErrSVM)
                    bestValidationsErrSVM = 1-auxACC;
                    best_lambdas(n) = lambda;
                    best_sigmas(n) = sigma;
                end
            end
        end
        model = train_rbfSVM( train_data_Y, train_data_X, best_lambdas(n), best_sigmas(n) );
        K_dense = exp( -L2_distance(train_data_X(model.svs,:)',test_data_X')/(2*model.sigma^2));
        prediction = model.vy(model.svs)' * K_dense;
        accuracies_rbf_nested(n) = mean(test_data_Y==sign(prediction)');
    
        % Decision trees parameters
        errParamMat = zeros(size(lambdas,2),size(sigmas,2));
        bestValidationsErrTree = ones(1,k);
        for i=1:size(minparents,2)
            minparent = minparents(i);
            auxACC = zeros(1,k_inner);
            for m=1:k
                trainI = cell2mat(kfolds_inner(setdiff((1:k_inner),m)));
                testI = kfolds_inner{m};

                trainX = train_data(trainI,1:end-1);
                trainY = train_data(trainI,end);
                testX = train_data(testI,1:end-1);
                testY = train_data(testI,end);

                model = classregtree(trainX, trainY, 'minparent', minparent);
                predY = eval(model,testX);
                auxACC(m) = mean(testY==sign(predY));
            end
            errParamMat(i,j) = 1-mean(auxACC);
            if (1-mean(auxACC)) < mean(bestValidationsErrTree) 
                bestValidationsErrTree = 1-auxACC;
                best_miniparents(n) = minparent;
            end
        end
        model = classregtree(trainX, trainY, 'minparent', best_miniparents(n));
        predY = eval(model,testX);
        accuracies_trees_nested(n) = mean(testY==sign(predY));
    end
    total_accuracies(1,d) = mean(accuracies_rbf_nested);
    total_accuracies(2,d) = mean(accuracies_trees_nested);
end
aux = total_accuracies;
save('../Data/total_accuracies','total_accuracies');
load('../Data/total_accuracies')
