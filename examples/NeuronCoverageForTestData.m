%% Neuron Coverage for Test Data
% Train an image classification network and compute the neuron coverage using 
% test data.

%% 
% Load the training and test data. The data contains synthetic images of digits 
% from 0 to 9. Each digit image is 28-by-28 pixels.

[XTrain,YTrain] = digitTrain4DArrayData;
[XTest,YTest] = digitTest4DArrayData;
%% 
% Define the convolutional neural network architecture.

layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,Padding="same")
    batchNormalizationLayer
    reluLayer   
    
    maxPooling2dLayer(2,Stride=2)
    
    convolution2dLayer(3,16,Padding="same")
    batchNormalizationLayer
    reluLayer   
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
%% 
% Specify training options for stochastic gradient descent with momentum. Set 
% the maximum number of epochs to 30 and start the training with an initial learning 
% rate of 0.001.

options = trainingOptions("sgdm", ...
    MaxEpochs=30,...
    InitialLearnRate=1e-3, ...
    Verbose=false, ...
    Plots="training-progress", ...
    ExecutionEnvironment="auto");
%% 
% Train the network.

net = trainNetwork(XTrain,YTrain,layers,options);
%% 
% To compute the neuron coverage, you must convert the network to a |dlnetwork| 
% object and the data to a |dlarray| object.
% 
% Remove the output layer and convert the network to a |dlnetwork| object.

lgraph = layerGraph(net);
lgraph = removeLayers(lgraph,"classoutput");
net = dlnetwork(lgraph);
%% 
% Convert the test data to a |dlarray| object.

XTest = dlarray(XTest,"SSCB");
%% 
% Compute the neuron coverage for the trained network and test data. By default, 
% |neuronCoverage| computes the neuron coverage for a subset of the network layers. 
% To specify additional layers, use the 
% |LayerNames|> name-value argument.

nc = neuronCoverage(net,Data=XTest)
%% 
% View the neuron coverage for the default layers.

nc.LayerCoverage
%% 
% Change the activation threshold. 

nc.Threshold = 0.2;
%% 
% The |LayerCoverage| and |AggregateCoverage| properties update to use the new 
% threshold value. View the layer coverage for the updated threshold. 

nc.LayerCoverage
%% 
% To get the coverage for specific layers, use the |getCoverageForLayer| function.

getCoverageForLayer(nc,"relu_1")
%% 
% Add more data to the |neuronCoverage| object. Invert the first test image 
% so that it represents a black digit on a white background. 

XNew = 1 - XTest(:,:,:,1);

figure 
subplot(1,2,1) 
imshow(extractdata(XTest(:,:,:,1)))
subplot(1,2,2) 
imshow(extractdata(XNew))
%% 
% Add the new image to the |neuronCoverage| object and compare the aggregate 
% coverage before and after adding the data.

nc.AggregateCoverage
nc = addData(nc,XNew);
nc.AggregateCoverage
%% 
% Reset the |neuronCoverage| object and remove all the test data information.

nc = resetData(nc)
%% 
% _Copyright 2022 The MathWorks, Inc._
% 
%