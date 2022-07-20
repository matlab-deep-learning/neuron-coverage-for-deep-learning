%% Neuron Coverage for Every Layer
% Train an image regression network and compute the neuron coverage for every 
% layer.

%% 
% Load the training and test data. The data set contains synthetic images of 
% handwritten digits together with the corresponding angles (in degrees) by which 
% each image is rotated. Each digit image is 28-by-28 pixels.

[XTrain,~,YTrain] = digitTrain4DArrayData;
[XTest,~,YTest] = digitTrain4DArrayData;
%% 
% Define the convolutional neural network architecture.

layers = [
    imageInputLayer([28 28 1])
    convolution2dLayer(3,8,Padding="same")
    batchNormalizationLayer
    reluLayer

    averagePooling2dLayer(2,Stride=2)

    convolution2dLayer(3,16,Padding="same")
    batchNormalizationLayer
    reluLayer

    averagePooling2dLayer(2,Stride=2)

    convolution2dLayer(3,32,Padding="same")
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,32,Padding="same")
    batchNormalizationLayer
    reluLayer
   
    dropoutLayer(0.2)
    
    fullyConnectedLayer(1)
    regressionLayer];
%% 
% Specify training options for stochastic gradient descent with momentum. Set 
% the maximum number of epochs to 30 and start the training with an initial learning 
% rate of 0.001.

options = trainingOptions("sgdm", ...
    MaxEpochs=30, ...
    InitialLearnRate=1e-3, ...
    Plots="training-progress", ...
    Verbose=false, ...
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
lgraph = removeLayers(lgraph,"regressionoutput");
net = dlnetwork(lgraph);
%% 
% Convert the test data to a |dlarray| object.

XTest = dlarray(XTest,"SSCB");
%% 
% Compute the neuron coverage for every layer in the network.

nc = neuronCoverage(net,Data=XTest,LayerNames={net.Layers.Name})
%% 
% Plot the results.

bar(categorical(nc.LayerNames),nc.LayerCoverage{:,1})
ylabel("Neuron Coverage")
xlabel("Layer Name")
%% 
% Get the coverage for specific layers.

getCoverageForLayer(nc,["conv_2","conv_3"])
%% 
% _Copyright 2022 The MathWorks, Inc._