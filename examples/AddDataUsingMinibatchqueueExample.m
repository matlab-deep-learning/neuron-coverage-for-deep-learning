%% Add Data Using |minibatchqueue|
% Add data to a |neuronCoverage| object in mini-batches using |minibatchqueue|.

%% 
% Load the training and test data. The data contains 10,000 synthetic images 
% of digits from 0 to 9. Each digit image is 28-by-28 pixels. Load the data as 
% an |ImageDatastore| object.

dataFolder = fullfile(toolboxdir("nnet"),"nndemos","nndatasets","DigitDataset");
imds = imageDatastore(dataFolder, ...
    IncludeSubfolders=true, ...
    LabelSource="foldernames");
%% 
% Divide the datastore so that the training set has 70% of the images for each 
% class and the testing set has the remaining images.

[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,0.3,"randomize");
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

net = trainNetwork(imds,layers,options);
%% 
% To compute the neuron coverage, you must convert the network to a |dlnetwork| 
% object and the data to a |dlarray| object.
% 
% Remove the output layer and convert the network to a |dlnetwork| object.

lgraph = layerGraph(net);
lgraph = removeLayers(lgraph,"classoutput");
net = dlnetwork(lgraph);
%% 
% Create a |neuronCoverage| object without data and with a threshold value of 
% 0.6. Specify the |LayerNames| name-value argument to compute the neuron coverage 
% for the two convolutional layers, the fully connected layer, and the softmax 
% layer.

nc = neuronCoverage(net,0.6,LayerNames=["conv_1","conv_2","fc","softmax"]);
%% 
% Create a |minibatchqueue| object containing the test data. Set the |MiniBatchSize| 
% property to |30|. The test data contains 3000 images that are evenly split across 
% 10 classes.  Because the data is not shuffled, batches 1–10 contain images from 
% class 0, batches 11–20 will contain images from class 1, and so on.

minibatchsize = 30;
mbq = minibatchqueue(imdsTest, ...
    MiniBatchSize=minibatchsize, ...
    MiniBatchFormat="SSBC");
%% 
% Add each mini-batch to the |neuronCoverage| object and compute the coverage.

i = 0;
while hasdata(mbq)
    i = i+1;

    dataBatch = next(mbq);
    nc = addData(nc,dataBatch);

    coverage(i,:) = nc.LayerCoverage{:,1}';
end
%% 
% Plot the results.

plot(coverage,"-")
legend(nc.LayerNames)
xlabel("Batch")
ylabel("Neuron Coverage")
%% 
% The neuron coverage increases as you add more test data.
% 
% _Copyright 2022 The MathWorks, Inc._
% 
%