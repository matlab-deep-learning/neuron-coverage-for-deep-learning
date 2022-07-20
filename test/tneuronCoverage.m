classdef tneuronCoverage < matlab.unittest.TestCase
    % tneuronCoverage   Unit tests for neuronCoverage class
    %   Copyright 2022 The MathWorks, Inc.

    properties (TestParameter)
        invalidThreshold = iInvalidThreshold();
    end

    methods(Test)
        %% Positive tests
        function canConstructNeuronCoverageClass(test)
            % Verify the class construction works and exists for dlnetworks
            DLNet = iMakeDLNetwork();

            actCov = neuronCoverage(DLNet);
            test.verifyWarningFree(@() neuronCoverage(DLNet));
            test.verifyClass(actCov,'neuronCoverage');
            test.verifyEmpty(actCov.LayerCoverage);
            test.verifyEmpty(actCov.AggregateCoverage);
            test.verifyEqual(actCov.Threshold, 0.75);
            
            actCov = neuronCoverage(DLNet, 0.75);
            test.verifyClass(actCov,'neuronCoverage');

            actCov = neuronCoverage(DLNet, 0.75, 'LayerNames', {DLNet.Layers(3).Name, DLNet.Layers(4).Name});
            test.verifyClass(actCov,'neuronCoverage');

            actCov = neuronCoverage(DLNet, 0.75, 'Data', dlarray(ones(28,28,1,10),'SSCB'));
            test.verifyClass(actCov,'neuronCoverage');
        end

        function canComputeNeuronCoverageRNN(test)
            net = iMakeSeqToSeqRNN();
            nc = neuronCoverage(net);

            % Add test data.
            [XTest, ~] = japaneseVowelsTestData;
            Xnc = padsequences(XTest, 2, Direction="right");
            Xnc = dlarray(Xnc, "CTB");
            test.verifyNotEmpty(@() nc.addData(Xnc));
        end

        function canComputeCoverageMultiInputNetwork(test)
            % Verify the multi input network succeeds
            net = iMakeMultiInputNet();

            X1 = dlarray( ones([28 28 1 7]), "SSCB" );
            X2 = dlarray( 0.5*ones(7), "CB" );

            nc = neuronCoverage(net,0.9);
            nc = nc.addData({X1,X2});

            test.verifyNotEmpty(nc.LayerCoverage);
            test.verifyNotEmpty(nc.AggregateCoverage);

            ref = nc.getCoverageForLayer("relu");

            nc.Threshold = 0.1;

            % Check the coverage has changed if the threshold has changed
            test.verifyNotEqual(ref, nc.getCoverageForLayer("relu"));

            % Add a test for the constructor to work with data supplied
            nc = neuronCoverage(net,0.9,'LayerNames',"relu",'Data',{X1,X2});
            test.verifyClass(nc,'neuronCoverage');
            test.verifyNotEmpty(nc.LayerCoverage);
            test.verifyNotEmpty(nc.AggregateCoverage);
        end

        function verifyGetCoverageForLayerCharArray(test)
            net = iMakeMultiInputNet();

            X1 = dlarray( ones([28 28 1 7]), "SSCB" );
            X2 = dlarray( 0.5*ones(7), "CB" );

            nc = neuronCoverage(net,0.5);
            nc = nc.addData({X1,X2});
            
            test.verifyEqual(nc.getCoverageForLayer({'relu','layer'}), nc.getCoverageForLayer(["relu","layer"]));
        end

        function removedDuplicateLayerNames(test)
            net = iMakeDLNetwork();

            nc = neuronCoverage(net,0.5,'LayerNames',{net.Layers([2 2]).Name});
            nc = nc.addData(dlarray(ones(28,28,1,10),'SSCB'));
            
            test.verifyEqual(numel(nc.LayerCoverage),1);
        end

        function returnEmptyGetCoverageForLayers(test)
            net = iMakeSeriesNetwork();

            lgraph = layerGraph(net);
            lgraph = removeLayers(lgraph,net.Layers(end).Name);
            net = dlnetwork(lgraph);

            nc = neuronCoverage(net);

            test.verifyEmpty(nc.getCoverageForLayer("relu"));
        end

        function returnsUniqueLayerNames(test)
            dlnet = iMakeDLNetwork();
            nc = neuronCoverage(dlnet,'LayerNames',{'relu','relu'});

            test.verifyEqual(numel(nc.LayerNames), 1);
        end

        function checksTableOrderMatchesLayerNamesOrder(test)
            dlnet = iMakeMultiInputNet();
            nc = neuronCoverage(dlnet,'LayerNames',{'gap','relu','flatten'});

            X1 = dlarray( ones([28 28 1 7]), "SSCB" );
            X2 = dlarray( 0.5*ones(7), "CB" );

            nc = nc.addData({X1,X2});

            test.verifyEqual(transpose(string(nc.LayerCoverage.Row)), nc.LayerNames);
        end

        %% Negative Tests
        function cannotConvertSeriesNetwork(test)
            % Verify the class construction fails for series networks
            seriesNet = iMakeSeriesNetwork();
            test.verifyError(@() neuronCoverage(seriesNet), 'MATLAB:validators:mustBeA');
        end

        function cannotConvertDAGNetwork(test)
            % Verify the class construction fails for DAGnetworks
            DAGNet = squeezenet();
            test.verifyError(@() neuronCoverage(DAGNet), 'MATLAB:validators:mustBeA');
        end

        function errorsWithInvalidThreshold(test,invalidThreshold)
            % Verify the class construction fails for dlnetworks with bad
            % parameters
            DLNet = iMakeDLNetwork();
            test.verifyError(@() neuronCoverage(DLNet, invalidThreshold{1}), invalidThreshold{2});
        end

        function errorsWithInvalidNames(test)
            % Verify the class construction fails for dlnetworks with bad
            % parameters
            DLNet = iMakeDLNetwork();
            
            test.verifyError(@() neuronCoverage(DLNet, 'Layernames', 'badName'), 'neuroncoverage:internal:coverage:neuronCoverage:LayerNotFoundInNetwork');
        end

        function cannotGetCoverageForLayersBadName(test)
            net = iMakeSeriesNetwork();

            lgraph = layerGraph(net);
            lgraph = removeLayers(lgraph,net.Layers(end).Name);
            net = dlnetwork(lgraph);

            nc = neuronCoverage(net);

            test.verifyError(@() nc.getCoverageForLayer("badName"), 'neuroncoverage:internal:coverage:neuronCoverage:NeuronCoverageNotComputedForThisLayer')

            nc = nc.addData(dlarray(randn(28,28,1,3),'SSCB'));

            test.verifyError(@() nc.getCoverageForLayer(["relu","badName"]), 'neuroncoverage:internal:coverage:neuronCoverage:NeuronCoverageNotComputedForThisLayer');
        end

        function cannotGetCoverageForLayersBadIndex(test)
            net = iMakeSeriesNetwork();

            lgraph = layerGraph(net);
            lgraph = removeLayers(lgraph,net.Layers(end).Name);
            net = dlnetwork(lgraph);

            nc = neuronCoverage(net,'Data',dlarray(randn(28,28,1,3),'SSCB'));

            test.verifyError(@() nc.getCoverageForLayer([1 4]), 'MATLAB:validators:mustBeText');
        end

        function errorsWithDefaultLayersMissing(test)
            net = iMakeNoActivationNet();
            test.verifyError(@() neuronCoverage(net),'neuroncoverage:internal:coverage:neuronCoverage:DefaultActivationLayersNotFoundInNetwork');
        end

        function cannotConvertLayersToDlnetwork(test)
            test.verifyError(@() neuronCoverage([featureInputLayer(1),fullyConnectedLayer(10)]),'MATLAB:validators:mustBeA');
        end

        function errorsForWrongNumberInputsInAddData(test)
            % Verify the multi input network succeeds
            net = iMakeMultiInputNet();

            X1 = dlarray( ones([28 28 1 7]), "SSCB" );
            X2 = dlarray( 0.5*ones(7), "CB" );

            nc = neuronCoverage(net,0.9);
            test.verifyError(@() nc.addData({X1,X2,X2}), 'nnet_cnn:dlnetwork:WrongNumInputs');
        end

        function errorsForIncompatibleBatchSizeForMultiInputNetwork(test)
            % Verify the multi input network succeeds
            net = iMakeMultiInputNet();

            X1 = dlarray( ones([28 28 1 3]), "SSCB" );
            X2 = dlarray( 0.5*ones(7), "CB" );

            % Add a test that fails if incompatible batch sizes for add
            % layers
            nc = neuronCoverage(net,0.9,'LayerNames',"fc");
            test.verifyError(@() nc.addData({X1,X2}), 'MATLAB:assertion:failed');
        end

        function errorsForMultiOutputLayers(test)
            multiOutputNet = iMakeMultiOutputNetwork();
            nc = neuronCoverage(multiOutputNet,'LayerNames',{'relu','maxpool','relu'});

            test.verifyError(@() nc.addData(dlarray(randn(5,4,3,2),"SSCB")), 'neuroncoverage:internal:coverage:neuronCoverage:MultiOutputLayersNotSupported');
        end
    end

end

function dlnet = iMakeDLNetwork()
% make a toy dlnetwork object
layers = [ ...
    imageInputLayer([28 28 1],'Normalization','none')
    convolution2dLayer(5,20)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(10)
    softmaxLayer ];
dlnet = dlnetwork(layers);
end

function seriesNet = iMakeSeriesNetwork()
% make a SeriesNetwork object
layers = [ ...
    imageInputLayer([28 28 1],'Normalization','none')
    convolution2dLayer(5,20,'Weights',ones(5,5,1,20),'Bias',ones(1,1,20))
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(10,'Weights',ones(10,2880),'Bias',ones(10,1))
    regressionLayer];
seriesNet = assembleNetwork(layers);
end

function multiInputNet = iMakeMultiInputNet()
% make a multi-input network object
branch1 = [ imageInputLayer([28 28 1], Normalization="none")
    convolution2dLayer(3, 16)
    batchNormalizationLayer()
    reluLayer()
    globalAveragePooling2dLayer()
    flattenLayer()
    additionLayer(2, Name="add")
    fullyConnectedLayer(2) ];
 
branch2 = [ featureInputLayer(7)
    fullyConnectedLayer(16)
    swishLayer()
    fullyConnectedLayer(16, Name="fc2") ];
 
lg = layerGraph(branch1);
lg = lg.addLayers( branch2 );
lg = lg.connectLayers( "fc2", "add/in2" );
 
multiInputNet = dlnetwork(lg);
end

function noActivationNet = iMakeNoActivationNet()
layers = [ ...
    imageInputLayer([28 28 1],'Normalization','none')
    convolution2dLayer(5,20)
    batchNormalizationLayer
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(10)
    ];
noActivationNet = dlnetwork(layers);
end

function seqNet = iMakeSeqToSeqRNN()
layers = [ ...
    sequenceInputLayer(12)
    lstmLayer(64)
    reluLayer()
    fullyConnectedLayer(9)
    softmaxLayer()
    ];
seqNet = dlnetwork(layers);
end

function multiOutputNet = iMakeMultiOutputNetwork()
lg = layerGraph([imageInputLayer([5,4,3],Normalization="none");maxPooling2dLayer(2,Padding="same",HasUnpoolingOutputs=true,Stride=2);reluLayer;maxUnpooling2dLayer(Name="unpool")]);
lg = lg.connectLayers("maxpool/indices","unpool/indices");
lg = lg.connectLayers("maxpool/size","unpool/size");
multiOutputNet = dlnetwork(lg);
end

function invalidThreshold = iInvalidThreshold()
invalidThreshold = {{-1,'MATLAB:expectedNonnegative'},...
            {[0.1 0.2],'MATLAB:expectedScalar'},...
            {NaN,'MATLAB:notLessEqual'}};
end