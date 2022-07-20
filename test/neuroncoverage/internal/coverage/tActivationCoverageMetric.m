classdef tActivationCoverageMetric < matlab.unittest.TestCase
    % tActivationCoverageMetric   Unit tests for
    % neuroncoverage.internal.coverage.ActivationCoverageMetric

    %   Copyright 2022 The MathWorks, Inc.

    properties(TestParameter)
        ThresholdAndData = iGetThresholdAndData()
        Data = iGetActivationsData()
    end

    methods(Test)
        %% Positive tests
        function computeCoverageGetsCorrectResults(test, ThresholdAndData)
            % Verify that the computations of the coverage metric match the
            % expected results
            layerNames = keys(ThresholdAndData.ActivationValues);
            threshold = ThresholdAndData.ActivationThreshold;

            metric = neuroncoverage.internal.coverage.ActivationCoverageMetric(threshold, layerNames);

            metric.ActivationMap = ThresholdAndData.ActivationValues;

            covData = metric.computeCoverage();

            % Verify that the bounds are correct.
            actualNetworkCoverage = covData.CoverageTable.LayerCoverage(end);
            test.verifyEqual( actualNetworkCoverage, ThresholdAndData.Expected , 'RelTol', 1e-15, 'AbsTol', 1e-15);

        end

        function canAddActivations(test, Data)
            % Verify that activation data is added correctly into the
            % activation table and that the relative operations happen. 
            threshold = 0.5;
            layerName = 'layer';
            layerNameKey = {layerName};

            metric = neuroncoverage.internal.coverage.ActivationCoverageMetric(threshold, layerNameKey);
            
            metric.addActivation(layerName,Data.Data);

            % Verify that the bounds are correct.
            actualActivationMapData = metric.ActivationMap('layer');
            test.verifyEqual( actualActivationMapData, Data.Expected , 'RelTol', 1e-15, 'AbsTol', 1e-15);

        end
    end

end

function s = iGetThresholdAndData()

data = containers.Map;
data('layer') = [1 0.1 1];
s.OneLayerIntermediate = struct('ActivationThreshold', 0.5, 'ActivationValues',data,'Expected',2/3);

data = containers.Map;
data('layer') = [1 0.1 1];
s.OneLayerEdgeAll = struct('ActivationThreshold', 0,'ActivationValues',data,'Expected',1);

data = containers.Map;
data('layer') = [1 0.1 1];
s.OneLayerEdgeNone = struct('ActivationThreshold', 1, 'ActivationValues',data,'Expected',0);

data = containers.Map;
data('layer1') = [1 0.1 1];
data('layer2') = [0 0.1 0 0.1 0.2];
s.TwoLayers = struct('ActivationThreshold', 0.5,  'ActivationValues',data,'Expected',2/8);

data = containers.Map;
data('layer1') = [1 0.1 1];
data('layer2') = [0 0.1 0 0.1 0.2];
data('layer3') = [0.6 0.8 0.7];
data('layer4') = [1];
data('layer5') = [0.4 0.6];
s.FiveLayers = struct('ActivationThreshold', 0.5,  'ActivationValues',data,'Expected',7/14);

end

function s = iGetActivationsData()
% With C data, only no averaging is performed
data = dlarray([2 4 6],'C');
s.C = struct('Data',data, 'Expected',[0 0.5 1]');

% With SC data, rescaling and averaging are performed
data = dlarray(iArbitraryData([3 5]),'SC');
x = extractdata(data);
x = (x - min(x(:))) ./ (max(x(:)) - min(x(:)));
dataExpected = mean(x,1);
s.SSC = struct('Data', data, 'Expected', dataExpected);

% With SSC data, rescaling and averaging are performed
data = dlarray(iArbitraryData([3 5 4]),'SSC');
x = extractdata(data);
x = (x - min(x(:))) ./ (max(x(:)) - min(x(:)));
dataExpected = mean(x,[1 2]);
s.SSC = struct('Data', data, 'Expected', dataExpected);

% With SSSC data, rescaling and averaging are performed
data = dlarray(iArbitraryData([3 5 6 4]),'SSSC');
x = extractdata(data);
x = (x - min(x(:))) ./ (max(x(:)) - min(x(:)));
dataExpected = mean(x,[1 2 3]);
s.SSSC = struct('Data', data, 'Expected', dataExpected);

% With SCT data, rescaling and averaging are performed
data = dlarray(iArbitraryData([3 5 6]),'SCT');
x = extractdata(data);
x = (x - min(x(:))) ./ (max(x(:)) - min(x(:)));
dataExpected = mean(x,[1 3]);
s.SCT = struct('Data', data, 'Expected', dataExpected);

% With CT data, rescaling and averaging are performed
data = dlarray(iArbitraryData([3 5]),'CT');
x = extractdata(data);
x = (x - min(x(:))) ./ (max(x(:)) - min(x(:)));
dataExpected = mean(x,2);
s.CT = struct('Data', data, 'Expected', dataExpected);

% With SCT data, scaling and normalization are performed
data = dlarray(iArbitraryData([3 5 6]),'SCT');
x = extractdata(data);
x = (x - min(x(:))) ./ (max(x(:)) - min(x(:)));
dataExpected = mean(x,[1 3]);
s.SCT = struct('Data', data, 'Expected', dataExpected);

% With SSCT data, rescaling and averaging are performed
data = dlarray(iArbitraryData([3 5 6 8]),'SSCT');
x = extractdata(data);
x = (x - min(x(:))) ./ (max(x(:)) - min(x(:)));
dataExpected = mean(x,[1 2 4]);
s.SSCT = struct('Data', data, 'Expected', dataExpected);

% With SSSCT data, rescaling and averaging are performed
data = dlarray(iArbitraryData([3 5 6 9 11]),'SSSCT');
x = extractdata(data);
x = (x - min(x(:))) ./ (max(x(:)) - min(x(:)));
dataExpected = mean(x,[1 2 3 5]);
s.SSSCT = struct('Data', data, 'Expected', dataExpected);
end

function data = iArbitraryData(sz) 
data = reshape( (1:prod(sz,'all'))/prod(sz,'all'), sz );
end