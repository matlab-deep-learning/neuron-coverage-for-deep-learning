classdef (Abstract) NeuronCoverage
    % NeuronCoverage   Interface for neuron coverage classes.

    %   Copyright 2022 The MathWorks, Inc.

    properties (Abstract, SetAccess = protected)
        % LayerNames    Names of layers to compute neuron coverage on.
        LayerNames
    end

    properties (Abstract, SetAccess = protected)
        % Coverage metric for internal calculations.
        CoverageMetric

        % Coverage data collected for internal calculations.
        CoverageDataInternal
    end

    properties(Abstract, SetAccess = private)
        % Network    The dlnetwork under analysis.
        Network

        % LayerCoverage   Neuron coverage for each layer in LayerNames.
        LayerCoverage

        % AggregateCoverage   Weighted average neuron coverage across each
        % layer in LayerNames. The weight for each layer is a fraction,
        % number of neurons on that layer over sum total number of neurons
        % on all layers in LayerCoverage.
        AggregateCoverage
    end

    methods (Access = public)

        function this = resetData(this)
            % resetData   Reset neuron coverage.
            this.CoverageDataInternal = neuroncoverage.internal.coverage.CoverageData.empty();
            this.CoverageMetric = neuroncoverage.internal.coverage.ActivationCoverageMetric(this.Threshold, this.LayerNames);
        end

        function netCov = getCoverageForLayer(this, layername)
            % getCoverageForLayer   Return neuron coverage for specified
            % layers.
            arguments
                this
                layername {mustBeText(layername)}
            end

            [tf, badLayerName] = iIsNetworkLayersNames(this.LayerNames,layername);
            % layername could be a char array
            if ~tf
                error('neuroncoverage:internal:coverage:neuronCoverage:NeuronCoverageNotComputedForThisLayer', ...
                    "Unable to find neuron coverage for layer '%s'. To compute the neuron coverage for layer '%s', specify '%s' in the 'LayerNames' value of neuronCoverage.", ...
                    badLayerName, badLayerName, badLayerName);
            end

            if isempty(this.LayerCoverage)
                netCov = [];
            else
                netCov = this.CoverageDataInternal.getCoverage(layername);
            end
        end

    end

    methods (Abstract)
        % addData   Update neuron coverage with additional data.
        addData(inputData)
    end

    methods (Access = protected)

        function this = addNumericData(this,inputData)
            layerActivations = this.getLayerActivations(inputData);

            for k = 1:numel(this.LayerNames)
                layerName = this.LayerNames{k};
                this.CoverageMetric.addActivation(layerName, layerActivations{k});
                this.CoverageDataInternal = this.CoverageMetric.computeCoverage();
            end
        end

        function layerouts = getLayerActivations(this, inputs)
            layerouts = cell(numel(this.LayerNames),1);
            [layerouts{:}] = this.Network.predict(inputs{:}, 'Outputs', this.LayerNames);
        end

        function layerNames = populateLayerNames(this)
            % populateLayerNames   Populate the LayerNames list.
            % These are the layers that contribute to the metric
            % and that will appear in the LayerCoverage table.
            sortedlayers = this.Network.Layers(this.Network.TopologicalOrder);

            supportedLayerFlag = arrayfun(@iIsSupportedLayer,sortedlayers);
            if any(supportedLayerFlag)
                % cast to string array
                layerNames = string({sortedlayers(supportedLayerFlag).Name});
            else
                error('neuroncoverage:internal:coverage:neuronCoverage:DefaultActivationLayersNotFoundInNetwork', ...
                    "Unable to select default layers. To compute the neuron coverage, specify layer names using the 'LayerNames' argument.");
            end
        end

        function validateLayersNames(this, layersNames)
            % validateLayersNames   Cross check the custom LayerNames
            % provided match those in the network
            netLayersNames = string({this.Network.Layers(:).Name});

            [tf, badLayerName] = iIsNetworkLayersNames(netLayersNames,layersNames);
            if ~tf
                error('neuroncoverage:internal:coverage:neuronCoverage:LayerNotFoundInNetwork', ...
                    "Invalid layer names '%s'.", badLayerName);
            end
        end

    end

end

%% Helpers
function tf = iIsSupportedLayer(layer)

tf = isa(layer,'nnet.cnn.layer.ReLULayer') || ...
    isa(layer,'nnet.cnn.layer.LeakyReLULayer') || ...
    isa(layer,'nnet.cnn.layer.ClippedReLULayer') || ...
    isa(layer,'nnet.cnn.layer.TanhLayer') || ...
    isa(layer,'nnet.cnn.layer.ELULayer') || ...
    isa(layer,'nnet.cnn.layer.SwishLayer') || ...
    isa(layer,'rl.layer.SoftplusLayer') || ...
    isa(layer,'nnet.cnn.layer.SoftmaxLayer') || ...
    isa(layer,'nnet.cnn.layer.Sigmoid');

end

function [tf,badLayerName] = iIsNetworkLayersNames(netLayersNames,layersNames)
% Check that LayerNames match the networks layer names and find a missing
% layer if present

% convert to string from char array
layersNames = string(layersNames);

tf = zeros(size(layersNames));
for p = 1 : numel(layersNames)
    tf(p) = any(strcmp(netLayersNames,string(layersNames(p))));
end

if ~all(tf)
    badIdx = find(~tf,1);
    badLayerName = layersNames(badIdx);
else
    badLayerName = [];
end

tf = all(tf);
end