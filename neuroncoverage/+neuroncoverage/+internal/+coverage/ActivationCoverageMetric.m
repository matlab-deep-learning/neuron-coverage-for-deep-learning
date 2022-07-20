classdef ActivationCoverageMetric < handle
    % ActivationCoverageMetric Implementation of the Neuron Coverage
    % metric, as proposed in "DeepXplore: Automated Whitebox Testing of
    % Deep Learning Systems", available here:
    % https://arxiv.org/pdf/1705.06640
    % 
    % This metric computes neuron coverage. First, for each layer, neurons
    % are rescaled across all channel, space and temporal dimensions such
    % that the largest activation value is 1 and the smallest is 0. Neurons
    % are then average pooled in all spatio-temporal dimensions, resulting
    % in 'C' neurons, where C is the channel dimension of the output of the
    % layer. The neuron coverage on each layer is the fraction of neurons
    % that exceed an activation threshold for at least one test data input.
    %
    % For example, suppose you have a layer with activation output size and
    % format 24(S)x24(S)x3(C). This layer has 3 averaged pooled neurons and
    % the neuron coverage will be 0, 1/3, 2/3, or 1.
    % 
    % Neuron coverage is usually applied after the spatial dimensions have
    % been flattened, for example after a fully connected layer. To
    % generalize neuron coverage to other layers, the activations are first
    % flattened by averaging the neurons across all spatial and time
    % dimensions.
    %
    %   Copyright 2022 The MathWorks, Inc.

    properties
        ActivationMap
    end

    properties (SetAccess = ?neuronCoverage)
        ActivationThreshold
    end

    methods (Access = public)
        % Constructor
        function this = ActivationCoverageMetric(activationThreshold, layerNames)
            this.ActivationThreshold = activationThreshold;
            this.ActivationMap = this.initializeActivationMap(layerNames);
        end

        % populate activations
        function addActivation(this, layerName, actData)
            % addActivation   Add activation actData to ActivationTable for
            % layer layerName
            actData = gather(actData);

            spatiotemporalDimsIdx = [finddim(actData,'S'),finddim(actData, 'T')];
            obsDim = finddim(actData,'B');

            actData = extractdata(actData);

            if isempty(obsDim) || (size(actData, obsDim) == 1)
                % Handle one observation case.
                this.processOneObservation(layerName, actData, spatiotemporalDimsIdx);
            else
                % Handle batch case
                numObservations = size(actData, obsDim);
                this.processBatch(layerName, actData, spatiotemporalDimsIdx, numObservations, obsDim);
            end
        end

        % initialize activation map
        function activationMap = initializeActivationMap(~, layerNames)
            numLayers = numel(layerNames);
            initialValue = zeros(1,numLayers); 

            activationMap = containers.Map(layerNames, initialValue, "UniformValues", false);
        end       
 
        function coverageData = computeCoverage(this)
            % computeCoverage   Computes coverage data using the activation
            % threshold stored inside the metric object
            layerNames = keys(this.ActivationMap);

            LayerCoverage = ones(numel(layerNames)+1,1);

            totalNeurons = 0;
            totalActivated = 0;
            for k = 1:numel(layerNames)
                layerName = layerNames{k};

                activations = this.ActivationMap(layerName);
                covMap = activations > this.ActivationThreshold;

                numNeurons = numel(activations);
                numActivated = sum(covMap, 'all');
                LayerCoverage(k,1) = numActivated/numNeurons;

                totalNeurons = totalNeurons + numNeurons;
                totalActivated = totalActivated + numActivated;

            end
            LayerCoverage(end,1) =  totalActivated/totalNeurons;

            % prepare coverage table
            layerNames{end+1} = 'aggregated';
            covTable = table(LayerCoverage, 'RowNames', layerNames);
            coverageData = neuroncoverage.internal.coverage.CoverageData(covTable);
        end
    end

    methods (Access = protected)
        function out = scale(~, in)
            out = rescale(in, 0, 1);
        end

    end

    methods (Access = private)
        function this = processOneObservation(this, layerName, activations, spatiotemporalDimsIdx)
            % processOneObservation   Process one observation, scaling it,
            % apply normalization if needed and then updating the
            % activation table.
            activations = this.scale(activations);

            if iNeedsMean(spatiotemporalDimsIdx)
                activations = mean(activations, spatiotemporalDimsIdx);
            end

            this.fillActivationTable(layerName, activations);
        end

        function this = processBatch(this, layerName, activationsData, spatiotemporalDimsIdx, numObservations, obsDim)
            % processBatch   Process one batch, scaling it,
            % apply normalization if needed and then updating the
            % activation table.
            actSize = size(activationsData);

            % idxVector is a cell array of size 1xN, where N is the
            % number of dimensions of activationsData. The i-th cell
            % array contains a numeric array 1:size of i-th dimension.
            idxVector(:) = {':'};
            for k = 1:numel(actSize)
                idxVector{1,k} = 1:actSize(k);
            end

            for p = 1:numObservations
                idxVector{obsDim} = p;
                activations = activationsData(idxVector{:});

                this.processOneObservation(layerName, activations, spatiotemporalDimsIdx);
            end
        end

        function this = fillActivationTable(this, layerName, activations)
            % fillActivationTable   Fill the activation table using key
            % value syntax. Since we are interested in all the neurons
            % that are > a given threshold, we just store the maximum value
            % of each neuron in the data that we analyzed. If that maximum
            % value is above the threshold, it means that that neuron is
            % activated by at least one observation, and hence it is
            % considered covered.
            this.ActivationMap(layerName) = max(activations, ...
                this.ActivationMap(layerName));
        end

    end

end

function tf = iNeedsMean(spatiotemporalDimsIdx)
if ~isempty(spatiotemporalDimsIdx)
    tf = true;
else
    tf = false;
end
end