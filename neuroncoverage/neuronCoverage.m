classdef neuronCoverage < neuroncoverage.internal.coverage.NeuronCoverage
    %NEURONCOVERAGE   Neuron coverage 
    %
    %       NC = NEURONCOVERAGE(NET) creates a neuronCoverage
    %       object for the network NET. NET must be a dlnetwork object.
    %       
    %       NC = NEURONCOVERAGE(NET,THRESHOLD) creates a neuronCoverage
    %       object and specifies the threshold. The default threshold is
    %       0.75.
    %
    %       NC = NEURONCOVERAGE(__,'PARAM1',VAL1,'PARAM2',VAL2, ...)
    %       creates a neuronCoverage object specifying additional
    %       options using one or more name-value arguments.
    %    
    %           'LayerNames' - Names of layers to compute neuron coverage
    %           on, specified as a character vector, string array, or cell
    %           array of character vectors. By default, neuronCoverage
    %           computes the neuron coverage for the following built-in
    %           layers: ClippedReLULayer, ELULayer, LeakyReLULayer,
    %           ReLULayer, SigmoidLayer, SoftmaxLayer, SoftplusLayer,
    %           SwishLayer, and TanhLayer.
    %
    %           'Data' - Data on which to compute neuron coverage. Data
    %           must be a dlarray object or a cell array of dlarray
    %           objects.
    %
    %   neuronCoverage properties:
    %       Network             -   Network to use in neuron coverage computation
    %       Threshold           -   Neuron coverage threshold
    %       LayerNames          -   Names of layers to compute neuron coverage on
    %       LayerCoverage       -   Neuron coverage for each layer in LayerNames
    %       AggregateCoverage   -   Weighted average neuron coverage across each layer in LayerNames
    %
    %   neuronCoverage methods:
    %       addData             -   Update neuron coverage with additional
    %                               data
    %       resetData           -   Reset neuron coverage
    %       getCoverageForLayer -   Return neuron coverage for specified layers
    %
    %   References 
    %   ---------- 
    %   - Pei, Kexin, et al. "DeepXplore: Automated
    %   Whitebox Testing of Deep Learning Systems." Proceedings of the 26th
    %   Symposium on Operating Systems Principles, Oct. 2017, pp. 1–18.
    %   arXiv.org, https://doi.org/10.1145/3132747.3132785.
    %
    %   - https://github.com/peikexin9/deepxplore
    
    % Copyright 2022 The MathWorks, Inc.

    properties (SetAccess = private)
        % Network    Network to use in neuron coverage computation.
        Network
    end

    properties (Dependent)
        % Threshold   Neuron coverage threshold.
        Threshold
    end
    
    properties (SetAccess = protected)
        % LayerNames    Names of layers to compute neuron coverage on.
        LayerNames
    end
    
    properties (Hidden, SetAccess = protected)
        % Coverage metric for internal calculations.
        CoverageMetric

        % Coverage data collected for internal calculations.
        CoverageDataInternal
    end

    properties (SetAccess = private, Dependent)
        % LayerCoverage   Neuron coverage for each layer in LayerNames.
        LayerCoverage

        % AggregateCoverage   Weighted average neuron coverage across all
        % layers in LayerNames. The aggregate coverage is the weighted sum
        % of the neuron coverage for each layer. The weight for each layer
        % is the number of neurons in that layer divided by the total
        % number of neurons in all the layers in LayersNames.
        AggregateCoverage
    end

    methods (Access = public)
        function this = neuronCoverage(net, threshold, nameValueArgs)
            arguments
                net {mustBeA(net,"dlnetwork")}
                threshold {mustBeValidThreshold(threshold)} = 0.75
                nameValueArgs.LayerNames {mustBeText(nameValueArgs.LayerNames)} = string.empty()
                nameValueArgs.Data {mustBeValidData(nameValueArgs.Data)} = cell(1,0)
            end

            this.Network = net;

            % Validate LayerNames provided by the user, or otherwise
            % consider the LayerNames to be the default activation layers.
            layerNames = string(nameValueArgs.LayerNames);
            if isempty(layerNames)
                try
                    this.LayerNames = this.populateLayerNames();
                catch MX
                    error(MX.identifier,MX.message);
                end
            else
                try
                    this.validateLayersNames(layerNames);
                    % set the LayerNames removing any duplicates
                    this.LayerNames = unique(layerNames,'stable');
                catch MX
                    error(MX.identifier,MX.message);
                end
            end

            % Create the CoverageMetric class
            this.CoverageMetric = neuroncoverage.internal.coverage.ActivationCoverageMetric(threshold, this.LayerNames);

            % Set the threshold
            this.Threshold = threshold;

            % Add the data if provided
            if ~isempty(nameValueArgs.Data)
                this = this.addData(nameValueArgs.Data);
            end
        end

        function this = addData(this,inputData)
            % addData   Update neuron coverage with additional data.
            arguments
                this
                inputData {mustBeValidData(inputData)}
            end

            if ~isa(inputData, "cell")
                inputData = {inputData};
            end

            if ~isempty(inputData)
                try
                    this = addNumericData(this,inputData);
                catch MX
                    if isequal(MX.identifier,'nnet_cnn:dlnetwork:MustSpecifyOutputForMultipleOutputLayer')
                        error('neuroncoverage:internal:coverage:neuronCoverage:MultiOutputLayersNotSupported', ...
                            "Unable to compute neuron coverage for layer '%s'. Layers with multiple outputs are not supported.", ...
                            iFindMultiOutputLayer(this.Network,this.LayerNames));
                    else
                        throwAsCaller(MX);
                    end
                end
            end
        end

        function this = resetData(this)
            % resetData   Reset neuron coverage.
            this = resetData@neuroncoverage.internal.coverage.NeuronCoverage(this);
        end

        function netCov = getCoverageForLayer(this, layername)
            % getCoverageForLayer   Return neuron coverage for specified
            % layers.
            netCov = getCoverageForLayer@neuroncoverage.internal.coverage.NeuronCoverage(this, layername);
        end
    end

    % Getters and setters
    methods
        function this = set.Threshold(this,value)
            arguments
                this
                value {mustBeValidThreshold(value)}
            end

            % update the coverage metrics if the threshold has changed
            if ~isequal(value, this.Threshold)
                this.CoverageMetric.ActivationThreshold = value;
                if ~isempty(this.CoverageDataInternal)
                    this.CoverageDataInternal = this.CoverageMetric.computeCoverage();
                end
            end
        end

        function val = get.Threshold(this)
            val = this.CoverageMetric.ActivationThreshold;
        end

        function netCov = get.AggregateCoverage(this)
            if isempty(this.CoverageDataInternal)
                netCov = [];
            else
                netCov = this.CoverageDataInternal.getAggregatedCoverage();
            end
        end

        function tb = get.LayerCoverage(this)
            if isempty(this.CoverageDataInternal)
                tb = table([]);
            else
                tb = this.CoverageDataInternal.getCoverageTable(this.LayerNames);
            end
        end

        function out = saveobj(this)
            out.Version = 1.0;
            out.Network = this.Network;
            out.LayerNames = this.LayerNames;
            
            % Hidden properties
            out.ActivationMap = this.CoverageMetric.ActivationMap;
            out.ActivationThreshold = this.CoverageMetric.ActivationThreshold;
        end
    end

    methods(Static)
        function nc = loadobj(in)
            nc = neuronCoverage(in.Network, in.ActivationThreshold, 'LayerNames', in.LayerNames);

            % Hidden properties
            nc.CoverageMetric = neuroncoverage.internal.coverage.ActivationCoverageMetric(in.ActivationThreshold, in.LayerNames);
            nc.CoverageMetric.ActivationMap = in.ActivationMap;
            nc.CoverageDataInternal = nc.CoverageMetric.computeCoverage();
        end
    end
end

%% Helpers
function mustBeValidThreshold(value)
validateattributes(value,'numeric',{'scalar','nonnegative','<=',1})
end

function mustBeValidData(value)
if ~isa(value, "dlarray") && (~isa(value, "cell") || ~all(cellfun(@(value) isa(value,"dlarray"),value)))
    error('neuroncoverage:internal:coverage:neuronCoverage:DataFormatNotCorrect', ...
        "'Data' value must be a dlarray object or a cell array of dlarray objects.");
end
end

function val = iFindMultiOutputLayer(dlnet,layersNames)
% Find the layer with more than 1 output
netLayersNames = string({dlnet.Layers(:).Name});

for p = 1 : numel(layersNames)
    idx = find(strcmp(netLayersNames,layersNames(p)) == 1);
    if dlnet.Layers(idx).NumOutputs > 1
        val = dlnet.Layers(idx).Name;
        break;
    end
end
end