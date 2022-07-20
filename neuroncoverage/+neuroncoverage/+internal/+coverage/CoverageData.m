classdef CoverageData < handle
    %   Copyright 2022 The MathWorks, Inc.

    properties    
        CoverageTable  
    end
    
    methods
        function this = CoverageData(covTable)
            this.CoverageTable = covTable;         
        end
        
        function covTable = getCoverageTable(this,layerNames)
            % Do not display the aggregated coverage here
            covTable = head(this.CoverageTable,size(this.CoverageTable,1)-1);

            % Reorder to match the output of LayerNames
            covTable = covTable(layerNames(:),:);
        end
        
        function cov  = getCoverage(this, layername)
            try
                cov = this.CoverageTable{layername,:};
            catch
                error('neuroncoverage:internal:coverage:neuronCoverage:NeuronCoverageNotComputedForThisLayer', ...
                    'Unable to find neuron coverage for layer %s. To compute the neuron coverage for layer %s, specify %s in the LayerNames value of neuronCoverage.', ...
                    layername, layername, layername);
            end
        end
        
        function cov  = getAggregatedCoverage(this)
            cov = this.CoverageTable{'aggregated',:};
        end                                  
               
    end
    
end