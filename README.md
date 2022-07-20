# Neuron Coverage for Deep Learning Networks

This repository enables you to compute neuron coverage for a deep neural network. 

Neuron coverage is a metric for measuring the parts of a network activated by a particular set of test data. This example computes the neuron coverage using neuronCoverage. This implementation of neuron coverage is based on the DeepXplore framework [1]. 

A neuron is activated if, after rescaling and spatial averaging, the absolute value of the neuron is greater than the activation threshold for at least one observation in the test data. The neuron coverage of a layer is then the fraction of activated neurons. For more information, see the Algorithms section of the [neuronCoverage doc](doc/neuronCoverage.md).

## Requirements
- [MATLAB&reg;](http://www.mathworks.com) R2022a or later
- [Deep Learning Toolbox&trade;](https://www.mathworks.com/products/deep-learning.html)

## Get Started
Download or clone this repository to your machine and open it in MATLAB.
Add the neuroncoverage directory to the search path. Go to the location of the repository and run the command: `addpath('neuroncoverage\')`.

Compute the neuron coverage for a network `net` using:
`nc = neuronCoverage(net,threshold,LayerNames=myLayers,Data=myData)`
Use this object to probe the coverage for the layers specified by `myLayers` using the data `myData` and threshold `threshold`. The object also contains the aggregate coverage across all of the layers specified. Change the threshold using: `nc.threshold = threshold/2` for example. Add more data using: `nc = addData(nc, moreData)` function to see how the coverage changes on the layers.

Run [`NeuronCoverageForDeepLearning.mlx`](examples/NeuronCoverageForDeepLearningNetworkExample.mlx) to compute neuron coverage for an image classification network.

## Objects and Functions
The following constructor function that creates the neuron coverage object is:
- `neuronCoverage` - Creates `neuronCoverage` object 

This object has five properties:
- `Network` - Network to use in neuron coverage computation
- `Threshold` - Neuron coverage threshold
- `LayerNames` - Names of layers to compute neuron coverage on
- `LayerCoverage` - Neuron coverage for each layer in LayerNames
- `AggregateCoverage` - Weighted average neuron coverage across all layers in LayerNames

This object has three methods:
- `addData` - Add more data to the coverage metric
- `getCoverageForLayer` - Query coverage on a layer(s)
- `resetData` - Clear all data from the coverage metric

## Documentation
Documentation and examples showing how to compute the neuron coverage.

### Workflow Examples (Long)
- [Neuron Coverage for Deep Learning Network](examples/NeuronCoverageForDeepLearningNetworkExample.mlx)
- [Limitation of Neuron Coverage](examples/LimitationsOfNeuronCoverage.mlx)

### Reference
- [`neuronConverage.md`](doc/neuronCoverage.md)
- [`neuronConverage.adddata.md`](doc/neuroncoverage.adddata.md)
- [`neuronConverage.getCoverageForLayer.md`](doc/neuroncoverage.getcoverageforlayer.md)
- [`neuronConverage.resetData.md`](doc/neuroncoverage.resetdata.md)

### Reference Examples (Short)
- [Add Data Using `miniBatchQueue`](examples/AddDataUsingMinibatchqueueExample.m)
- [Compute Neuron Coverage for Every Layer](examples/NeuronCoverageForEveryLayerExample.m)
- [Compute Neuron Coverage for Test Data](examples/NeuronCoverageForTestData.m)

## References
[1] Pei, Kexin, Yinzhi Cao, Junfeng Yang, and Suman Jana. “DeepXplore: Automated Whitebox Testing of Deep Learning Systems.” GetMobile: Mobile Computing and Communications 22, no. 3 (January 17, 2019): 36–38. https://doi.org/10.1145/3308755.3308767.

Copyright (c) 2022, The MathWorks, Inc.
