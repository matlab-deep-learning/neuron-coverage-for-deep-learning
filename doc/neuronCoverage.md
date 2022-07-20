  neuronCoverage
==============

Neuron coverage for deep learning neural networks

Description
-----------

Neuron coverage is a measurement of the parts of a deep neural network activated by a specific test data set. For more information about neuron coverage and the method that this object uses to calculate it, see [Neuron Coverage](#Algorithm_NeuronCoverage).

You can create a `neuronCoverage` object for a specific activation threshold. You can specify test data using the [`Data`](#Input_Data) name-value argument and add test data using the `addData`function. The software uses the network, threshold, and test data to compute the neuron coverage for each layer specified by the [`LayerNames`](#Properties_LayerNames) property. For more information, see [Algorithms](#Algorithm_NeuronCoverage).

Creation
--------

### Syntax

`nc = neuronCoverage(net)`

`nc = neuronCoverage(net,threshold)`

`nc = neuronCoverage(___,Name=Value)`

### Description

`nc = neuronCoverage(net)` creates a `neuronCoverage` object and sets the [`Network`](#Properties_Network) property. You can specify test data using the [`Data`](#Input_Data) name-value argument or add test data to the `neuronCoverage` object using the `addData` function.

`nc = neuronCoverage(net,threshold)` creates a `neuronCoverage` object using network `net` and sets the [`Threshold`](#Properties_Threshold) property.


`nc = neuronCoverage(___,Name=Value)` creates a `neuronCoverage` object and specifies the test data or sets the optional [`LayerNames`](#Properties_LayerNames) property using name-value arguments in addition to the input arguments in previous syntaxes. For example, to compute the neuron coverage for the `conv1` and `conv2` layers, specify `LayerNames=["conv1" "conv2"]`.

### Input Arguments

**Name-Value Arguments**

Specify optional pairs of arguments as `Name1=Value1,...,NameN=ValueN`, where `Name` is the argument name and `Value` is the corresponding value. Name-value arguments must appear after other arguments, but the order of the pairs does not matter.

**Example:** `neuronCoverage(net,Data=X,LayerNames=["conv1" "conv2"])` creates a `neuronCoverage` object and computes the neuron coverage for the `conv1` and `conv2` layers using network `net` and test data `X`.

### `Data` — Test data  <a name=Input_Data></a>
**Value:** `dlarray` object | cell array of `dlarray` objects

Test data on which to compute the neuron coverage, specified as a `dlarray` object or cell array of `dlarray` objects. The test data format depends on the number of network input layers.

*   For networks with a single input layer, specify the test data as a `dlarray` object.
    
*   For networks with multiple input layers, specify the test data as a cell array of `dlarray` objects, where each element of the cell array corresponds to an input layer. For example, for a network with three input layers, the data must be specified as `X={X1,X2,X3}`, where `X1`, `X2`, and `X3` are `dlarray` objects.
    

To add more test data to the `neuronCoverage` object, use the `addData` function. To reset the data information in the `neuronCoverage` object, use the `resetData` function.

If the test data is a `dlarray` object with underlying data of type `gpuArray`, then `neuronCoverage` uses the GPU. Using a GPU or parallel options requires Parallel Computing Toolbox™ software. Using a GPU also requires a supported GPU device. For information on supported devices, see [GPU Support by Release](https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html) (Parallel Computing Toolbox).

Properties
----------


### `Network` — Trained network   <a name=Properties_Network></a>
**Value:** `dlnetwork` object

Trained network, specified as a `dlnetwork` object. You can get a trained `dlnetwork` object by training a network using a custom training loop. To use a network that you have trained using `trainNetwork`, you must convert the network to a `dlnetwork` object. For more information, see [Tips](#Tips).

### `Threshold` — Activation threshold  <a name=Properties_Threshold></a>
**Value:** `0.75` (default) | scalar in the range \[0, 1\]

Activation threshold, specified as a scalar in the range \[0, 1\]. A neuron is activated if its value is greater than the activation threshold for at least one of the test data observations.

When you change this property, the [`LayerCoverage`](#Properties_LayerCoverage) and [`AggregateCoverage`](#Properties_AggregateCoverage) properties update to use the new threshold.

**Note:** This property is read-only after you create the `neuronCoverage` object.

**Data Types:** `single` | `double` | `int8` | `int16` | `int32` | `int64` | `uint8` | `uint16` | `uint32` | `uint64`

### `LayerNames` — Names of layers  <a name=Properties_LayerNames></a>
**Value:** string array | character vector | cell array of character vectors

Names of layers for which to compute the neuron coverage, specified as a character vector, string array, or cell array of character vectors. You can compute the neuron coverage for built-in or custom layers with a single output. The object does not support layers with multiple outputs The layer names must match the names of layers in the network.

If you do not specify any layers, `neuronCoverage` computes the neuron coverage for these built-in layers: `ClippedReLULayer`, `ELULayer`, `LeakyReLULayer`, `ReLULayer`, `SigmoidLayer`, `SoftmaxLayer`, `SoftplusLayer`, `SwishLayer`, and `TanhLayer`. To specify the layers for which to compute the neuron coverage, specify the [`LayerNames`](#Properties_LayerNames) name-value argument when you create the `neuronCoverage` object.

**Data Types:** `char` | `string` | `cell`

### `LayerCoverage` — Neuron coverage for each layer  <a name=Properties_LayerCoverage></a>
**Value:** table

This property is read-only.

Neuron coverage for each layer specified by the [`LayerNames`](#Properties_LayerNames) property, returned as a table. For more information about computing the layer coverage, see [Neuron Coverage](#Algorithm_NeuronCoverage).

**Data Types:** `table`

### `AggregateCoverage` — Weighted average neuron coverage  <a name=Properties_AggregateCoverage></a>
**Value:** scalar in the range \[0, 1\]

This property is read-only.

Weighted average neuron coverage across each layer specified by the [`LayerNames`](#Properties_LayerNames) property, returned as a scalar in the range \[0, 1\].

The aggregate coverage is the weighted average neuron coverage across each layer in `LayerNames`. For more information about computing the aggregate coverage, see [Aggregate Coverage](#Algorithm_AggregateCoverage).

**Data Types:** `double`


Examples
--------

Tips <a name=Tips></a>
----

To compute the neuron coverage for a `DAGNetwork` or `SeriesNetworks` object, you must convert the network to a `dlnetwork` object.

1.  Convert the network to a `layerGraph` object.
    
2.  Remove the output layers.
    
3.  Convert the network to a `dlnetwork` object.
    

For example, convert the pretrained network `googlenet` to a `dlnetwork`.
```
net = googlenet;
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph,"output");
dlnet = dlnetwork(lgraph)
```
Algorithms
----------

### Neuron Coverage <a name="Algorithm_NeuronCoverage"></a>

To compute the neuron coverage, you need a network and a test data set. The `neuronCoverage` object passes observations from the test data through the network and the computes the values of the neurons in the layers specified by the [`LayerNames`](#Properties_LayerNames) property. The software then rescales the values of the neurons to be in the range \[0, 1\] and averages the neurons across all spatial and time dimensions. The software labels any averaged neuron with a value that is greater than the activation threshold for at least one of the test data observations as _activated_. The neuron coverage for a layer is the proportion of activated neurons. This implementation is based on the DeepXplore framework \[1\].

To compute the neuron coverage for a specific layer and set of test data, `neuronCoverage` performs these steps.

1.  For each observation in the test data:
    
    1.  Pass the observation through the network and find the value of each neuron in the layer.
        
    2.  Rescale the neurons to be in the range \[0, 1\]. If $`n_i^{(l)}`$ denotes the ith neuron in layer $`l`$, then the rescaled neuron is $`\hat{n}_i^{(l)}=\frac{n_i^{(l)}−\min_j n_j^{(l)}}{\max_j n_j^{(l)}−\min_jn_j^{(l)}}`$.
        
    3.  Average the rescaled neurons over all the spatial and time dimensions. For example, if the layer outputs data with "SSC" (spatial, spatial, channel) dimensions, then the software averages the neurons over the two spatial dimensions to produce a 1-by-1-by-$`c`$ array of averaged neurons, where $`c`$ is the number of channels in the layer.
        
    
2.  Label any averaged neuron with a value that is greater than the activation threshold for at least one observation in the test data as activated.
    
3.  Return the neuron coverage for the layer as the number of activated neurons divided by the total number of averaged neurons in the layer.
    

### Aggregate Coverage <a name="Algorithm_AggregateCoverage"></a>

The aggregate coverage is the weighted average neuron coverage across all of the layers specified by the [`LayerNames`](#Properties_LayerNames) property. The software computes the aggregate coverage using the rescaled and averaged neurons. For more information, see [Neuron Coverage](#Algorithm_NeuronCoverage). The weight for each layer is the number of averaged neurons in that layer divided by the total number of averaged neurons across the specified layers. For example, suppose you compute these results for the neuron coverage for three layers.

| Layer Name | Output Size | Number of Averaged Neurons | Number of Activated Neurons | Layer Coverage |
| --- | --- | --- | --- | --- |
| `'conv_1'` | 28-by-28-by-8 | 8 | 2 | 0.25 |
| `'conv_2'` | 14-by-14-by-16 | 16 | 6 | 0.375 |
| `'fc'` | 1-by-1-by-10 | 10 | 7 | 0.7 |

The aggregate coverage across these layers is 134((8×0.25)+(16×0.375)+(10×0.7))\=0.4412.

References
----------

\[1\] Pei, Kexin, Yinzhi Cao, Junfeng Yang, and Suman Jana, “DeepXplore: Automated Whitebox Testing of Deep Learning Systems,” in _Proceedings of the 26th Symposium on Operating Systems Principles_, 1–18, Shanghai China: ACM, 2017. https://doi.org/10.1145/3132747.3132785.

\[2\] _Pei, Kexin. "DeepXplore: Systematic DNN Testing (SOSP'17)." Accessed April 11, 2022_.[https://github.com/peikexin9/deepxplore](https://github.com/peikexin9/deepxplore)

Extended Capabilities
---------------------

### GPU Arrays  <a name="ExtendenCapabilities_GPU"></a>
Accelerate code by running on a graphics processing unit (GPU) using Parallel Computing Toolbox™.

Usage notes and limitations:

*   This function runs on the GPU if its inputs meet either or both of these conditions:
    
    *   Any of the values of the network learnable parameters inside `nc.Network.Learnables.Value` are `dlarray` objects with underlying data of type `gpuArray`.
        
    *   The target data `X` is a `dlarray` with underlying data of type `gpuArray`.
        
    

For more information, see [Run MATLAB Functions on a GPU](#https://www.mathworks.com/help/parallel-computing/run-matlab-functions-on-a-gpu.html) (Parallel Computing Toolbox).


© 1994-2022 The MathWorks, Inc.

