getCoverageForLayer
===================

Get neuron coverage for network layers

Syntax
------

`layerCoverage = getCoverageForLayer(nc,layerNames)`

Description
-----------

`layerCoverage = getCoverageForLayer(nc,layerNames)` returns the neuron coverage of the layers specified in the `layerNames` property of the `neuronCoverage` object `nc`.


Input Arguments
---------------

### `nc` — Neuron coverage  
**Value:** `neuronCoverage` object

Neuron coverage, specified as a `neuronCoverage` object.

### `layerNames` — Names of layers  
**Value:** string array | character vector | cell array of character vectors

Names of layers, specified as a character vector, string array, or cell array of character vectors. `layerNames` must contain the names of layers in the `neuronCoverage` object. To see the available layers, query the `LayerNames` property of the `neuronCoverage` object.

To compute the neuron coverage for layers that are not listed in the `LayerNames` property, create a new `neuronCoverage` object and specify the layer names using the `LayerNames` name-value argument. For example, use `neuronCoverage(net,X,LayerNames=["conv_1" "conv_2"])` to compute the neuron coverage for the `conv_1` and `conv_2` layers. For more information, see `LayerNames`.

**Data Types:** `char` | `string` | `cell`

Output Arguments
----------------

### `layerCoverage` — Neuron coverage for each layer  
**Value:** numeric array

Neuron coverage for each layer, returned as a numeric array. For more information on the neuron coverage metric, see the Algorithms section of the `neuronCoverage` object documentation.

© 1994-2022 The MathWorks, Inc.

