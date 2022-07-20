addData
=======

Update neuron coverage with additional data

Syntax
------

`ncUpdated = addData(nc,X)`

Description
-----------

`ncUpdated = addData(nc,X)` updates the `neuronCoverage` object `nc` with additional test data `X`.

The `LayerCoverage` and `AggregateCoverage` properties of the `neuronCoverage` object automatically update to include the additional test data.


Input Arguments
---------------

### `nc` — Neuron coverage
**Value:** `neuronCoverage` object

Neuron coverage, specified as a `neuronCoverage`object.

### `X` — Test data  
**Value:** `dlarray` object | cell array of `dlarray` objects

Test data to add to the `neuronCoverage` object, specified as a `dlarray` object or a cell array of `dlarray` objects. The test data format depends on the number of network input layers.

*   For networks with a single input layer, specify the test data as a `dlarray` object.
    
*   For networks with multiple input layers, specify the test data as a cell array of `dlarray` objects, where each element of the cell array corresponds to an input layer. For example, for a network with three input layers, the data must be specified as `X={X1,X2,X3}`, where `X1`, `X2`, and `X3` are `dlarray` objects.
    

Output Arguments
----------------

### `ncUpdated` — Updated neuron coverage  
**Value:** `neuronCoverage` object

Updated neuron coverage, returned as a `neuronCoverage` object.

The `LayerCoverage` and `AggregateCoverage` properties of the `neuronCoverage` object automatically update to include the additional test data.

© 2022 The MathWorks, Inc.
