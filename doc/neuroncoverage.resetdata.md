  resetData
=========

Reset neuron coverage test data

Syntax
------

`ncReset = resetData(nc)`

Description
-----------

`ncReset = resetData(nc)` resets the test data in the `neuronCoverage`object `nc`.

Input Arguments
---------------

### `nc` — Neuron coverage  
**Value:** `neuronCoverage` object

Neuron coverage, specified as a `neuronCoverage` object.

Output Arguments
----------------

### `ncReset` — Updated neuron coverage  
**Value:** `neuronCoverage` object.

Updated neuron coverage, returned as a `neuronCoverage` object. The function resets the test data information in the `neuronCoverage` and sets the `LayerCoverage` and `AggregateCoverage` properties to an empty table and an empty array, respectively.

To add new test data to the `neuronCoverage` object, use the `addData` function.

© 1994-2022 The MathWorks, Inc.
