# SIMD Simulator

## Build The Kernels

`nvcc kernels.cu`

`cuobjdump -xelf all a.out`

`nvdisasm xxxx-1.sm_52.cubin > kernels.sasm`

## Example tests

Tests 00 - 07 run example tests, performance benchmarks, and unit
tests on the three kernels that are in `kernels.cu` and
`kernels.sasm`. 

`matplotlib` and `numpy` python packages are required to run the
tests.

