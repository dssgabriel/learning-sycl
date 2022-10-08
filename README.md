# Learning SYCL with Intel DPC++

This repository is my SYCL programming model learning ground.
I use it to test and benchmark small SYCL kernels against standard C++ code.

The code in this repository relies on Intel's Data Parallel C++ (DPC++) implementation of SYCL.

## Build
To build the project, please make sure you have Intel oneAPI Base toolkit as well as CMake 3.12 installed on your machine.
```
cmake -S . -B target -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
cmake --build target
```

You can then execute the binary `target/learn-sycl`.
Note that currently, the program only benchmarks a classic SAXPY kernel.
You must specify the length of the vectors from the command line when running the application, e.g.:
```
target/learn-sycl 1048576
```