#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace svd {

__device__ void decomposition(int m, int n, double** B, double** U, double* S, double** V);

__device__ void dec(int m, int n, int p, double **A, double** U, double* S, double** V);

__device__ void solve(int m, int n, int nx, double **A, double **B, double** X);

__device__ void test();

}