#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace qr {

__device__ void dec(int m, int n, int p, double **A, double** QR, double* RDiag);
__device__ bool isFullRank(int p, double* RDiag);
__device__ bool solve(int m, int n, double** A, double *b, double* x_);
__device__ bool solve(int m, int n, int nx, double** A, double **B, double** X_);
__device__ void test();

// void test();

}