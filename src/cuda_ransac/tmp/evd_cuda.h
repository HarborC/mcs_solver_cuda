#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace evd {

__device__ void tred2(int n, double** V, double* d, double* e);
__device__ void hqr2(int n1, double** V, double* d, double* e, double **H, double *ort);
__device__ void cdiv(double xr, double xi, double yr, double yi, double* cdivr, double* cdivi);
__device__ void others(int n, double** V, double **H, double *ort);
__device__ void tql2(int n, double** V, double* d, double* e);
__device__ void dec(int n, double** A, double** V, double* d);
__device__ void getRealV(int n, double** V, double* d);
// __device__ void test();

// void test();

}