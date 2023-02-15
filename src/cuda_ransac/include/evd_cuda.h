#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace evd {

__device__ void tred2(const int n, double* V, double* d, double* e);
__device__ void hqr2(const int n1, double* V, double* d, double* e, double *H, double *ort);
__device__ void cdiv(double xr, double xi, double yr, double yi, double* cdivr, double* cdivi);
__device__ void others(const int n, double* V, double *H, double *ort);
__device__ void tql2(const int n, double* V, double* d, double* e);
__device__ void dec(const int n, double* A, double* V, double* d, double* e, double *H, double *ort, double* cV);
__device__ void getRealV(const int n, double* V, double* d, double *cV);

}