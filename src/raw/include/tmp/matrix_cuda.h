// #pragma once

// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <stdio.h> 
// #include <iostream> 

// struct CMatrix {
// 	double** data;
// 	int m;
// 	int n;

// 	void c_malloc(int m_, int n_) {
// 		m = m_;
// 		n = n_;
// 		data = (double**)malloc(m * sizeof(double*));
// 		for (int i = 0; i < m; i++) {
// 			(data)[i] = (double*)malloc(n * sizeof(double));
// 		}
// 	}

// 	void c_print() {
// 		printf("matrix (%d,%d): \n", m, n);
// 		for (int i = 0; i < m; i++) {
// 			for (int j = 0; j < n; j++) {
// 				printf(" %lf ", data[i][j]);
// 			}
// 			printf("\n");
// 		}
// 	}

// 	void c_memcopy(double* data_) {
// 		for (int i = 0; i < m; i++) {
// 			for (int j = 0; j < n; j++) {
// 				data[i][j] = data_[i*n+j];
// 			}
// 		}
// 	}

// 	void c_free() {
// 		for (int i = 0; i < m; i++) {
// 			free(data[i]);
// 		}
// 		free(data);
// 		data = nullptr;
// 	}
// };

// struct CVector {
// 	double* data;
// 	int n;

// 	void c_malloc(int n_) {
// 		n = n_;
// 		data = (double*)malloc(n * sizeof(double));
// 	}

// 	void c_print() {
// 		printf("vector (%lf): \n", n);
// 		for (int i = 0; i < n; i++) {
// 			printf(" %d ", data[i]);
// 		}
// 		printf("\n");
// 	}

// 	void c_memcopy(double* data_) {
// 		for (int j = 0; j < n; j++) {
// 			data[j] = data_[j];
// 		}
// 	}

// 	void c_free() {
// 		free(data);
// 		data = nullptr;
// 	}
// };

// __device__ void print_matrix(int m, int n, double** A) {
// 	printf("matrix (%d, %d): \n", m, n);
// 	for (int i = 0; i < m; i++) {
// 		for (int j = 0; j < n; j++) {
// 			printf(" %lf ", A[i][j]);
// 		}
// 		printf("\n");
// 	}
// }

// __device__ void print_vector(int m, double* a) {
//     printf("vector (%d): \n", m);
// 	for (int i = 0; i < m; i++) {
// 		printf(" %lf ", a[i]);
// 	}
// 	printf("\n");
// }

// __device__ void malloc_matrix(int m, int n, double*** A) {
//     *A = (double**)malloc(m * sizeof(double*));
// 	for (int i = 0; i < m; i++) {
// 		(*A)[i] = (double*)malloc(n * sizeof(double));
// 	}
// }

// __device__ void malloc_vector(int m, double** a) {
//     *a = (double*)malloc(m * sizeof(double));
// }

// __device__ void memcopy_matrix(int m, int n, double** A, double* A_vector) {
// 	for (int i = 0; i < m; i++) {
// 		for (int j = 0; j < n; j++) {
// 			A[i][j] = A_vector[i*n+j];
// 		}
// 	}
// }

// __device__ void memcopy_vector(int m, double** a, double* a_vector) {
//     for (int i = 0; i < m; i++) {
// 		(*a)[i] = a_vector[i];
// 	}
// }

// __device__ void free_matrix(int m, int n, double** A) {
// 	for (int i = 0; i < m; i++) {
// 		free(A[i]);
// 	}
//     free(A);
// }

// __device__ void free_vector(int m, double* a) {
//     free(a);
// }
