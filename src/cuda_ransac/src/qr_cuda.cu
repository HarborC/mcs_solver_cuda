#include "qr_cuda.h"
#include "utils_cuda.h"

// #include <iostream>
// #include <cmath>
// using namespace std;

namespace qr {
	
__device__ void dec(int m, int n, int p, double **A, double** QR, double* RDiag) {
	// main loop.
	for (int k = 0; k<p; ++k) {
		// Compute 2-norm of k-th column without under/overflow.
		double nrm = 0;
		for (int i = k; i<m; ++i)
			nrm = hypot(nrm, QR[i][k]);
		//        for( int i=k; i<m; ++i )
		//            nrm += QR[i][k]*QR[i][k];
		//        nrm = sqrt(nrm);

		if (nrm != 0) {
			// Form k-th Householder vector.
			if (QR[k][k] < 0)
				nrm = -nrm;

			for (int i = k; i<m; ++i)
				QR[i][k] /= nrm;

			QR[k][k] += 1;

			// Apply transformation to remaining columns.
			for (int j = k + 1; j<n; ++j) {
				double s = 0;
				for (int i = k; i<m; ++i)
					s += QR[i][k] * QR[i][j];

				s = -s / QR[k][k];
				for (int i = k; i<m; ++i)
					QR[i][j] += s*QR[i][k];
			}
		}

		RDiag[k] = -nrm;
	}
}

__device__ bool isFullRank(int p, double* RDiag) {
	for (int j = 0; j<p; ++j)
	if (RDiag[j] == 0)
		return false;

	return true;
}

__device__ bool solve(int m, int n, double** A, double *b, double* x_) {
	int p = min(m, n);

	double** QR;
	double* RDiag;
	malloc_matrix(m, n, &QR);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			QR[i][j] = A[i][j];
		}
	}
	malloc_vector(p, &RDiag);

	dec(m, n, p, A, QR, RDiag);

	// matrix is rank deficient
	if (!isFullRank(p, RDiag)) {
		free_matrix(m, n, QR);
		free_vector(p, RDiag);
		return false;
	}

	double* x;
	malloc_vector(n, &x);
	for (int i = 0; i<n; ++i)
		x[i] = b[i];

	// compute y = transpose(Q)*b
	for (int k = 0; k<n; ++k) {
		double s = 0;
		for (int i = k; i<m; ++i)
			s += QR[i][k] * x[i];

		s = -s / QR[k][k];
		for (int i = k; i<m; ++i)
			x[i] += s*QR[i][k];
	}

	// solve R*x = y;
	for (int k = n - 1; k >= 0; --k) {
		x[k] /= RDiag[k];
		for (int i = 0; i<k; ++i)
			x[i] -= x[k] * QR[i][k];
	}

	// return n portion of x
	for (int i = 0; i<n; ++i)
		x_[i] = x[i];

	free_vector(n, x);
	free_matrix(m, n, QR);
	free_vector(p, RDiag);
	return true;
}

__device__ bool solve(int m, int n, int nx, double** A, double **B, double** X_) {
	int p = min(m, n);

	double** QR;
	double* RDiag;
	malloc_matrix(m, n, &QR);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			QR[i][j] = A[i][j];
		}
	}
	malloc_vector(p, &RDiag);

	dec(m, n, p, A, QR, RDiag);

	// matrix is rank deficient
	if (!isFullRank(p, RDiag)) {
		free_matrix(m, n, QR);
		free_vector(p, RDiag);
		return false;
	}

	double** X;
	malloc_matrix(m, nx, &X);
	for (int i = 0; i<m; ++i) {
		for (int j = 0; j<nx; ++j) {
			X[i][j] = B[i][j];
		}
	}
	
	// compute Y = transpose(Q)*B
	for (int k = 0; k<n; ++k)
		for (int j = 0; j<nx; ++j) {
			double s = 0;
			for (int i = k; i<m; ++i)
				s += QR[i][k] * X[i][j];

			s = -s / QR[k][k];
			for (int i = k; i<m; ++i)
				X[i][j] += s*QR[i][k];
		}

	// solve R*X = Y;
	for (int k = n - 1; k >= 0; --k) {
		for (int j = 0; j<nx; ++j)
			X[k][j] /= RDiag[k];

		for (int i = 0; i<k; ++i)
			for (int j = 0; j<nx; ++j)
				X[i][j] -= X[k][j] * QR[i][k];
	}

	// return n x nx portion of X
	// Matrix<double> X_(n, nx);
	for (int i = 0; i<n; ++i)
		for (int j = 0; j<nx; ++j)
			X_[i][j] = X[i][j];

	free_matrix(m, nx, X);
	free_matrix(m, n, QR);
	free_vector(p, RDiag);

	return true;
}

__device__ void test() {
	int m = 4;
    int n = 3;
    int nx = 4;
    double** A;
    double** B;
    double** X;

    double A_vec[] = {1,0,0,
					  1,2,4,
					  1,3,9,
					  1,3,9};

    malloc_matrix(m, n, &A);
	memcopy_matrix(m, n, A, &(A_vec[0]));

    double B_vec[] = {1,0,0,0,
                      0,1,0,0,
					  0,0,1,0,
					  0,0,0,1};
    malloc_matrix(m, nx, &B);
	memcopy_matrix(m, nx, B, &B_vec[0]);

    malloc_matrix(n, nx, &X);
    solve(m, n, nx, A, B, X);

    print_matrix(n, nx, X);
}

}