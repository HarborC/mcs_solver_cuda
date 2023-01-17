#include "svd_cuda.h"
#include "utils_cuda.h"

namespace svd {

__device__ void decomposition(int m, int n, double** B, double** U, double* S, double** V) {
	double* e;
	double* work;
    malloc_vector(n, &e);
    malloc_vector(m, &work);

	// boolean
	int wantu = 1;
	int wantv = 1;

	// Reduce A to bidiagonal form, storing the diagonal elements
	// in s and the super-diagonal elements in e.
	int nct = min(m - 1, n);
	int nrt = max(0, n - 2);
	int i = 0,
		j = 0,
		k = 0;

	for (k = 0; k<max(nct, nrt); ++k) {
		if (k < nct) {
			// Compute the transformation for the k-th column and
			// place the k-th diagonal in s[k].
			// Compute 2-norm of k-th column without under/overflow.
			S[k] = 0;
			for (i = k; i<m; ++i)
				S[k] = hypot(S[k], B[i][k]);

			if (S[k] != 0) {
				if (B[k][k] < 0)
					S[k] = -S[k];

				for (i = k; i<m; ++i)
					B[i][k] /= S[k];
				B[k][k] += 1;
			}
			S[k] = -S[k];
		}

		for (j = k + 1; j<n; ++j) {
			if ((k < nct) && (S[k] != 0)) {
				// apply the transformation
				double t = 0;
				for (i = k; i<m; ++i)
					t += B[i][k] * B[i][j];

				t = -t / B[k][k];
				for (i = k; i<m; ++i)
					B[i][j] += t*B[i][k];
			}
			e[j] = B[k][j];
		}

		// Place the transformation in U for subsequent back
		// multiplication.
		if (wantu & (k < nct))
            for (i = k; i<m; ++i)
                U[i][k] = B[i][k];

		if (k < nrt) {
			// Compute the k-th row transformation and place the
			// k-th super-diagonal in e[k].
			// Compute 2-norm without under/overflow.
			e[k] = 0;
			for (i = k + 1; i<n; ++i)
				e[k] = hypot(e[k], e[i]);

			if (e[k] != 0) {
				if (e[k + 1] < 0)
					e[k] = -e[k];

				for (i = k + 1; i<n; ++i)
					e[i] /= e[k];
				e[k + 1] += 1;
			}
			e[k] = -e[k];

			if ((k + 1 < m) && (e[k] != 0)) {
				// apply the transformation
				for (i = k + 1; i<m; ++i)
					work[i] = 0;

				for (j = k + 1; j<n; ++j)
                    for (i = k + 1; i<m; ++i)
                        work[i] += e[j] * B[i][j];

				for (j = k + 1; j<n; ++j) {
					double t = -e[j] / e[k + 1];
					for (i = k + 1; i<m; ++i)
						B[i][j] += t * work[i];
				}
			}

			// Place the transformation in V for subsequent
			// back multiplication.
			if (wantv)
                for (i = k + 1; i<n; ++i)
                    V[i][k] = e[i];
		}
	}

	// Set up the final bidiagonal matrix or order p.
	int p = n;

	if (nct < n)
		S[nct] = B[nct][nct];
	if (m < p)
		S[p - 1] = 0;

	if (nrt + 1 < p)
		e[nrt] = B[nrt][p - 1];
	e[p - 1] = 0;

	// if required, generate U
	if (wantu) {
		for (j = nct; j<n; ++j) {
			for (i = 0; i<m; ++i)
				U[i][j] = 0;
			U[j][j] = 1;
		}

		for (k = nct - 1; k >= 0; --k)
            if (S[k] != 0) {
                for (j = k + 1; j<n; ++j) {
                    double t = 0;
                    for (i = k; i<m; ++i)
                        t += U[i][k] * U[i][j];
                    t = -t / U[k][k];

                    for (i = k; i<m; ++i)
                        U[i][j] += t * U[i][k];
                }

                for (i = k; i<m; ++i)
                    U[i][k] = -U[i][k];
                U[k][k] = 1 + U[k][k];

                for (i = 0; i<k - 1; ++i)
                    U[i][k] = 0;
            } else {
                for (i = 0; i<m; ++i)
                    U[i][k] = 0;
                U[k][k] = 1;
            }
	}

	// if required, generate V
	if (wantv)
        for (k = n - 1; k >= 0; --k) {
            if ((k < nrt) && (e[k] != 0))
                for (j = k + 1; j<n; ++j) {
                    double t = 0;
                    for (i = k + 1; i<n; ++i)
                        t += V[i][k] * V[i][j];
                    t = -t / V[k + 1][k];

                    for (i = k + 1; i<n; ++i)
                        V[i][j] += t * V[i][k];
                }

            for (i = 0; i<n; ++i)
                V[i][k] = 0;
            V[k][k] = 1;
        }

	// main iteration loop for the singular values
	int pp = p - 1;
	int iter = 0;
	double eps = pow(2.0, -52.0);

	while (p > 0) {
		int k = 0;
		int kase = 0;

		// Here is where a test for too many iterations would go.
		// This section of the program inspects for negligible
		// elements in the s and e arrays. On completion the
		// variables kase and k are set as follows.
		// kase = 1     if s(p) and e[k-1] are negligible and k<p
		// kase = 2     if s(k) is negligible and k<p
		// kase = 3     if e[k-1] is negligible, k<p, and
		//				s(k), ..., s(p) are not negligible
		// kase = 4     if e(p-1) is negligible (convergence).
		for (k = p - 2; k >= -1; --k) {
			if (k == -1)
				break;

			if (abs(e[k]) <= eps*(abs(S[k]) + abs(S[k + 1]))) {
				e[k] = 0;
				break;
			}
		}

		if (k == p - 2)
			kase = 4;
		else {
			int ks;
			for (ks = p - 1; ks >= k; --ks) {
				if (ks == k)
					break;

				double t = ((ks != p) ? abs(e[ks]) : 0) +
					((ks != k + 1) ? abs(e[ks - 1]) : 0);

				if (abs(S[ks]) <= eps*t) {
					S[ks] = 0;
					break;
				}
			}

			if (ks == k)
				kase = 3;
			else if (ks == p - 1)
				kase = 1;
			else {
				kase = 2;
				k = ks;
			}
		}
		k++;

		// Perform the task indicated by kase.
		switch (kase) {
			// deflate negligible s(p)
		    case 1: {
                double f = e[p - 2];
                e[p - 2] = 0;

                for (j = p - 2; j >= k; --j) {
                    double t = hypot(S[j], f);
                    double cs = S[j] / t;
                    double sn = f / t;
                    S[j] = t;

                    if (j != k) {
                        f = -sn * e[j - 1];
                        e[j - 1] = cs * e[j - 1];
                    }

                    if (wantv)
                        for (i = 0; i<n; ++i) {
                            t = cs*V[i][j] + sn*V[i][p - 1];
                            V[i][p - 1] = -sn*V[i][j] + cs*V[i][p - 1];
                            V[i][j] = t;
                        }
				}
		    }
			break;

			// split at negligible s(k)
            case 2: {
                double f = e[k - 1];
                e[k - 1] = 0;

                for (j = k; j<p; ++j) {
                    double t = hypot(S[j], f);
                    double cs = S[j] / t;
                    double sn = f / t;
                    S[j] = t;
                    f = -sn * e[j];
                    e[j] = cs * e[j];

                    if (wantu)
                        for (i = 0; i<m; ++i) {
                            t = cs*U[i][j] + sn*U[i][k - 1];
                            U[i][k - 1] = -sn*U[i][j] + cs*U[i][k - 1];
                            U[i][j] = t;
                        }
				}
		    }
			break;

			// perform one qr step
		    case 3: {
                // calculate the shift
                double scale = max(max(max(max(
                    abs(S[p - 1]), abs(S[p - 2])), abs(e[p - 2])),
                    abs(S[k])), abs(e[k]));
                double sp = S[p - 1] / scale;
                double spm1 = S[p - 2] / scale;
                double epm1 = e[p - 2] / scale;
                double sk = S[k] / scale;
                double ek = e[k] / scale;
                double b = ((spm1 + sp)*(spm1 - sp) + epm1*epm1) / 2.0;
                double c = (sp*epm1) * (sp*epm1);
                double shift = 0;

				if ((b != 0) || (c != 0)) {
                    shift = sqrt(b*b + c);
                    if (b < 0)
                        shift = -shift;
                    shift = c / (b + shift);
                }
                double f = (sk + sp)*(sk - sp) + shift;
                double g = sk * ek;

                // chase zeros
                for (j = k; j<p - 1; ++j) {
                    double t = hypot(f, g);
                    double cs = f / t;
                    double sn = g / t;
                    if (j != k)
                        e[j - 1] = t;

                    f = cs*S[j] + sn*e[j];
                    e[j] = cs*e[j] - sn*S[j];
                    g = sn * S[j + 1];
                    S[j + 1] = cs * S[j + 1];

                    if (wantv)
                        for (i = 0; i<n; ++i) {
                            t = cs*V[i][j] + sn*V[i][j + 1];
                            V[i][j + 1] = -sn*V[i][j] + cs*V[i][j + 1];
                            V[i][j] = t;
                        }

                    t = hypot(f, g);
                    cs = f / t;
                    sn = g / t;
                    S[j] = t;
                    f = cs*e[j] + sn*S[j + 1];
                    S[j + 1] = -sn*e[j] + cs*S[j + 1];
                    g = sn * e[j + 1];
                    e[j + 1] = cs * e[j + 1];

                    if (wantu && (j < m - 1))
                    for (i = 0; i<m; ++i) {
                        t = cs*U[i][j] + sn*U[i][j + 1];
                        U[i][j + 1] = -sn*U[i][j] + cs*U[i][j + 1];
                        U[i][j] = t;
                    }
                }
                e[p - 2] = f;
                iter = iter + 1;
		    }
			break;

			// convergence
		    case 4: {
                // Make the singular values positive.
                if (S[k] <= 0) {
                    S[k] = (S[k] < 0) ? -S[k] : 0;
                    if (wantv)
                        for (i = 0; i <= pp; ++i)
                            V[i][k] = -V[i][k];
                }

                // Order the singular values.
                while (k < pp) {
                    if (S[k] >= S[k + 1])
                        break;

                    double t = S[k];
                    S[k] = S[k + 1];
                    S[k + 1] = t;

                    if (wantv && (k < n - 1))
                        for (i = 0; i<n; ++i) {
                            double t = V[i][k + 1];
                            V[i][k + 1] = V[i][k];
                            V[i][k] = t;
                        }

                    if (wantu && (k < m - 1))
                        for (i = 0; i<m; ++i) {
                            double t = U[i][k];
                            U[i][k] = U[i][k + 1];
                            U[i][k + 1] = t;
                        }
                        
                    k++;
                }
                iter = 0;
                p--;
		    }
			break;
		}
	}

    free_vector(n, e);
    free_vector(m, work);
}

__device__ void dec(int m, int n, int p, double **A, double** U, double* S, double** V) {	
	if (m >= n) {
        // Matrix<double> B(A);
        double** B;
        malloc_matrix(m, n, &B);

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                B[i][j] = A[i][j];
            }
        }

		decomposition(m, n, B, U, S, V);

        free_matrix(m, n, B);
	} else {
        // Matrix<double> B(trT(A));
        double** B;
        malloc_matrix(n, m, &B);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                B[i][j] = A[j][i];
            }
        }
		
		decomposition(n, m, B, V, S, U);

        free_matrix(n, m, B);
	}
}

__device__ void solve(int m, int n, int nx, double **A, double **B, double** X) {
    int p = min(m, n);

    // U = Matrix<double>(m, p);
	// V = Matrix<double>(n, p);
	// S = Vector<double>(p);
    double** U;
    double** V;
    double* S;
    malloc_matrix(m, p, &U);
    malloc_matrix(n, p, &V);
    malloc_vector(p, &S);

    dec(m, n, p, A, U, S, V);

	double** inv_SV;
    malloc_matrix(p, p, &inv_SV);

    for (int i = 0; i<p; ++i)
        for (int j = 0; j<p; ++j)
            inv_SV[i][j] = 0.0;

	for (int i = 0; i<p; ++i)
		inv_SV[i][i] = 1.0/S[i];

    free_vector(p, S);

    double** UT;
    malloc_matrix(p, m, &UT);
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < m; j++) {
            UT[i][j] = U[j][i];
        }
    }

    double** UTB;
    malloc_matrix(p, nx, &UTB);
    mult_matrix(p, m, nx, UT, B, UTB);
    free_matrix(p, m, UT);

    double** invSVUTB;
    malloc_matrix(p, nx, &invSVUTB);
    mult_matrix(p, p, nx, inv_SV, UTB, invSVUTB);
    free_matrix(p, nx, UTB);
    free_matrix(p, p, inv_SV);

    mult_matrix(n, p, nx, V, invSVUTB, X);

    free_matrix(p, nx, invSVUTB);
    free_matrix(m, p, U);
    free_matrix(n, p, V);
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