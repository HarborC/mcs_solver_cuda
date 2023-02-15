#include "evd_cuda.h"
#include "utils_cuda.h"

// #include <iostream>
// #include <cmath>
// #include <vector>
// using namespace std;

namespace evd {

__device__ void tred2(int n, double** V, double* d, double* e) {
	for (int j = 0; j < n; ++j)
		d[j] = V[n - 1][j];

	// Householder reduction to tridiagonal form
	for (int i = n - 1; i>0; --i) {
		// scale to avoid under/overflow
		double scale = 0;
		double h = 0;

		for (int k = 0; k<i; ++k)
			scale += abs(d[k]);

		if (scale == 0) {
			e[i] = d[i - 1];
			for (int j = 0; j<i; ++j) {
				d[j] = V[i - 1][j];
				V[i][j] = 0;
				V[j][i] = 0;
			}
		} else {
			// generate Householder vector
			for (int k = 0; k<i; ++k) {
				d[k] /= scale;
				h += d[k] * d[k];
			}

			double f = d[i - 1];
			double g = sqrt(h);
			if (f > 0)
				g = -g;

			e[i] = scale * g;
			h = h - f * g;
			d[i - 1] = f - g;

			for (int j = 0; j<i; ++j)
				e[j] = 0;

			// Apply similarity transformation to remaining columns.
			for (int j = 0; j<i; ++j) {
				f = d[j];
				V[j][i] = f;
				g = e[j] + V[j][j] * f;

				for (int k = j + 1; k <= i - 1; ++k) {
					g += V[k][j] * d[k];
					e[k] += V[k][j] * f;
				}
				e[j] = g;
			}

			f = 0;
			for (int j = 0; j<i; ++j) {
				e[j] /= h;
				f += e[j] * d[j];
			}

			double hh = f / (h + h);
			for (int j = 0; j<i; ++j)
				e[j] -= hh * d[j];

			for (int j = 0; j<i; ++j) {
				f = d[j];
				g = e[j];
				for (int k = j; k <= i - 1; ++k)
					V[k][j] -= (f*e[k] + g*d[k]);

				d[j] = V[i - 1][j];
				V[i][j] = 0;
			}
		}
		d[i] = h;
	}

	// accumulate transformations
	for (int i = 0; i<n - 1; i++) {
		V[n - 1][i] = V[i][i];
		V[i][i] = 1;
		double h = d[i + 1];

		if (h != 0) {
			for (int k = 0; k <= i; ++k)
				d[k] = V[k][i + 1] / h;

			for (int j = 0; j <= i; ++j) {
				double g = 0;
				for (int k = 0; k <= i; ++k)
					g += V[k][i + 1] * V[k][j];

				for (int k = 0; k <= i; ++k)
					V[k][j] -= g * d[k];
			}
		}

		for (int k = 0; k <= i; ++k)
			V[k][i + 1] = 0;
	}

	for (int j = 0; j<n; ++j) {
		d[j] = V[n - 1][j];
		V[n - 1][j] = 0;
	}

	V[n - 1][n - 1] = 1;
	e[0] = 0;
}

__device__ void tql2(int n, double** V, double* d, double* e) {
	for (int i = 1; i<n; ++i)
		e[i - 1] = e[i];
	e[n - 1] = 0;

	double f = 0;
	double tst1 = 0;
	double eps = pow(2.0, -52.0);

	for (int l = 0; l<n; ++l) {
		// find small subdiagonal element
		tst1 = max(tst1, abs(d[l]) + abs(e[l]));
		int m = l;

		// original while-loop from Java code
		while (m < n) {
			if (abs(e[m]) <= eps*tst1)
				break;
			m++;
		}

		// if m == l, d[l] is an eigenvalue, otherwise, iterate
		if (m > l) {
			int iter = 0;
			do {
				iter = iter + 1;

				// compute implicit shift
				double g = d[l];
				double p = (d[l + 1] - g) / (2.0 * e[l]);
				double r = hypot(p, 1.0);
				if (p < 0)
					r = -r;

				d[l] = e[l] / (p + r);
				d[l + 1] = e[l] * (p + r);
				double dl1 = d[l + 1];
				double h = g - d[l];

				for (int i = l + 2; i<n; ++i)
					d[i] -= h;
				f += h;

				// implicit QL transformation.
				p = d[m];
				double c = 1;
				double c2 = c;
				double c3 = c;
				double el1 = e[l + 1];
				double s = 0;
				double s2 = 0;

				for (int i = m - 1; i >= l; --i) {
					c3 = c2;
					c2 = c;
					s2 = s;
					g = c * e[i];
					h = c * p;
					r = hypot(p, e[i]);
					e[i + 1] = s * r;
					s = e[i] / r;
					c = p / r;
					p = c * d[i] - s * g;
					d[i + 1] = h + s * (c * g + s * d[i]);

					// accumulate transformation.
					for (int k = 0; k<n; ++k) {
						h = V[k][i + 1];
						V[k][i + 1] = s * V[k][i] + c * h;
						V[k][i] = c * V[k][i] - s * h;
					}
				}

				p = -s * s2 * c3 * el1 * e[l] / dl1;
				e[l] = s * p;
				d[l] = c * p;

			} while (abs(e[l]) > eps*tst1);
		}

		d[l] += f;
		e[l] = 0;
	}

	// Sort eigenvalues and corresponding vectors.
	for (int i = 0; i<n - 1; ++i) {
		int k = i;
		double p = d[i];

		for (int j = i + 1; j<n; ++j)
		if (d[j] < p) {
			k = j;
			p = d[j];
		}

		if (k != i) {
			d[k] = d[i];
			d[i] = p;
			for (int j = 0; j<n; ++j) {
                double t = V[j][k];
                V[j][i] = V[j][k];
                V[j][k] = t;
            }
		}
	}
}

__device__ void others(int n, double** V, double **H, double *ort) {
	int low = 0;
	int high = n - 1;

	for (int m = low + 1; m <= high - 1; ++m) {
		// scale column.
		double scale = 0;
		for (int i = m; i <= high; ++i)
			scale += abs(H[i][m - 1]);

		if (scale != 0) {
			// compute Householder transformation.
			double h = 0;
			for (int i = high; i >= m; --i) {
				ort[i] = H[i][m - 1] / scale;
				h += ort[i] * ort[i];
			}

			double g = sqrt(h);
			if (ort[m] > 0)
				g = -g;

			h = h - ort[m] * g;
			ort[m] = ort[m] - g;

			// Apply Householder similarity transformation.
			for (int j = m; j<n; ++j) {
				double f = 0;
				for (int i = high; i >= m; --i)
					f += ort[i] * H[i][j];
				f = f / h;

				for (int i = m; i <= high; ++i)
					H[i][j] -= f*ort[i];
			}

			for (int i = 0; i <= high; ++i) {
				double f = 0;
				for (int j = high; j >= m; --j)
					f += ort[j] * H[i][j];
				f = f / h;

				for (int j = m; j <= high; ++j)
					H[i][j] -= f*ort[j];
			}

			ort[m] = scale * ort[m];
			H[m][m - 1] = scale * g;
		}
	}

	// accumulate transformations (Algol's ortran)
	for (int i = 0; i<n; ++i)
        for (int j = 0; j<n; ++j)
            V[i][j] = (i == j) ? 1 : 0;

	for (int m = high - 1; m >= low + 1; --m)
        if (H[m][m - 1] != 0) {
            for (int i = m + 1; i <= high; ++i)
                ort[i] = H[i][m - 1];

            for (int j = m; j <= high; ++j) {
                double g = 0;
                for (int i = m; i <= high; ++i)
                    g += ort[i] * V[i][j];

                // double division avoids possible underflow
                g = (g / ort[m]) / H[m][m - 1];
                for (int i = m; i <= high; ++i)
                    V[i][j] += g * ort[i];
            }
        }
}

__device__ void cdiv(double xr, double xi, double yr, double yi, double* cdivr, double* cdivi) {
	double r, d;
	if (abs(yr) > abs(yi)) {
		r = yi / yr;
		d = yr + r*yi;
		*cdivr = (xr + r*xi) / d;
		*cdivi = (xi - r*xr) / d;
	} else {
		r = yr / yi;
		d = yi + r*yr;
		*cdivr = (r*xr + xi) / d;
		*cdivi = (r*xi - xr) / d;
	}
}

__device__ void hqr2(int n1, double** V, double* d, double* e, double **H, double *ort) {
    double cdivr, cdivi;

	// initialize
	int nn = n1;
	int n = nn - 1;
	int low = 0;
	int high = nn - 1;
	double eps = pow(2.0, -52.0);
	double exshift = 0;
	double p = 0, q = 0, r = 0, s = 0, z = 0, t, w, x, y;

	// Store roots isolated by balanc and compute matrix norm.
	double norm = 0;
	for (int i = 0; i<nn; ++i) {
		if ((i < low) || (i > high)) {
			d[i] = H[i][i];
			e[i] = 0;
		}

		for (int j = max(i - 1, 0); j<nn; ++j)
			norm += abs(H[i][j]);
	}

	// outer loop over eigenvalue index
	int iter = 0;
	while (n >= low) {
		// Look for single small sub-diagonal element.
		int l = n;
		while (l > low) {
			s = abs(H[l - 1][l - 1]) + abs(H[l][l]);
			if (s == 0)
				s = norm;

			if (abs(H[l][l - 1]) < eps*s)
				break;

			l--;
		}

		// one root found
		if (l == n) {
			H[n][n] = H[n][n] + exshift;
			d[n] = H[n][n];
			e[n] = 0;
			n--;
			iter = 0;
		}
		// two roots found
		else if (l == n - 1) {
			w = H[n][n - 1] * H[n - 1][n];
			p = (H[n - 1][n - 1] - H[n][n]) / 2.0;
			q = p * p + w;
			z = sqrt(abs(q));
			H[n][n] = H[n][n] + exshift;
			H[n - 1][n - 1] = H[n - 1][n - 1] + exshift;
			x = H[n][n];

			// real pair
			if (q >= 0) {
				if (p >= 0)
					z = p + z;
				else
					z = p - z;

				d[n - 1] = x + z;
				d[n] = d[n - 1];
				if (z != 0)
					d[n] = x - w / z;

				e[n - 1] = 0;
				e[n] = 0;
				x = H[n][n - 1];
				s = abs(x) + abs(z);
				p = x / s;
				q = z / s;
				r = sqrt(p * p + q * q);
				p = p / r;
				q = q / r;

				// row modification
				for (int j = n - 1; j<nn; ++j) {
					z = H[n - 1][j];
					H[n - 1][j] = q * z + p * H[n][j];
					H[n][j] = q * H[n][j] - p * z;
				}

				// column modification
				for (int i = 0; i <= n; ++i) {
					z = H[i][n - 1];
					H[i][n - 1] = q * z + p * H[i][n];
					H[i][n] = q * H[i][n] - p * z;
				}

				// accumulate transformations
				for (int i = low; i <= high; ++i) {
					z = V[i][n - 1];
					V[i][n - 1] = q * z + p * V[i][n];
					V[i][n] = q * V[i][n] - p * z;
				}
			}
			// complex pair
			else {
				d[n - 1] = x + p;
				d[n] = x + p;
				e[n - 1] = z;
				e[n] = -z;
			}

			n = n - 2;
			iter = 0;
		} else {

			// form shift
			x = H[n][n];
			y = 0;
			w = 0;

			if (l < n) {
				y = H[n - 1][n - 1];
				w = H[n][n - 1] * H[n - 1][n];
			}

			// Wilkinson's original ad hoc shift
			if (iter == 10) {
				exshift += x;
				for (int i = low; i <= n; ++i)
					H[i][i] -= x;

				s = abs(H[n][n - 1]) + abs(H[n - 1][n - 2]);
				x = y = 0.75 * s;
				w = -0.4375 * s * s;
			}

			// MATLAB's new ad hoc shift
			if (iter == 30) {
				s = (y - x) / 2.0;
				s = s * s + w;
				if (s > 0) {
					s = sqrt(s);
					if (y < x)
						s = -s;

					s = x - w / ((y - x) / 2.0 + s);
					for (int i = low; i <= n; ++i)
						H[i][i] -= s;

					exshift += s;
					x = y = w = 0.964;
				}
			}

			iter = iter + 1;

			// Look for two consecutive small sub-diagonal elements.
			int m = n - 2;
			while (m >= l) {
				z = H[m][m];
				r = x - z;
				s = y - z;
				p = (r * s - w) / H[m + 1][m] + H[m][m + 1];
				q = H[m + 1][m + 1] - z - r - s;
				r = H[m + 2][m + 1];
				s = abs(p) + abs(q) + abs(r);
				p = p / s;
				q = q / s;
				r = r / s;

				if (m == l)
					break;

				if (abs(H[m][m - 1]) * (abs(q) + abs(r)) <
					eps * (abs(p) * (abs(H[m - 1][m - 1]) + abs(z) +
					abs(H[m + 1][m + 1]))))
					break;

				m--;
			}

			for (int i = m + 2; i <= n; ++i) {
				H[i][i - 2] = 0;
				if (i > m + 2)
					H[i][i - 3] = 0;
			}

			// double QR step involving rows l:n and columns m:n
			for (int k = m; k <= n - 1; ++k) {
				int notlast = (k != n - 1);
				if (k != m) {
					p = H[k][k - 1];
					q = H[k + 1][k - 1];
					r = (notlast ? H[k + 2][k - 1] : 0);
					x = abs(p) + abs(q) + abs(r);

					if (x != 0) {
						p = p / x;
						q = q / x;
						r = r / x;
					}
				}

				if (x == 0)
					break;

				s = sqrt(p * p + q * q + r * r);
				if (p < 0)
					s = -s;

				if (s != 0) {
					if (k != m)
						H[k][k - 1] = -s * x;
					else if (l != m)
						H[k][k - 1] = -H[k][k - 1];

					p = p + s;
					x = p / s;
					y = q / s;
					z = r / s;
					q = q / p;
					r = r / p;

					// row modification
					for (int j = k; j<nn; ++j) {
						p = H[k][j] + q * H[k + 1][j];
						if (notlast) {
							p = p + r * H[k + 2][j];
							H[k + 2][j] = H[k + 2][j] - p * z;
						}

						H[k][j] = H[k][j] - p * x;
						H[k + 1][j] = H[k + 1][j] - p * y;
					}

					// column modification
					for (int i = 0; i <= min(n, k + 3); ++i) {
						p = x * H[i][k] + y * H[i][k + 1];
						if (notlast) {
							p = p + z * H[i][k + 2];
							H[i][k + 2] = H[i][k + 2] - p * r;
						}
						H[i][k] = H[i][k] - p;
						H[i][k + 1] = H[i][k + 1] - p * q;
					}

					// accumulate transformations
					for (int i = low; i <= high; ++i) {
						p = x * V[i][k] + y * V[i][k + 1];
						if (notlast)
						{
							p = p + z * V[i][k + 2];
							V[i][k + 2] = V[i][k + 2] - p * r;
						}
						V[i][k] = V[i][k] - p;
						V[i][k + 1] = V[i][k + 1] - p * q;
					}
				}  // (s != 0 )
			}  // k loop
		}  // check convergence
	}  // while ( n >= low )

	// Backsubstitute to find vectors of upper triangular form.
	if (norm == 0)
		return;

	for (n = nn - 1; n >= 0; --n) {
		p = d[n];
		q = e[n];

		// real vector
		if (q == 0) {
			int l = n;
			H[n][n] = 1;
			for (int i = n - 1; i >= 0; --i) {
				w = H[i][i] - p;
				r = 0;
				for (int j = l; j <= n; ++j)
					r = r + H[i][j] * H[j][n];

				if (e[i] < 0) {
					z = w;
					s = r;
				} else {
					l = i;
					if (e[i] == 0) {
						if (w != 0)
							H[i][n] = -r / w;
						else
							H[i][n] = -r / (eps * norm);
					}
					// solve real equations
					else {
						x = H[i][i + 1];
						y = H[i + 1][i];
						q = (d[i] - p) * (d[i] - p) + e[i] * e[i];
						t = (x * s - z * r) / q;
						H[i][n] = t;

						if (abs(x) > abs(z))
							H[i + 1][n] = (-r - w * t) / x;
						else
							H[i + 1][n] = (-s - y * t) / z;
					}

					// overflow control
					t = abs(H[i][n]);
					if ((eps*t)*t > 1)
						for (int j = i; j <= n; ++j)
							H[j][n] = H[j][n] / t;
				}
			}
		}
		// complex vector
		else if (q < 0) {
			int l = n - 1;

			// last vector component imaginary so matrix is triangular
			if (abs(H[n][n - 1]) > abs(H[n - 1][n])) {
				H[n - 1][n - 1] = q / H[n][n - 1];
				H[n - 1][n] = -(H[n][n] - p) / H[n][n - 1];
			} else {
				cdiv(0, -H[n - 1][n], H[n - 1][n - 1] - p, q, &cdivr, &cdivi);
				H[n - 1][n - 1] = cdivr;
				H[n - 1][n] = cdivi;
			}

			H[n][n - 1] = 0;
			H[n][n] = 1;
			for (int i = n - 2; i >= 0; --i) {
				double ra, sa, vr, vi;
				ra = 0;
				sa = 0;
				for (int j = l; j <= n; ++j) {
					ra = ra + H[i][j] * H[j][n - 1];
					sa = sa + H[i][j] * H[j][n];
				}
				w = H[i][i] - p;

				if (e[i] < 0) {
					z = w;
					r = ra;
					s = sa;
				} else {
					l = i;
					if (e[i] == 0) {
						cdiv(-ra, -sa, w, q, &cdivr, &cdivi);
						H[i][n - 1] = cdivr;
						H[i][n] = cdivi;
					} else {
						// solve complex equations
						x = H[i][i + 1];
						y = H[i + 1][i];
						vr = (d[i] - p) * (d[i] - p) + e[i] * e[i] - q*q;
						vi = (d[i] - p) * 2.0 * q;
						if ((vr == 0) && (vi == 0))
							vr = eps * norm * (abs(w) + abs(q) +
							abs(x) + abs(y) + abs(z));

						cdiv(x*r - z*ra + q*sa, x*s - z*sa - q*ra, vr, vi, &cdivr, &cdivi);
						H[i][n - 1] = cdivr;
						H[i][n] = cdivi;

						if (abs(x) > (abs(z) + abs(q))) {
							H[i + 1][n - 1] = (-ra - w*H[i][n - 1] + q*H[i][n]) / x;
							H[i + 1][n] = (-sa - w*H[i][n] - q*H[i][n - 1]) / x;
						} else {
							cdiv(-r - y*H[i][n - 1], -s - y*H[i][n], z, q, &cdivr, &cdivi);
							H[i + 1][n - 1] = cdivr;
							H[i + 1][n] = cdivi;
						}
					}

					// overflow control
					t = max(abs(H[i][n - 1]), abs(H[i][n]));
					if ((eps*t)*t > 1)
						for (int j = i; j <= n; ++j) {
							H[j][n - 1] = H[j][n - 1] / t;
							H[j][n] = H[j][n] / t;
						}
				}
			}
		}
	}

	// vectors of isolated roots
	for (int i = 0; i<nn; ++i)
		if ((i < low) || (i > high))
			for (int j = i; j<nn; ++j)
				V[i][j] = H[i][j];

	// Back transformation to get eigenvectors of original matrix.
	for (int j = nn - 1; j >= low; --j)
		for (int i = low; i <= high; ++i) {
			z = 0;
			for (int k = low; k <= min(j, high); ++k)
				z += V[i][k] * H[k][j];

			V[i][j] = z;
		}
}

__device__ void getRealV(int n, double** V, double* d) {
	double** cV;
	malloc_matrix(n, n, &cV);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cV[i][j] = V[i][j];
		}
	}

	int col = 0;
	while (col < n - 1) {
		// eigenvalues d[col] and d[col+1] are complex
		if (d[col] == d[col + 1]) {
			for (int i = 0; i<n; ++i) {
				cV[i][col] = V[i][col];
				cV[i][col + 1] = cV[i][col];
			}
			col += 2;
		}
		// eigenvalue d[col] is real
		else {
			for (int i = 0; i<n; ++i)
				cV[i][col] = V[i][col];
			col += 1;
		}
	}

	// eigenvalue d[n-1] is real
	if (col == n - 1) {
		for (int i = 0; i<n; ++i)
			cV[i][col] = V[i][col];
		col += 1;
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			V[i][j] = cV[i][j];
		}
	}

	free_matrix(n, n, cV);
}

__device__ void dec(int n, double** A, double** V, double* d) {

	// V = Matrix<double>(n, n);
	// d = Vector<double>(n);
	// e = Vector<double>(n);
    double* e;
	malloc_vector(n, &e);

	bool symmetric = true;
	for (int j = 0; (j<n) && symmetric; ++j)
        for (int i = 0; (i<n) && symmetric; ++i)
            symmetric = (A[i][j] == A[j][i]);

	if (symmetric) {
		for (int i = 0; i<n; ++i)
            for (int j = 0; j<n; ++j)
                V[i][j] = A[i][j];

		// tridiagonalize.
		tred2(n, V, d, e);

		// diagonalize.
		tql2(n, V, d, e);
	} else {
		double **H; // = Matrix<double>(n, n);
		double* ort; // = Vector<double>(n);
		malloc_matrix(n, n, &H);
		malloc_vector(n, &ort);

		for (int j = 0; j<n; ++j)
            for (int i = 0; i<n; ++i)
                H[i][j] = A[i][j];

		// reduce to Hessenberg form
		others(n, V, H, ort);

		// print_matrix(n, n, H);
		
		// reduce Hessenberg to real Schur form
		hqr2(n, V, d, e, H, ort);

		// print_matrix(n, n, H);

		free_matrix(n, n, H);
		free_vector(n, ort);
	}

	getRealV(n, V, d);

	free_vector(n, e);
}

// __device__ void test() {
//     int n = 7;
//     double** A;
//     double** V;
//     double* d;

// 	double A_vec[] = {3,4,5,10,3,7,1,
// 					  7,9,2,6,2,4,8,
// 					  5,2,4,7,1,9,4,
// 					  12,78,32,65,31,23,16,
// 					  95,74,36,21,28,49,58,
// 					  12,18,16,17,34,52,29,
// 					  2,7,99,31,57,90,83};
// 	malloc_matrix(n, n, &A);
// 	memcopy_matrix(n, n, A, &(A_vec[0]));
// 	malloc_matrix(n, n, &V);
// 	malloc_vector(n, &d);

// 	evd::dec(n, A, V, d);

// 	print_matrix(n, n, A);
// 	print_matrix(n, n, V);
// 	print_vector(n, d);

// 	free_matrix(n, n, A);
// 	free_matrix(n, n, V);
// 	free_vector(n, d);
// }

}