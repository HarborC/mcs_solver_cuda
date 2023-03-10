#include "evd_cuda.h"
#include "utils_cuda.h"

// #include <iostream>
// #include <cmath>
// #include <vector>
// using namespace std;

namespace evd {

__device__ void tred2(const int n, double* V, double* d, double* e) {
	for (int j = 0; j < n; ++j)
		d[j] = V[(n - 1)*n+j];

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
				d[j] = V[(i - 1)*n+j];
				V[i*n+j] = 0;
				V[j*n+i] = 0;
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
				V[j*n+i] = f;
				g = e[j] + V[j*n+j] * f;

				for (int k = j + 1; k <= i - 1; ++k) {
					g += V[k*n+j] * d[k];
					e[k] += V[k*n+j] * f;
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
					V[k*n+j] -= (f*e[k] + g*d[k]);

				d[j] = V[(i - 1)*n+j];
				V[i*n+j] = 0;
			}
		}
		d[i] = h;
	}

	// accumulate transformations
	for (int i = 0; i<n - 1; i++) {
		V[(n - 1)*n+i] = V[i*n+i];
		V[i*n+i] = 1;
		double h = d[i + 1];

		if (h != 0) {
			for (int k = 0; k <= i; ++k)
				d[k] = V[k*n+i + 1] / h;

			for (int j = 0; j <= i; ++j) {
				double g = 0;
				for (int k = 0; k <= i; ++k)
					g += V[k*n+i + 1] * V[k*n+j];

				for (int k = 0; k <= i; ++k)
					V[k*n+j] -= g * d[k];
			}
		}

		for (int k = 0; k <= i; ++k)
			V[k*n+i + 1] = 0;
	}

	for (int j = 0; j<n; ++j) {
		d[j] = V[(n - 1)*n+j];
		V[(n - 1)*n+j] = 0;
	}

	V[(n - 1)*n+n - 1] = 1;
	e[0] = 0;
}

__device__ void tql2(const int n, double* V, double* d, double* e) {
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
						h = V[k*n+i + 1];
						V[k*n+i + 1] = s * V[k*n+i] + c * h;
						V[k*n+i] = c * V[k*n+i] - s * h;
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
                double t = V[j*n+k];
                V[j*n+i] = V[j*n+k];
                V[j*n+k] = t;
            }
		}
	}
}

__device__ void others(const int n, double* V, double *H, double *ort) {
	int low = 0;
	int high = n - 1;

	for (int m = low + 1; m <= high - 1; ++m) {
		// scale column.
		double scale = 0;
		for (int i = m; i <= high; ++i)
			scale += abs(H[i*n+m - 1]);

		if (scale != 0) {
			// compute Householder transformation.
			double h = 0;
			for (int i = high; i >= m; --i) {
				ort[i] = H[i*n+m - 1] / scale;
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
					f += ort[i] * H[i*n+j];
				f = f / h;

				for (int i = m; i <= high; ++i)
					H[i*n+j] -= f*ort[i];
			}

			for (int i = 0; i <= high; ++i) {
				double f = 0;
				for (int j = high; j >= m; --j)
					f += ort[j] * H[i*n+j];
				f = f / h;

				for (int j = m; j <= high; ++j)
					H[i*n+j] -= f*ort[j];
			}

			ort[m] = scale * ort[m];
			H[m*n+m - 1] = scale * g;
		}
	}

	// accumulate transformations (Algol's ortran)
	for (int i = 0; i<n; ++i)
        for (int j = 0; j<n; ++j)
            V[i*n+j] = (i == j) ? 1 : 0;

	for (int m = high - 1; m >= low + 1; --m)
        if (H[m*n+m - 1] != 0) {
            for (int i = m + 1; i <= high; ++i)
                ort[i] = H[i*n+m - 1];

            for (int j = m; j <= high; ++j) {
                double g = 0;
                for (int i = m; i <= high; ++i)
                    g += ort[i] * V[i*n+j];

                // double division avoids possible underflow
                g = (g / ort[m]) / H[m*n+m - 1];
                for (int i = m; i <= high; ++i)
                    V[i*n+j] += g * ort[i];
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

__device__ void hqr2(const int n1, double* V, double* d, double* e, double *H, double *ort) {
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
			d[i] = H[i*n1+i];
			e[i] = 0;
		}

		for (int j = max(i - 1, 0); j<nn; ++j)
			norm += abs(H[i*n1+j]);
	}

	// outer loop over eigenvalue index
	int iter = 0;
	while (n >= low) {
		// Look for single small sub-diagonal element.
		int l = n;
		while (l > low) {
			s = abs(H[(l - 1)*n1+l - 1]) + abs(H[l*n1+l]);
			if (s == 0)
				s = norm;

			if (abs(H[l*n1+l - 1]) < eps*s)
				break;

			l--;
		}

		// one root found
		if (l == n) {
			H[n*n1+n] = H[n*n1+n] + exshift;
			d[n] = H[n*n1+n];
			e[n] = 0;
			n--;
			iter = 0;
		}
		// two roots found
		else if (l == n - 1) {
			w = H[n*n1+n - 1] * H[(n - 1)*n1+n];
			p = (H[(n - 1)*n1+n - 1] - H[n*n1+n]) / 2.0;
			q = p * p + w;
			z = sqrt(abs(q));
			H[n*n1+n] = H[n*n1+n] + exshift;
			H[(n - 1)*n1+n - 1] = H[(n - 1)*n1+n - 1] + exshift;
			x = H[n*n1+n];

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
				x = H[n*n1+n - 1];
				s = abs(x) + abs(z);
				p = x / s;
				q = z / s;
				r = sqrt(p * p + q * q);
				p = p / r;
				q = q / r;

				// row modification
				for (int j = n - 1; j<nn; ++j) {
					z = H[(n - 1)*n1+j];
					H[(n - 1)*n1+j] = q * z + p * H[n*n1+j];
					H[n*n1+j] = q * H[n*n1+j] - p * z;
				}

				// column modification
				for (int i = 0; i <= n; ++i) {
					z = H[i*n1+n - 1];
					H[i*n1+n - 1] = q * z + p * H[i*n1+n];
					H[i*n1+n] = q * H[i*n1+n] - p * z;
				}

				// accumulate transformations
				for (int i = low; i <= high; ++i) {
					z = V[i*n1+n - 1];
					V[i*n1+n - 1] = q * z + p * V[i*n1+n];
					V[i*n1+n] = q * V[i*n1+n] - p * z;
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
			x = H[n*n1+n];
			y = 0;
			w = 0;

			if (l < n) {
				y = H[(n - 1)*n1+n - 1];
				w = H[n*n1+n - 1] * H[(n - 1)*n1+n];
			}

			// Wilkinson's original ad hoc shift
			if (iter == 10) {
				exshift += x;
				for (int i = low; i <= n; ++i)
					H[i*n1+i] -= x;

				s = abs(H[n*n1+n - 1]) + abs(H[(n - 1)*n1+n - 2]);
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
						H[i*n1+i] -= s;

					exshift += s;
					x = y = w = 0.964;
				}
			}

			iter = iter + 1;

			// Look for two consecutive small sub-diagonal elements.
			int m = n - 2;
			while (m >= l) {
				z = H[m*n1+m];
				r = x - z;
				s = y - z;
				p = (r * s - w) / H[(m + 1)*n1+m] + H[m*n1+m + 1];
				q = H[(m + 1)*n1+m + 1] - z - r - s;
				r = H[(m + 2)*n1+m + 1];
				s = abs(p) + abs(q) + abs(r);
				p = p / s;
				q = q / s;
				r = r / s;

				if (m == l)
					break;

				if (abs(H[m*n1+m - 1]) * (abs(q) + abs(r)) <
					eps * (abs(p) * (abs(H[(m - 1)*n1+m - 1]) + abs(z) +
					abs(H[(m + 1)*n1+m + 1]))))
					break;

				m--;
			}

			for (int i = m + 2; i <= n; ++i) {
				H[i*n1+i - 2] = 0;
				if (i > m + 2)
					H[i*n1+i - 3] = 0;
			}

			// double QR step involving rows l:n and columns m:n
			for (int k = m; k <= n - 1; ++k) {
				int notlast = (k != n - 1);
				if (k != m) {
					p = H[k*n1+k - 1];
					q = H[(k + 1)*n1+k - 1];
					r = (notlast ? H[(k + 2)*n1+k - 1] : 0);
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
						H[k*n1+k - 1] = -s * x;
					else if (l != m)
						H[k*n1+k - 1] = -H[k*n1+k - 1];

					p = p + s;
					x = p / s;
					y = q / s;
					z = r / s;
					q = q / p;
					r = r / p;

					// row modification
					for (int j = k; j<nn; ++j) {
						p = H[k*n1+j] + q * H[(k + 1)*n1+j];
						if (notlast) {
							p = p + r * H[(k + 2)*n1+j];
							H[(k + 2)*n1+j] = H[(k + 2)*n1+j] - p * z;
						}

						H[k*n1+j] = H[k*n1+j] - p * x;
						H[(k + 1)*n1+j] = H[(k + 1)*n1+j] - p * y;
					}

					// column modification
					for (int i = 0; i <= min(n, k + 3); ++i) {
						p = x * H[i*n1+k] + y * H[i*n1+k + 1];
						if (notlast) {
							p = p + z * H[i*n1+k + 2];
							H[i*n1+k + 2] = H[i*n1+k + 2] - p * r;
						}
						H[i*n1+k] = H[i*n1+k] - p;
						H[i*n1+k + 1] = H[i*n1+k + 1] - p * q;
					}

					// accumulate transformations
					for (int i = low; i <= high; ++i) {
						p = x * V[i*n1+k] + y * V[i*n1+k + 1];
						if (notlast)
						{
							p = p + z * V[i*n1+k + 2];
							V[i*n1+k + 2] = V[i*n1+k + 2] - p * r;
						}
						V[i*n1+k] = V[i*n1+k] - p;
						V[i*n1+k + 1] = V[i*n1+k + 1] - p * q;
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
			H[n*n1+n] = 1;
			for (int i = n - 1; i >= 0; --i) {
				w = H[i*n1+i] - p;
				r = 0;
				for (int j = l; j <= n; ++j)
					r = r + H[i*n1+j] * H[j*n1+n];

				if (e[i] < 0) {
					z = w;
					s = r;
				} else {
					l = i;
					if (e[i] == 0) {
						if (w != 0)
							H[i*n1+n] = -r / w;
						else
							H[i*n1+n] = -r / (eps * norm);
					}
					// solve real equations
					else {
						x = H[i*n1+i + 1];
						y = H[(i + 1)*n1+i];
						q = (d[i] - p) * (d[i] - p) + e[i] * e[i];
						t = (x * s - z * r) / q;
						H[i*n1+n] = t;

						if (abs(x) > abs(z))
							H[(i + 1)*n1+n] = (-r - w * t) / x;
						else
							H[(i + 1)*n1+n] = (-s - y * t) / z;
					}

					// overflow control
					t = abs(H[i*n1+n]);
					if ((eps*t)*t > 1)
						for (int j = i; j <= n; ++j)
							H[j*n1+n] = H[j*n1+n] / t;
				}
			}
		}
		// complex vector
		else if (q < 0) {
			int l = n - 1;

			// last vector component imaginary so matrix is triangular
			if (abs(H[n*n1+n - 1]) > abs(H[(n - 1)*n1+n])) {
				H[(n - 1)*n1+n - 1] = q / H[n*n1+n - 1];
				H[(n - 1)*n1+n] = -(H[n*n1+n] - p) / H[n*n1+n - 1];
			} else {
				cdiv(0, -H[(n - 1)*n1+n], H[(n - 1)*n1+n - 1] - p, q, &cdivr, &cdivi);
				H[(n - 1)*n1+n - 1] = cdivr;
				H[(n - 1)*n1+n] = cdivi;
			}

			H[n*n1+n - 1] = 0;
			H[n*n1+n] = 1;
			for (int i = n - 2; i >= 0; --i) {
				double ra, sa, vr, vi;
				ra = 0;
				sa = 0;
				for (int j = l; j <= n; ++j) {
					ra = ra + H[i*n1+j] * H[j*n1+n - 1];
					sa = sa + H[i*n1+j] * H[j*n1+n];
				}
				w = H[i*n1+i] - p;

				if (e[i] < 0) {
					z = w;
					r = ra;
					s = sa;
				} else {
					l = i;
					if (e[i] == 0) {
						cdiv(-ra, -sa, w, q, &cdivr, &cdivi);
						H[i*n1+n - 1] = cdivr;
						H[i*n1+n] = cdivi;
					} else {
						// solve complex equations
						x = H[i*n1+i + 1];
						y = H[(i + 1)*n1+i];
						vr = (d[i] - p) * (d[i] - p) + e[i] * e[i] - q*q;
						vi = (d[i] - p) * 2.0 * q;
						if ((vr == 0) && (vi == 0))
							vr = eps * norm * (abs(w) + abs(q) +
							abs(x) + abs(y) + abs(z));

						cdiv(x*r - z*ra + q*sa, x*s - z*sa - q*ra, vr, vi, &cdivr, &cdivi);
						H[i*n1+n - 1] = cdivr;
						H[i*n1+n] = cdivi;

						if (abs(x) > (abs(z) + abs(q))) {
							H[(i + 1)*n1+n - 1] = (-ra - w*H[i*n1+n - 1] + q*H[i*n1+n]) / x;
							H[(i + 1)*n1+n] = (-sa - w*H[i*n1+n] - q*H[i*n1+n - 1]) / x;
						} else {
							cdiv(-r - y*H[i*n1+n - 1], -s - y*H[i*n1+n], z, q, &cdivr, &cdivi);
							H[(i + 1)*n1+n - 1] = cdivr;
							H[(i + 1)*n1+n] = cdivi;
						}
					}

					// overflow control
					t = max(abs(H[i*n1+n - 1]), abs(H[i*n1+n]));
					if ((eps*t)*t > 1)
						for (int j = i; j <= n; ++j) {
							H[j*n1+n - 1] = H[j*n1+n - 1] / t;
							H[j*n1+n] = H[j*n1+n] / t;
						}
				}
			}
		}
	}

	// vectors of isolated roots
	for (int i = 0; i<nn; ++i)
		if ((i < low) || (i > high))
			for (int j = i; j<nn; ++j)
				V[i*n1+j] = H[i*n1+j];

	// Back transformation to get eigenvectors of original matrix.
	for (int j = nn - 1; j >= low; --j)
		for (int i = low; i <= high; ++i) {
			z = 0;
			for (int k = low; k <= min(j, high); ++k)
				z += V[i*n1+k] * H[k*n1+j];

			V[i*n1+j] = z;
		}
}

__device__ void getRealV(const int n, double* V, double* d, double* cV) {
	// double cV[n*n];
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cV[i*n+j] = V[i*n+j];
		}
	}

	int col = 0;
	while (col < n - 1) {
		// eigenvalues d[col] and d[col+1] are complex
		if (d[col] == d[col + 1]) {
			for (int i = 0; i<n; ++i) {
				cV[i*n+col] = V[i*n+col];
				cV[i*n+col + 1] = cV[i*n+col];
			}
			col += 2;
		}
		// eigenvalue d[col] is real
		else {
			for (int i = 0; i<n; ++i)
				cV[i*n+col] = V[i*n+col];
			col += 1;
		}
	}

	// eigenvalue d[n-1] is real
	if (col == n - 1) {
		for (int i = 0; i<n; ++i)
			cV[i*n+col] = V[i*n+col];
		col += 1;
	}

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			V[i*n+j] = cV[i*n+j];
		}
	}
}

__device__ void dec(const int n, double* A, double* V, double* d, double* e, double *H, double*ort, double* cV) {

    // double e[n];

	bool symmetric = true;
	for (int j = 0; (j<n) && symmetric; ++j)
        for (int i = 0; (i<n) && symmetric; ++i)
            symmetric = (A[i*n+j] == A[j*n+i]);

	if (symmetric) {
		for (int i = 0; i<n; ++i)
            for (int j = 0; j<n; ++j)
                V[i*n+j] = A[i*n+j];

		// tridiagonalize.
		tred2(n, V, d, e);

		// diagonalize.
		tql2(n, V, d, e);
	} else {
		// double H[n*n]; // = Matrix<double>(n, n);
		// double ort[n]; // = Vector<double>(n);

		for (int j = 0; j<n; ++j)
            for (int i = 0; i<n; ++i)
                H[i*n+j] = A[i*n+j];

		// reduce to Hessenberg form
		others(n, V, H, ort);
		
		// reduce Hessenberg to real Schur form
		hqr2(n, V, d, e, H, ort);
	}

	getRealV(n, V, d, cV);
}

}