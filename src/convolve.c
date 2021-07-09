/*******************************************************************************
 * This file is part of 3FA.
 * Copyright (c) 2021 Willem Elbers (whe@willemelbers.com)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <math.h>

#include "../include/convolve.h"
#include "../include/fft.h"
#include "../include/fft_kernels.h"
#include "../include/strooklat.h"

void convolve(int N, double boxlen, const double *phi, double *out,
              struct growth_factors_2 *gfac2) {

    /* Allocate grids for the Fourier transforms */
    fftw_complex *fphi = malloc(N * N * (N/2+1) * sizeof(fftw_complex));
    fftw_complex *fout = malloc(N * N * (N/2+1) * sizeof(fftw_complex));
    bzero(fout, N * N * (N/2+1) * sizeof(fftw_complex));

    boxlen = 300.;

    /* Allocate a real grid to store a copy of the input */
    double *phi_copy = malloc(N * N * N * sizeof(double));

    /* Copy the input grid, so as not to destroy it */
    memcpy(phi_copy, phi, N * N * N * sizeof(double));

    /* Fourier transform the potential */
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, phi_copy, fphi, FFTW_ESTIMATE);
    fft_execute(r2c);
    fft_normalize_r2c(fphi, N, boxlen);
    fftw_destroy_plan(r2c);

    /* We are done with the copy */
    free(phi_copy);

    /* Grid constants */
    const double dk = 2.0 * M_PI / boxlen;
    const double dk_twopi = dk / (2.0 * M_PI);
    const double dk_twopi_3 = dk_twopi * dk_twopi * dk_twopi;

    /* Allocate memory for tables of growth factors */
    int nk = gfac2->nk;
    double *D2_A_arr = malloc(nk * nk * sizeof(double));
    double *D2_B_arr = malloc(nk * nk * sizeof(double));

    /* Determine the asymptotic second order growth factor */
    const double k_cutoff = 0.25;
    double D2_A_asymp_sum = 0;
    double D2_B_asymp_sum = 0;
    int asymp_count = 0;

    for (int i=0; i<nk; i++) {
        for (int j1=0; j1<nk; j1++) {
            for (int j2=0; j2<nk; j2++) {
                double k = gfac2->k[i];
                double k1 = gfac2->k[j1];
                double k2 = gfac2->k[j2];
                if (k1 > k_cutoff && k2 > k_cutoff && k > k_cutoff) {
                    D2_A_asymp_sum += gfac2->D2_A[i * nk * nk + j1 * nk + j2];
                    D2_B_asymp_sum += gfac2->D2_B[i * nk * nk + j1 * nk + j2];
                    asymp_count++;
                }
            }
        }
    }

    double D2_A_asymp = D2_A_asymp_sum / asymp_count;
    double D2_B_asymp = D2_B_asymp_sum / asymp_count;
    double D2_asymp = 0.5 * (D2_A_asymp + D2_B_asymp);

    printf("\n\n");
    printf("Asymptotic D2_A = %g\n", D2_A_asymp);
    printf("Asymptotic D2_B = %g\n", D2_B_asymp);
    printf("Difference = %g\n", D2_B_asymp - D2_A_asymp);

    /* Create three wavenumber splines */
    struct strooklat spline_k = {gfac2->k, gfac2->nk};
    struct strooklat spline_k1 = {gfac2->k, gfac2->nk};
    struct strooklat spline_k2 = {gfac2->k, gfac2->nk};
    init_strooklat_spline(&spline_k, 100);
    init_strooklat_spline(&spline_k1, 100);
    init_strooklat_spline(&spline_k2, 100);


    /* Do the convolution */
    #pragma omp parallel for
    for (int x=0; x<N; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                /* Calculate the wavevector */
                double kx,ky,kz,k;
                fft_wavevector(x, y, z, N, dk, &kx, &ky, &kz, &k);

                /* The id of the output cell */
                const int id = row_major_half(x,y,z,N);

                if (k == 0.) continue; //skip the DC mode
                // if (k > 0.25) continue; //low-pass filter

                /* Interpolate along the |k| = |k1 + k2| direction */
                int index;
                double u;
                strooklat_find_x(&spline_k, k, &index, &u);
                for (int i = 0; i < nk * nk; i++) {
                    D2_A_arr[i] = (1 - u) * gfac2->D2_A[index * nk * nk + i]
                                      + u * gfac2->D2_A[(index + 1) * nk * nk + i];
                    D2_B_arr[i] = (1 - u) * gfac2->D2_B[index * nk * nk + i]
                                      + u * gfac2->D2_B[(index + 1) * nk * nk + i];
                }

                /* Perform the integral */
                for (int x1=0; x1<N; x1++) {
                    for (int y1=0; y1<N; y1++) {
                        for (int z1=0; z1<N; z1++) {
                            /* Calculate the first wavevector */
                            double k1x = (x1 > N/2) ? (x1 - N) * dk : x1 * dk;
                            double k1y = (y1 > N/2) ? (y1 - N) * dk : y1 * dk;
                            double k1z = (z1 > N/2) ? (z1 - N) * dk : z1 * dk;
                            double k1k1 = (k1x * k1x) + (k1y * k1y) + (k1z * k1z);

                            if (k1k1 > 0.25 * 0.25) continue;

                            /* The first and second wavevectors sum up to k */
                            int x2 = wrap(x - x1, N);
                            int y2 = wrap(y - y1, N);
                            int z2 = wrap(z - z1, N);

                            /* Compute the second wave vector */
                            double k2x = (x2 > N/2) ? (x2 - N) * dk : x2 * dk;
                            double k2y = (y2 > N/2) ? (y2 - N) * dk : y2 * dk;
                            double k2z = (z2 > N/2) ? (z2 - N) * dk : z2 * dk;
                            double k2k2 = (k2x * k2x) + (k2y * k2y) + (k2z * k2z);

                            /* Calculate the inner product */
                            double k1k2 = (k1x * k2x) + (k1y * k2y) + (k1z * k2z);

                            /* Determine the growth factors */
                            double k1 = sqrt(k1k1);
                            double k2 = sqrt(k2k2);
                            double D2_A = strooklat_interp_2d(&spline_k1, &spline_k2, D2_A_arr, k1, k2) - D2_asymp;
                            double D2_B = strooklat_interp_2d(&spline_k1, &spline_k2, D2_B_arr, k1, k2) - D2_asymp;

                            /* Compute the kernel */
                            fftw_complex K = 0.5 * (D2_A * k1k1 * k2k2 - D2_B * k1k2 * k1k2) / (k * k);

                            /* Fetch the value of phi(k1) */
                            fftw_complex ph1;
                            if (z1 <= N/2) {
                                ph1 = fphi[row_major_half(x1, y1, z1, N)];
                            } else {
                                ph1 = conj(fphi[row_major_half(N - x1, N - y1, N - z1, N)]);
                            }

                            /* Fetch the value of phi(k2) */
                            fftw_complex ph2;
                            if (z2 <= N/2) {
                                ph2 = fphi[row_major_half(x2, y2, z2, N)];
                            } else {
                                ph2 = conj(fphi[row_major_half(N - x2, N - y2, N - z2, N)]);
                            }

                            /* Add the result */
                            fout[id] += K * (ph1 * ph2) * dk_twopi_3;
                        }
                    }
                }
            }
        }
    }

    printf("Done with convolution.\n");

    /* Apply inverse Fourier transform to obtain the final result */
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fout, out, FFTW_ESTIMATE);
    fft_execute(c2r);
    fft_normalize_c2r(out, N, boxlen);
    fftw_destroy_plan(c2r);

    /* Free all the intermediate grids */
    free(fphi);
    free(fout);

    /* Free the splines */
    free_strooklat_spline(&spline_k);
    free_strooklat_spline(&spline_k1);
    free_strooklat_spline(&spline_k2);

    /* Free the memory used for the growth factor tables */
    free(D2_A_arr);
    free(D2_B_arr);
}

void convolve_fft(int N, double boxlen, const double *phi, double *out) {
    /* Allocate grids for the Fourier transforms */
    fftw_complex *fphi = malloc(N * N * (N/2 + 1) * sizeof(fftw_complex));
    fftw_complex *fphi_xx = malloc(N * N * (N/2 + 1) * sizeof(fftw_complex));
    fftw_complex *fphi_yy = malloc(N * N * (N/2 + 1) * sizeof(fftw_complex));
    fftw_complex *fphi_zz = malloc(N * N * (N/2 + 1) * sizeof(fftw_complex));
    fftw_complex *fphi_xy = malloc(N * N * (N/2 + 1) * sizeof(fftw_complex));
    fftw_complex *fphi_xz = malloc(N * N * (N/2 + 1) * sizeof(fftw_complex));
    fftw_complex *fphi_yz = malloc(N * N * (N/2 + 1) * sizeof(fftw_complex));

    /* Allocate real grids */
    double *phi_copy = malloc(N * N * N * sizeof(double));
    double *phi_xx = malloc(N * N * N * sizeof(double));
    double *phi_yy = malloc(N * N * N * sizeof(double));
    double *phi_zz = malloc(N * N * N * sizeof(double));
    double *phi_xy = malloc(N * N * N * sizeof(double));
    double *phi_xz = malloc(N * N * N * sizeof(double));
    double *phi_yz = malloc(N * N * N * sizeof(double));

    /* Copy the input grid, so as not to destroy it */
    memcpy(phi_copy, phi, N * N * N * sizeof(double));

    /* Fourier transform the potential */
    fftw_plan r2c = fftw_plan_dft_r2c_3d(N, N, N, phi_copy, fphi, FFTW_ESTIMATE);
    fft_execute(r2c);
    fft_normalize_r2c(fphi, N, boxlen);
    fftw_destroy_plan(r2c);

    /* Copy into the derivative grids */
    memcpy(fphi_xx, fphi, N * N * (N/2 + 1) * sizeof(fftw_complex));
    memcpy(fphi_yy, fphi, N * N * (N/2 + 1) * sizeof(fftw_complex));
    memcpy(fphi_zz, fphi, N * N * (N/2 + 1) * sizeof(fftw_complex));
    memcpy(fphi_xy, fphi, N * N * (N/2 + 1) * sizeof(fftw_complex));
    memcpy(fphi_xz, fphi, N * N * (N/2 + 1) * sizeof(fftw_complex));
    memcpy(fphi_yz, fphi, N * N * (N/2 + 1) * sizeof(fftw_complex));

    /* Apply the derivative kernels */
    fft_apply_kernel(fphi_xx, fphi_xx, N, boxlen, kernel_dx, NULL);
    fft_apply_kernel(fphi_xx, fphi_xx, N, boxlen, kernel_dx, NULL);
    fft_apply_kernel(fphi_yy, fphi_yy, N, boxlen, kernel_dy, NULL);
    fft_apply_kernel(fphi_yy, fphi_yy, N, boxlen, kernel_dy, NULL);
    fft_apply_kernel(fphi_zz, fphi_zz, N, boxlen, kernel_dz, NULL);
    fft_apply_kernel(fphi_zz, fphi_zz, N, boxlen, kernel_dz, NULL);
    fft_apply_kernel(fphi_xy, fphi_xy, N, boxlen, kernel_dx, NULL);
    fft_apply_kernel(fphi_xy, fphi_xy, N, boxlen, kernel_dy, NULL);
    fft_apply_kernel(fphi_xz, fphi_xz, N, boxlen, kernel_dx, NULL);
    fft_apply_kernel(fphi_xz, fphi_xz, N, boxlen, kernel_dz, NULL);
    fft_apply_kernel(fphi_yz, fphi_yz, N, boxlen, kernel_dy, NULL);
    fft_apply_kernel(fphi_yz, fphi_yz, N, boxlen, kernel_dz, NULL);

    /* Apply inverse Fourier transforms */
    fftw_plan c2r_xx = fftw_plan_dft_c2r_3d(N, N, N, fphi_xx, phi_xx, FFTW_ESTIMATE);
    fftw_plan c2r_yy = fftw_plan_dft_c2r_3d(N, N, N, fphi_yy, phi_yy, FFTW_ESTIMATE);
    fftw_plan c2r_zz = fftw_plan_dft_c2r_3d(N, N, N, fphi_zz, phi_zz, FFTW_ESTIMATE);
    fftw_plan c2r_xy = fftw_plan_dft_c2r_3d(N, N, N, fphi_xy, phi_xy, FFTW_ESTIMATE);
    fftw_plan c2r_xz = fftw_plan_dft_c2r_3d(N, N, N, fphi_xz, phi_xz, FFTW_ESTIMATE);
    fftw_plan c2r_yz = fftw_plan_dft_c2r_3d(N, N, N, fphi_yz, phi_yz, FFTW_ESTIMATE);
    fft_execute(c2r_xx);
    fft_execute(c2r_yy);
    fft_execute(c2r_zz);
    fft_execute(c2r_xy);
    fft_execute(c2r_xz);
    fft_execute(c2r_yz);
    fft_normalize_c2r(phi_xx, N, boxlen);
    fft_normalize_c2r(phi_yy, N, boxlen);
    fft_normalize_c2r(phi_zz, N, boxlen);
    fft_normalize_c2r(phi_xy, N, boxlen);
    fft_normalize_c2r(phi_xz, N, boxlen);
    fft_normalize_c2r(phi_yz, N, boxlen);
    fftw_destroy_plan(c2r_xx);
    fftw_destroy_plan(c2r_yy);
    fftw_destroy_plan(c2r_zz);
    fftw_destroy_plan(c2r_xy);
    fftw_destroy_plan(c2r_xz);
    fftw_destroy_plan(c2r_yz);

    /* Sum the components */
    for (int i = 0; i < N * N * N; i++) {
        out[i] = phi_xx[i] * phi_yy[i] + phi_xx[i] * phi_zz[i] +
                 phi_yy[i] * phi_zz[i] - phi_xy[i] * phi_xy[i] -
                 phi_xz[i] * phi_xz[i] - phi_yz[i] * phi_yz[i];
        /* Flip the sign */
        out[i] *= -1.0;
    }

    /* Fourier transform the sum */
    fftw_plan r2c2 = fftw_plan_dft_r2c_3d(N, N, N, out, fphi, FFTW_ESTIMATE);
    fft_execute(r2c2);
    fft_normalize_r2c(fphi, N, boxlen);
    fftw_destroy_plan(r2c2);

    /* Apply the inverse Poisson kernel */
    fft_apply_kernel(fphi, fphi, N, boxlen, kernel_inv_poisson, NULL);

    /* Apply a low-pass filter */
    double k_max = 0.1;
    fft_apply_kernel(fphi, fphi, N, boxlen, kernel_lowpass, &k_max);

    /* Apply inverse Fourier transform to obtain the final result */
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fphi, out, FFTW_ESTIMATE);
    fft_execute(c2r);
    fft_normalize_c2r(out, N, boxlen);
    fftw_destroy_plan(c2r);

    /* Free all the intermediate grids */
    free(fphi);
    free(fphi_xx);
    free(fphi_yy);
    free(fphi_zz);
    free(fphi_xy);
    free(fphi_xz);
    free(fphi_yz);
    free(phi_copy);
    free(phi_xx);
    free(phi_yy);
    free(phi_zz);
    free(phi_xy);
    free(phi_xz);
    free(phi_yz);
}
