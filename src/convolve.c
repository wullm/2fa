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
              struct growth_factors_2 *gfac2, double k_cutoff,
              double D2_asymp, int X_min, int X_max, int verbose) {

    /* Allocate grids for the Fourier transforms */
    fftw_complex *fphi = malloc(N * N * (N/2+1) * sizeof(fftw_complex));
    fftw_complex *fout = malloc(N * N * (N/2+1) * sizeof(fftw_complex));
    bzero(fout, N * N * (N/2+1) * sizeof(fftw_complex));

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

    /* Create three wavenumber splines */
    struct strooklat spline_k = {gfac2->k, gfac2->nk};
    struct strooklat spline_k1 = {gfac2->k, gfac2->nk};
    struct strooklat spline_k2 = {gfac2->k, gfac2->nk};
    init_strooklat_spline(&spline_k, 1000);
    init_strooklat_spline(&spline_k1, 1000);
    init_strooklat_spline(&spline_k2, 1000);

    /* Cut-off scale */
    const double k_cutoff2 = k_cutoff * k_cutoff;

    /* Determine how many cells in the inner loop are below the cut-off */
    long long relevant_cells = 0;
    for (int x1=0; x1<N; x1++) {
        for (int y1=0; y1<N; y1++) {
            for (int z1=0; z1<N; z1++) {
                double k1x = (x1 > N/2) ? (x1 - N) * dk : x1 * dk;
                double k1y = (y1 > N/2) ? (y1 - N) * dk : y1 * dk;
                double k1z = (z1 > N/2) ? (z1 - N) * dk : z1 * dk;
                double k1k1 = (k1x * k1x) + (k1y * k1y) + (k1z * k1z);

                /* Skip the DC mode */
                if (k1k1 == 0.) continue;

                relevant_cells += (k1k1 < k_cutoff2);
            }
        }
    }

    if (verbose) {
        printf("\n");
        printf("There are %lld relevant cells in the inner loop.\n", relevant_cells);
    }

    /* Allocate memory for pre-computed inner loop quantities */
    double *u_k1_vec = malloc(relevant_cells * sizeof(double));
    int *ind_k1_vec = malloc(relevant_cells * sizeof(int));
    fftw_complex *phi_k1 = malloc(relevant_cells * sizeof(fftw_complex));
    int *x1_vec = malloc(relevant_cells * sizeof(int));
    int *y1_vec = malloc(relevant_cells * sizeof(int));
    int *z1_vec = malloc(relevant_cells * sizeof(int));

    /* Pre-compute the inner loop quantities */
    int count = 0;
    for (int x1=0; x1<N; x1++) {
        for (int y1=0; y1<N; y1++) {
            for (int z1=0; z1<N; z1++) {
                /* Calculate the first wavevector */
                double k1x = (x1 > N/2) ? (x1 - N) * dk : x1 * dk;
                double k1y = (y1 > N/2) ? (y1 - N) * dk : y1 * dk;
                double k1z = (z1 > N/2) ? (z1 - N) * dk : z1 * dk;
                double k1k1 = (k1x * k1x) + (k1y * k1y) + (k1z * k1z);

                /* Skip irrelevant cells */
                if (k1k1 >= k_cutoff2) continue;

                /* Skip the DC mode */
                if (k1k1 == 0.) continue;

                /* Store the cell index */
                x1_vec[count] = x1;
                y1_vec[count] = y1;
                z1_vec[count] = z1;

                /* Compute the magnitude of the wavevector */
                double k1 = sqrt(k1k1);

                /* Determine the cooresponding spline index */
                int ind_k1;
                double u_k1;
                strooklat_find_x(&spline_k1, k1, &ind_k1, &u_k1);

                /* Store the wavenumber and spline index */
                u_k1_vec[count] = u_k1;
                ind_k1_vec[count] = ind_k1;

                /* Fetch and store the grid value */
                if (z1 <= N/2) {
                    phi_k1[count] = fphi[row_major_half(x1, y1, z1, N)];
                } else {
                    phi_k1[count] = conj(fphi[row_major_half(N - x1, N - y1, N - z1, N)]);
                }

                count++;
            }
        }
    }

    if (verbose) {
        printf("Done with pre-computation of the inner loop.\n");
        printf("Performing convolution.\n\n");
    }

    /* For diagnostics only, build a histogram in (D, k) space */
    int hist_N = 100;
    int counts_A[hist_N * hist_N];
    int counts_B[hist_N * hist_N];
    for (int i = 0; i < hist_N * hist_N; i++) {
        counts_A[i] = 0;
        counts_B[i] = 0;
    }

    /* Histogram axis sizes */
    double hist_Dmin = 1.0 - 1e-3;
    double hist_Dmax = 1.0 + 5e-3;
    double hist_kmin = 5e-5;
    double hist_kmax = 1e1;

    /* Do the convolution */
    #pragma omp parallel for reduction(+:counts_A,counts_B)
    for (int x=X_min; x<X_max; x++) {
        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                /* Calculate the wavevector */
                double kx,ky,kz,k;
                fft_wavevector(x, y, z, N, dk, &kx, &ky, &kz, &k);

                if (k == 0.) continue; //skip the DC mode

                /* The id of the output cell */
                const int id = row_major_half(x,y,z,N);

                int ind[3];
                double u[3];
                strooklat_find_x(&spline_k, k, &ind[0], &u[0]);

                fftw_complex local_sum = 0;

                /* Skip irrelevant cells */
                if (k*k >= 4.0 * k_cutoff2) {
                    fout[id] = 0;
                    continue;
                }

                /* Perform the integral */
                for (int local_count = 0; local_count < relevant_cells; local_count++) {
                    /* Fetch the indices of the first wavevector */
                    int x1 = x1_vec[local_count];
                    int y1 = y1_vec[local_count];
                    int z1 = z1_vec[local_count];

                    /* The first and second wavevectors sum up to k */
                    int x2 = wrap(x - x1, N);
                    int y2 = wrap(y - y1, N);
                    int z2 = wrap(z - z1, N);

                    /* Compute the second wave vector */
                    double k2x = (x2 > N/2) ? (x2 - N) * dk : x2 * dk;
                    double k2y = (y2 > N/2) ? (y2 - N) * dk : y2 * dk;
                    double k2z = (z2 > N/2) ? (z2 - N) * dk : z2 * dk;
                    double k2k2 = (k2x * k2x) + (k2y * k2y) + (k2z * k2z);

                    /* Skip the DC mode */
                    if (k2k2 == 0.) continue;

                    /* Skip irrelevant cells */
                    if (k2k2 >= k_cutoff2) continue;

                    /* The first wavevector */
                    double k1x = (x1 > N/2) ? (x1 - N) * dk : x1 * dk;
                    double k1y = (y1 > N/2) ? (y1 - N) * dk : y1 * dk;
                    double k1z = (z1 > N/2) ? (z1 - N) * dk : z1 * dk;
                    double k1k1 = (k1x * k1x) + (k1y * k1y) + (k1z * k1z);

                    /* Fetch the corresponding spline index and offset */
                    ind[1] = ind_k1_vec[local_count];
                    u[1] = u_k1_vec[local_count];

                    /* Calculate the inner product */
                    double k1k2 = (k1x * k2x) + (k1y * k2y) + (k1z * k2z);

                    /* Determine the magnitude of the second wave vectors */
                    double k2 = sqrt(k2k2);
                    strooklat_find_x(&spline_k2, k2, &ind[2], &u[2]);

                    /* Interpolate the growth factors */
                    double D2_A = strooklat_interp_index_3d(&spline_k, &spline_k1, &spline_k2, gfac2->D2_A, ind, u) - D2_asymp;
                    double D2_B = strooklat_interp_index_3d(&spline_k, &spline_k1, &spline_k2, gfac2->D2_B, ind, u) - D2_asymp;
                    // double D2_C1 = strooklat_interp_index_3d(&spline_k, &spline_k1, &spline_k2, gfac2->D2_C1, ind, u);
                    // double D2_C2 = strooklat_interp_index_3d(&spline_k, &spline_k1, &spline_k2, gfac2->D2_C2, ind, u);

                    /* For diagnostics only, determine the histogram bins */
                    int bin_D_A = (int) (((D2_A + D2_asymp) - hist_Dmin) / (hist_Dmax - hist_Dmin) * hist_N);
                    int bin_D_B = (int) (((D2_B + D2_asymp) - hist_Dmin) / (hist_Dmax - hist_Dmin) * hist_N);
                    int bin_k = (int) ((log(k) - log(hist_kmin)) / (log(hist_kmax) - log(hist_kmin)) * hist_N);
                    int bid_A = bin_D_A * hist_N + bin_k;
                    int bid_B = bin_D_B * hist_N + bin_k;

                    /* Deposit into the histograms */
                    if (bid_A >= 0 && bid_A < hist_N * hist_N) counts_A[bid_A]++;
                    if (bid_B >= 0 && bid_B < hist_N * hist_N) counts_B[bid_B]++;

                    /* Compute the kernel */
                    fftw_complex K = 0.5 * (D2_A * k1k1 * k2k2 - D2_B * k1k2 * k1k2) / (k * k);
                    /* And the frame-lagging terms (now included in D2_A) */
                    // fftw_complex K_FL = 0.5 * (D2_C1 / k2k2 + D2_C2 / k1k1) * k1k1 * k2k2 * k1k2 / (k * k);
                    // fftw_complex K_FL = 0.5 * (D2_C1 + D2_C2) * k1k1 * k2k2 / (k * k);

                    /* Fetch the value of phi(k1) */
                    fftw_complex ph1 = phi_k1[local_count];

                    /* Fetch the value of phi(k2) */
                    fftw_complex ph2;
                    if (z2 <= N/2) {
                        ph2 = fphi[row_major_half(x2, y2, z2, N)];
                    } else {
                        ph2 = conj(fphi[row_major_half(N - x2, N - y2, N - z2, N)]);
                    }

                    /* Add the result */
                    local_sum += K * (ph1 * ph2) * dk_twopi_3;
                }

                /* Store the result of the convolution at this k */
                fout[id] = local_sum;
            }
        }
        if (x % 10 == 0) printf("%d / %d\n", x, N);
    }

    /* Free the memory of the inner loop pre-computations */
    free(phi_k1);
    free(x1_vec);
    free(y1_vec);
    free(z1_vec);
    free(ind_k1_vec);
    free(u_k1_vec);

    /* Apply inverse Fourier transform to obtain the final result */
    fftw_plan c2r = fftw_plan_dft_c2r_3d(N, N, N, fout, out, FFTW_ESTIMATE);
    fft_execute(c2r);
    fft_normalize_c2r(out, N, boxlen);
    fftw_destroy_plan(c2r);

    /* Free all the intermediate grids */
    free(fphi);
    free(fout);

    /* What is our rank? */
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* For diagnostics only, write the histograms (from this rank) to file */
    char fname_A[100], fname_B[100];
    sprintf(fname_A, "histogram_A_%d.txt", rank);
    sprintf(fname_B, "histogram_B_%d.txt", rank);
    FILE *f_A = fopen(fname_A, "w");
    FILE *f_B = fopen(fname_B, "w");
    for (int i=0; i<hist_N * hist_N; i++) {
        fprintf(f_A, "%d", counts_A[i]);
        fprintf(f_B, "%d", counts_B[i]);
        if (i < hist_N * hist_N - 1) {
            fprintf(f_A, ",");
            fprintf(f_B, ",");
        }
    }
    fclose(f_A);
    fclose(f_B);

    /* Free the splines */
    free_strooklat_spline(&spline_k);
    free_strooklat_spline(&spline_k1);
    free_strooklat_spline(&spline_k2);
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
