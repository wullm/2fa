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
#include <assert.h>
#include <math.h>
#include <sys/time.h>

#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_errno.h>

#include "../include/3fa.h"

int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("No parameter file specified.\n");
        return 0;
    }

    /* Timer quantities */
    struct timeval time_stop, time_inter, time_start;

    /* 3FA structures */
    struct params pars;
    struct units us;
    struct perturb_data ptdat;
    struct perturb_params ptpars;

    /* Initialize MPI for distributed memory parallelization */
    MPI_Init(&argc, &argv);
    fftw_mpi_init();

    /* Get the dimensions of the cluster */
    int rank, MPI_Rank_Count;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &MPI_Rank_Count);
    int verbose = (rank == 0);

    /* Read options */
    const char *fname = argv[1];
    if (verbose) printf("3FA initial conditions.\n");

    /* Read parameter file for parameters, units */
    readParams(&pars, fname);
    readUnits(&us, fname);

    /* Read the first order potential field on each MPI rank */
    double *box;
    double BoxLen;
    int N;

    char field_fname[50] = "phi.hdf5";
    if (verbose) printf("Reading input field from %s.\n", field_fname);
    readFieldFile_MPI(&box, &N, &BoxLen, MPI_COMM_WORLD, field_fname);

    /* We will determine the growth factors on rank 0 */
    struct growth_factors_2 gfac2;
    double D2_asymp;
    double k_cutoff;

    if (rank == 0) {

        printf("(N, L) = (%d, %g)\n", N, BoxLen);

        /* Read the perturbation data file */
        readPerturb(&pars, &us, &ptdat);
        readPerturbParams(&pars, &us, &ptpars);

        /* Specify the cosmological model */
        struct model m;
        m.h = ptpars.h;
        m.Omega_b = ptpars.Omega_b;
        m.Omega_c = ptpars.Omega_m - ptpars.Omega_b; // Omega_m excludes neutrinos
        m.Omega_k = ptpars.Omega_k;
        m.N_ur = 0.00441;
        // m.N_ur = 2.0308;
        m.N_nu = 1;
        m.T_CMB_0 = ptpars.T_CMB;
        m.w0 = -1.0;
        m.wa = 0.0;
        m.sim_neutrino_nonrel_masses = 0;

        /* Copy over the massive neutrino parameters */
        m.N_nu = ptpars.N_ncdm;
        double *M_nu = malloc(m.N_nu * sizeof(double));
        double *deg_nu = malloc(m.N_nu * sizeof(double));
        double T_nu_sum = 0.;
        for (int i=0; i<m.N_nu; i++) {
            M_nu[i] = ptpars.M_ncdm_eV[i];
            deg_nu[i] = 3.0; //TODO: make variable
            T_nu_sum += ptpars.T_ncdm[i] * ptpars.T_CMB;
        }
        m.T_nu_0 = T_nu_sum / m.N_nu;
        m.M_nu = M_nu;
        m.deg_nu = deg_nu;

        printf("\n");
        printf("Cosmological model:\n");
        printf("h = %g\n", m.h);
        printf("Omega_c = %g\n", m.Omega_c);
        printf("Omega_b = %g\n", m.Omega_b);
        printf("Omega_k = %g\n", m.Omega_k);
        printf("N_ur = %g\n", m.N_ur);
        printf("M_nu[0] = %g\n", m.M_nu[0]);
        printf("deg_nu[0] = %g\n", m.deg_nu[0]);
        printf("\n");

        printf("Integrating cosmological tables.\n");

        /* Integrate the cosmological tables */
        double z_end = 31.0;
        double z_start = 1e6;
        double a_start = 1.0 / (1. + z_start);
        double a_end = 1.0 / (1. + z_end);
        struct cosmology_tables tab;
        integrate_cosmology_tables(&m, &us, &tab, 10000, a_start, fmax(1.01, a_end));

        /* Determine a_nr at which half of the neutrinos are non-relativistic */
        double target_ratio = 0.5;
        double a_nonrel = get_a_non_relativic(&tab, target_ratio);
        double z_nonrel = 1. / a_nonrel - 1.0;
        printf("z_nonrel = %g.\n", z_nonrel);
        printf("\n");
        printf("Integrating second order fluid equations.\n");

        /* Wavenumbers for the 3D table of second-order growth factors (k,k1,k2) */
        const double k_min = 2 * M_PI / BoxLen * 0.9;
        const double k_max = 2 * M_PI / BoxLen * N * 2;
        const int nk = 7;

        /* Timer */
        gettimeofday(&time_start, NULL);

        /* Begin and end for the second order growth factor integration */
        int write_tables = 1;
        integrate_fluid_equations_2(&m, &us, &tab, &ptdat, &gfac2, a_nonrel, a_end,
                                    nk, k_min, k_max, write_tables, verbose);
        // import_growth_factors_2(&gfac2, nk, k_min, k_max, MPI_COMM_WORLD);

        /* Timer */
        gettimeofday(&time_inter, NULL);
        long unsigned microsec_inter = (time_inter.tv_sec - time_start.tv_sec) * 1000000
                                     + time_inter.tv_usec - time_start.tv_usec;
        printf("\nIntegrating second order equations took %.5f s\n\n", microsec_inter/1e6);

        /* Determine the asymptotic second order growth factor */
        double D2_A_asymp_sum = 0;
        double D2_B_asymp_sum = 0;
        int asymp_count = 0;

        /* The cut-off scale */
        k_cutoff = 0.25;

        for (int i=0; i<nk; i++) {
            for (int j1=0; j1<nk; j1++) {
                for (int j2=0; j2<nk; j2++) {
                    double k = gfac2.k[i];
                    double k1 = gfac2.k[j1];
                    double k2 = gfac2.k[j2];

                    /* Skip impossible configurations using |k1|^2 + |k2|^2 -
                     * 2|k1||k2| <= |k1 + k2| |k1|^2 + |k2|^2 + 2|k1||k2|) */
                    if (k*k < 0.8 * (k1*k1 + k2*k2 - 2*k1*k2) ||
                        k*k > 1.2 * (k1*k1 + k2*k2 + 2*k1*k2)) {
                        continue;
                    }

                    if (k1 > k_cutoff && k2 > k_cutoff && k > k_cutoff) {
                        D2_A_asymp_sum += gfac2.D2_A[i * nk * nk + j1 * nk + j2];
                        D2_B_asymp_sum += gfac2.D2_B[i * nk * nk + j1 * nk + j2];
                        asymp_count++;
                    }
                }
            }
        }

        double D2_A_asymp = D2_A_asymp_sum / asymp_count;
        double D2_B_asymp = D2_B_asymp_sum / asymp_count;
        D2_asymp = 0.5 * (D2_A_asymp + D2_B_asymp);

        /* Set skipped configurations to be on the safe side */
        for (int i=0; i<nk; i++) {
            for (int j1=0; j1<nk; j1++) {
                for (int j2=0; j2<nk; j2++) {
                    double k = gfac2.k[i];
                    double k1 = gfac2.k[j1];
                    double k2 = gfac2.k[j2];

                    /* Skip impossible configurations using |k1|^2 + |k2|^2 -
                     * 2|k1||k2| <= |k1 + k2| |k1|^2 + |k2|^2 + 2|k1||k2|) */
                    if (k*k < 0.8 * (k1*k1 + k2*k2 - 2*k1*k2) ||
                        k*k > 1.2 * (k1*k1 + k2*k2 + 2*k1*k2)) {
                        gfac2.D2_A[i * nk * nk + j1 * nk + j2] = D2_asymp;
                        gfac2.D2_B[i * nk * nk + j1 * nk + j2] = D2_asymp;
                    }
                }
            }
        }

        printf("Asymptotic D2_A = %.15g\n", D2_A_asymp);
        printf("Asymptotic D2_B = %.15g\n", D2_B_asymp);
        printf("Mean (D2_A + D2_B)/2 = %.15g\n", D2_asymp);
        printf("Difference = %g\n", D2_B_asymp - D2_A_asymp);
        printf("\n");

        free(M_nu);
        free(deg_nu);

        /* Free the cosmological tables */
        free_cosmology_tables(&tab);

        /* Clean up */
        cleanPerturb(&ptdat);
        cleanPerturbParams(&ptpars);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* Broadcast the asymptotic growth factor, cutoff, and number of wavenumbers */
    MPI_Bcast(&D2_asymp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k_cutoff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&gfac2.nk, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int nk = gfac2.nk;

    /* Allocate memory for the growth factor arrays */
    if (rank > 0) {
        gfac2.k = malloc(nk * sizeof(double));
        gfac2.D2_A = malloc(nk * nk * nk * sizeof(double));
        gfac2.D2_B = malloc(nk * nk * nk * sizeof(double));
        gfac2.D2_C1 = malloc(nk * nk * nk * sizeof(double));
        gfac2.D2_C2 = malloc(nk * nk * nk * sizeof(double));
        gfac2.D2_naive = malloc(nk * nk * nk * sizeof(double));
    }

    /* Broadcast the growth factor arrays */
    MPI_Bcast(gfac2.k, nk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(gfac2.D2_A, nk * nk * nk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(gfac2.D2_B, nk * nk * nk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(gfac2.D2_C1, nk * nk * nk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(gfac2.D2_C2, nk * nk * nk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(gfac2.D2_naive, nk * nk * nk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Divide the problem over the MPI ranks */
    const double fac = (double) N / MPI_Rank_Count;
    const int X_min = ceil(rank * fac);
    const int X_max = ceil((rank + 1) * fac);
    const int NX = X_max - X_min;
    const int localProblemSize = NX * N * N;

    /* Check that all cells have been assigned to a node */
    int totalProblemSize;
    MPI_Allreduce(&localProblemSize, &totalProblemSize, 1,
                   MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    assert(totalProblemSize == N * N * N);

    printf("%03d: first = %d, last = %d, local = %d, total = %d\n", rank, X_min, X_max, localProblemSize, totalProblemSize);

    MPI_Barrier(MPI_COMM_WORLD);

    /* Allocate an output grid */
    double *out = malloc(N * N * N * sizeof(double));

    /* Do the expensive convolution */
    // bzero(out, N * N * N * sizeof(double));
    convolve(N, BoxLen, box, out, &gfac2, k_cutoff, D2_asymp, X_min, X_max, verbose);

    MPI_Barrier(MPI_COMM_WORLD);

    /* Reduce the partial solutions to convolution in Fourier space */
    if (rank == 0)
        MPI_Reduce(MPI_IN_PLACE, out, N * N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    else
        MPI_Reduce(out, MPI_IN_PLACE, N * N * N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank > 0) {
        free(out);
        free(box);

        /* Free the growth factors */
        free_growth_factors_2(&gfac2);
    }

    if (rank == 0) {

        /* Timer */
        gettimeofday(&time_stop, NULL);
        long unsigned microsec = (time_stop.tv_sec - time_inter.tv_sec) * 1000000
                               + time_stop.tv_usec - time_inter.tv_usec;
        printf("\nThe convolution took %.5f s\n", microsec/1e6);

        /* Export the partial result */
        char out_fname1[50] = "out_partial.hdf5";
        writeFieldFile(out, N, BoxLen, out_fname1);
        printf("\n");
        printf("Output written to '%s'.\n", out_fname1);

        /* Allocate a secondary output grid */
        double *out2 = malloc(N * N * N * sizeof(double));

        /* Do the fast convolution with constant kernel using FFTs */
        convolve_fft(N, BoxLen, box, out2);

        /* Add (D2_asymp - 1) times the EdS result, to obtain the difference
         * from the EdS field */
        for (int i=0; i<N*N*N; i++) {
            out[i] += (D2_asymp - 1) * out2[i];
        }

        /* Export the output grid */
        char out_fname2[50] = "out_difference.hdf5";
        writeFieldFile(out, N, BoxLen, out_fname2);
        printf("Output written to '%s'.\n", out_fname2);

        /* Finally, add the remainder to obtain the total */
        for (int i=0; i<N*N*N; i++) {
            out[i] += 1 * out2[i];
        }

        /* Export the output grid */
        char out_fname3[50] = "out_total.hdf5";
        writeFieldFile(out, N, BoxLen, out_fname3);
        printf("Output written to '%s'.\n", out_fname3);

        free(out);
        free(box);

        /* Free the growth factors */
        free_growth_factors_2(&gfac2);
    }

    /* Clean up */
    cleanParams(&pars);

    /* Done with MPI parallelization */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

}
