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

void count_relevant_cells(double boxlen, int N, double k_cutoff, long long *work_at_x) {
    const double k_cutoff2 = k_cutoff * k_cutoff;
    const double dk = 2.0 * M_PI / boxlen;

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

    printf("There are %lld relevant cells in the inner loop.\n", relevant_cells);

    /* Allocate memory for pre-computed inner loop quantities */
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

                count++;
            }
        }
    }

    printf("Done with pre-computation of the inner loop.\n");

    for (int x=0; x<N; x++) {
        work_at_x[x] = 0;

        for (int y=0; y<N; y++) {
            for (int z=0; z<=N/2; z++) {
                /* Calculate the wavevector */
                double kx,ky,kz,k;
                fft_wavevector(x, y, z, N, dk, &kx, &ky, &kz, &k);

                if (k == 0.) continue; //skip the DC mode
                if (k*k >= 4.0 * k_cutoff2) continue;

                /* Sample the inner loop */
                for (int sample = 0; sample < 10; sample++) {
                    int local_count = rand() % relevant_cells;

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

                    work_at_x[x]++;
                }

            }
        }
    }

    /* Free memory from the inner loop */
    free(x1_vec);
    free(y1_vec);
    free(z1_vec);
}

long long relevant_cells_interval(int X_min, int X_max, long long *work_at_x) {
    long long count = 0;
    for (int x = X_min; x < X_max; x++) {
        count += work_at_x[x];
    }
    return count;
}

int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("No parameter file specified.\n");
        return 0;
    }

    /* Timer quantities */
    struct timeval time_stop, time_inter, time_work, time_start;

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

    if (verbose) printf("Reading input field from %s.\n", pars.InputFilename);
    int err = readFieldFile(&box, &N, &BoxLen, pars.InputFilename);
    if (err) {
        printf("Error reading file %s\n", pars.InputFilename);
        exit(1);
    }

    /* We will determine the growth factors on rank 0 */
    struct growth_factors_2 gfac2;
    double D2_asymp;

    /* The cut-off scale */
    const double k_cutoff = pars.CutOffScale;
    if (k_cutoff <= 0.) {
        printf("Specify a valid value for Simulation:CutOffScale.\n");
        exit(1);
    }

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
        const double k_max = 2 * M_PI / BoxLen * N * 1.1;
        const int nk = pars.WaveNumberSize;

        if (nk <= 0) {
            printf("Error: specify a positive number for Simulation:WaveNumberSize.\n");
            exit(1);
        }

        /* Timer */
        gettimeofday(&time_start, NULL);

        /* Either import the second order growth factors or integrate them */
        if (pars.ImportGrowthFactorTables) {
            import_growth_factors_2(&gfac2, nk, k_min, k_max, MPI_COMM_WORLD);
        } else {
            int write_tables = 1;
            integrate_fluid_equations_2(&m, &us, &tab, &ptdat, &gfac2, a_nonrel,
                                        a_end, nk, k_min, k_max, write_tables,
                                        verbose);
        }

        /* Verify the dimensions of the fluid equations array */
        assert(nk == gfac2.nk);
        assert((k_min - gfac2.k[0]) / k_min < 1e-9);
        assert((k_min - gfac2.k[nk - 1]) / k_max < 1e-9);

        /* Timer */
        gettimeofday(&time_inter, NULL);
        long unsigned microsec_inter = (time_inter.tv_sec - time_start.tv_sec) * 1000000
                                     + time_inter.tv_usec - time_start.tv_usec;
        printf("\nIntegrating second order equations took %.5f s\n\n", microsec_inter/1e6);

        /* Determine the asymptotic second order growth factor */
        double D2_A_asymp;
        double D2_B_asymp;
        double D2_mean_asymp;
        compute_asymptotic_values(&gfac2, &D2_A_asymp, &D2_B_asymp,
                                  &D2_mean_asymp, k_cutoff);

        D2_asymp = D2_mean_asymp;

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

    /* Determine an equitable distribution of work over the ranks */
    int X_edges[MPI_Rank_Count + 1];
    if (rank == 0 && MPI_Rank_Count > 1) {
        printf("Determining distribution of work over the ranks.\n");

        /* Determine how much work there is in total. */
        long long *work_at_x = malloc(N * sizeof(long long));
        count_relevant_cells(BoxLen, N, k_cutoff, work_at_x);
        const long long total_work = relevant_cells_interval(0, N, work_at_x);
        const long long expected_work = total_work / MPI_Rank_Count;
        printf("Total work %lld\n", total_work);

        /* Parameters for the work assignment */
        const double greedy_fraction = pars.GreedyFraction;
        const double length_penalty = pars.LengthPenalty;
        printf("Greedy fraction = %g\n", greedy_fraction);
        printf("Length penality = %g\n", length_penalty);

        /* Determine the grid intervals that each rank will operate on */
        X_edges[0] = 0;
        for (int i = 1; i < MPI_Rank_Count + 1; i++) {
            double penalty = 1.0;
            long long work = 0;
            X_edges[i] = X_edges[i-1];
            while(work * penalty < greedy_fraction * expected_work && X_edges[i] < N) {
                work = relevant_cells_interval(X_edges[i-1], X_edges[i], work_at_x);
                penalty *= length_penalty;
                X_edges[i]++;
            }

            /* Make sure that all the work is accounted for */
            if (i == MPI_Rank_Count) {
                X_edges[i] = N;
            }
        }

        printf("\nWork distribution:\n");
        for (int i = 0; i < MPI_Rank_Count; i++) {
            long long work = relevant_cells_interval(X_edges[i], X_edges[i+1], work_at_x);
            printf("%d %d %d %lld\n", i, X_edges[i], X_edges[i+1], work);
        }

        free(work_at_x);

        /* Timer */
        gettimeofday(&time_work, NULL);
        long unsigned microsec = (time_work.tv_sec - time_inter.tv_sec) * 1000000
                                + time_work.tv_usec - time_inter.tv_usec;
        printf("The work distribution took %.5f s\n\n", microsec/1e6);
    } else {
        X_edges[0] = 0;
        X_edges[1] = N;
        gettimeofday(&time_work, NULL);
    }

    /* Broadcast the edges of the grid that each node will operate on */
    MPI_Bcast(X_edges, MPI_Rank_Count + 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Divide the problem over the MPI ranks */
    const int X_min = X_edges[rank];
    const int X_max = X_edges[rank + 1];
    const int NX = X_max - X_min;
    const long long localProblemSize = (long long) NX * N * N;

    /* Check that all cells have been assigned to a node */
    long long totalProblemSize;
    MPI_Allreduce(&localProblemSize, &totalProblemSize, 1,
                   MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    assert(totalProblemSize == (long long) N * N * N);

    printf("%03d: first = %d, last = %d, local = %lld, total = %lld\n", rank, X_min, X_max, localProblemSize, totalProblemSize);

    MPI_Barrier(MPI_COMM_WORLD);

    /* Allocate an output grid */
    double *out = malloc(N * N * N * sizeof(double));

    /* Do the expensive convolution */
    // bzero(out, N * N * N * sizeof(double));
    convolve(N, BoxLen, box, out, &gfac2, k_cutoff, D2_asymp, X_min, X_max, verbose);

    /* Export the local version of the grid (we use gzip, so relatively cheap) */
    char out_fname_local[50];
    sprintf(out_fname_local, "local_partial_%d.hdf5", rank);
    writeFieldFileCompressed(out, N, BoxLen, out_fname_local, 0); // 0 = lossless

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
        long unsigned microsec = (time_stop.tv_sec - time_work.tv_sec) * 1000000
                               + time_stop.tv_usec - time_work.tv_usec;
        printf("\nThe convolution took %.5f s\n", microsec/1e6);

        /* Export the partial result */
        char out_fname1[50] = "out_partial.hdf5";
        writeFieldFileCompressed(out, N, BoxLen, out_fname1, 0); // 0 = lossless
        printf("\n");
        printf("Output written to '%s'.\n", out_fname1);

        /* Next, we will either compute or import the traditional result of
         * the second order potential (with EdS kernel) */
        double *phi2_EdS;
        if (pars.ImportSecondOrderPotential) {
            /* Read the second order potential field */
            double read_BoxLen;
            int read_N;

            if (verbose) printf("Reading input field from %s.\n", pars.SecondOrderPotentialFile);
            err = readFieldFile(&phi2_EdS, &read_N, &read_BoxLen, pars.SecondOrderPotentialFile);
            assert(N == read_N);
            assert(read_BoxLen == BoxLen);
            if (err) {
                printf("Error reading file %s\n", pars.SecondOrderPotentialFile);
                exit(1);
            }
        } else {
            /* Allocate a secondary output grid */
            phi2_EdS = malloc(N * N * N * sizeof(double));

            /* Do the fast convolution with constant kernel using FFTs */
            convolve_fft(N, BoxLen, box, phi2_EdS);
        }

        /* Add the EdS result to the partial result (without the asymptotic
         * contribution) */
        for (int i=0; i<N*N*N; i++) {
            out[i] += phi2_EdS[i];
        }

        /* Export the output grid */
        char out_fname2[50] = "out_perturbed.hdf5";
        writeFieldFileCompressed(out, N, BoxLen, out_fname2, 0); // 0 = lossless
        printf("Output written to '%s'.\n", out_fname2);

        /* Finally, add (D2_asymp - 1) times the EdS result to obtain the total */
        for (int i=0; i<N*N*N; i++) {
            out[i] += (D2_asymp - 1) * phi2_EdS[i];
        }

        /* Export the output grid */
        char out_fname3[50] = "out_total.hdf5";
        writeFieldFileCompressed(out, N, BoxLen, out_fname3, 0); // 0 = lossless
        printf("Output written to '%s'.\n", out_fname3);

        free(out);
        free(box);
        free(phi2_EdS);

        /* Free the growth factors */
        free_growth_factors_2(&gfac2);
    }

    /* Clean up */
    cleanParams(&pars);

    /* Done with MPI parallelization */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

}
