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

#include "../include/fluid_equations.h"
#include "../include/titles.h"
#include "../include/strooklat.h"

/* Parameters for the differential equation */
struct ode_params_2 {
    struct strooklat *spline;
    struct strooklat *spline_D;
    struct strooklat *pt_spline_a;
    struct strooklat *pt_spline_k;
    struct cosmology_tables *tab;
    double *ratio_dnu_dcb;
    double *ratio_dg_dcb;
    double *D_cb;
    double *g_asymp;
    double k, k1, k2;
    double f_b;
};

/* Differential equation for the first order growth factor
 * @param loga The independent variable: the logarithm of the scale factor
 * @param y The dependent variable: the growth factors and time derivatives
 * @param f The right-hand side of the equation (output)
 * @param params Parameters for the differential equation
*/
int ode_1st_order(double loga, const double y[2], double f[2], void *params) {
    /* Unpack ode parameters */
    struct ode_params_2 *p = (struct ode_params_2 *) params;
    struct strooklat *spline = p->spline;
    struct cosmology_tables *tab = p->tab;

    /* Scale factor and wavenumbers */
    double a = exp(loga);
    /* Cosmological functions of time only */
    double A = strooklat_interp(spline, tab->Avec, a);
    double B = strooklat_interp(spline, tab->Bvec, a);

    /* Differential equation for the asymptotic first order growth factor */
    f[0] = -y[1];
    f[1] = A * y[1] + B * y[0];

    return GSL_SUCCESS;
}

/* Differential equation for second order growth factors
 * @param D The independent variable: the asymptotic first order growth factor
 * @param y The dependent variable: the growth factors and time derivatives
 * @param f The right-hand side of the equation (output)
 * @param params Parameters for the differential equation
*/
int ode_2nd_order(double D, const double y[12], double f[12], void *params) {
    /* Unpack ode parameters */
    struct ode_params_2 *p = (struct ode_params_2 *) params;
    struct strooklat *spline_D = p->spline_D;
    struct cosmology_tables *tab = p->tab;
    struct strooklat *spline_a = p->pt_spline_a;
    struct strooklat *spline_k = p->pt_spline_k;

    /* The cosmological scale factor at this growth factor a(D) */
    double a = strooklat_interp(spline_D, tab->avec, D);

    /* Magnitude of wavevectors k1, k2, and k = k1 + k2 */
    double k1 = p->k1;
    double k2 = p->k2;
    double k = p->k; // | k | = | k1 + k2 |
    double k1_dot_k2 = 0.5 * (k*k - k1*k1 - k2*k2); // dot product

    /* Cosmological functions of time only */
    double f_nu_nr = strooklat_interp(spline_D, tab->f_nu_nr, D);
    double f_nu_tot = strooklat_interp(spline_D, tab->f_nu_tot, D);
    double f_nu_tot_0 = strooklat_interp(spline_D, tab->f_nu_tot, 1.0);   
    double f_nu_over_f_cb = f_nu_tot_0 / (1.0 - f_nu_tot_0); 
    double g_asymp = strooklat_interp(spline_D, p->g_asymp, D);
    /* Cosmological functions rewritten using D as time variable */
    double A = -1.5 * g_asymp / D;
    double B = -1.5 * g_asymp / (D * D);

    /* Ratio of neutrino density to cdm+baryon density at k, k1, k2 */
    double ratio_dnu_dcb_k = strooklat_interp_2d(spline_a, spline_k, p->ratio_dnu_dcb, a, k);
    double ratio_dnu_dcb_k1 = strooklat_interp_2d(spline_a, spline_k, p->ratio_dnu_dcb, a, k1);
    double ratio_dnu_dcb_k2 = strooklat_interp_2d(spline_a, spline_k, p->ratio_dnu_dcb, a, k2);
    
    /* Zero out the radiation part */
    ratio_dnu_dcb_k *= f_nu_nr / f_nu_tot;
    ratio_dnu_dcb_k1 *= f_nu_nr / f_nu_tot;
    ratio_dnu_dcb_k2 *= f_nu_nr / f_nu_tot;
    
    /* Pre-factor for the Poisson source terms */
    double B_k = B * (1.0 + f_nu_over_f_cb * ratio_dnu_dcb_k);
    double B_k1 = B * (1.0 + f_nu_over_f_cb * ratio_dnu_dcb_k1);
    double B_k2 = B * (1.0 + f_nu_over_f_cb * ratio_dnu_dcb_k2);

    /* First order growth factor for the cdm+baryon fluid at k1 and k2 */
    double D_cb_k1 = y[8];
    double D_cb_k2 = y[10];
    double D_cb_k1k2 = D_cb_k1 * D_cb_k2;

    /* ODE for the two second order growth factors at (k,k1,k2) */
    f[0] = -y[1];
    f[1] = A * y[1] + B_k * y[0] + B_k * D_cb_k1k2 + (B_k - B_k1) * k1_dot_k2 / (k2*k2) * D_cb_k1k2 + (B_k - B_k2) * k1_dot_k2 / (k1*k1) * D_cb_k1k2;
    f[2] = -y[3];
    f[3] = A * y[3] + B_k * y[2] + (B_k1 + B_k2 - B_k) * D_cb_k1k2;

    /* ODE for the frame-lagging terms at (k,k1,k2) */
    f[4] = -y[5];
    f[5] = A * y[5] + B_k * y[4] + (B_k - B_k1) * k1_dot_k2 / (k2*k2) * D_cb_k1k2;
    f[6] = -y[7];
    f[7] = A * y[7] + B_k * y[6] + (B_k - B_k2) * k1_dot_k2 / (k1*k1) * D_cb_k1k2;

    /* ODE for the first order growth factor evaluated at k1, k2 */
    f[8] = -y[9];
    f[9] = A * y[9] + B_k1 * y[10];
    f[10] = -y[11];
    f[11] = A * y[11] + B_k2 * y[10];

    return GSL_SUCCESS;
}

void integrate_fluid_equations_2(struct model *m, struct units *us,
                                 struct cosmology_tables *tab,
                                 struct perturb_data *ptdat,
                                 struct growth_factors_2 *gfac2, double a_final,
                                 int nk, double k_min, double k_max) {

    /* Find the necessary titles in the perturbation vector */
    int d_cdm = findTitle(ptdat->titles, "d_cdm", ptdat->n_functions);
    int d_b = findTitle(ptdat->titles, "d_b", ptdat->n_functions);
    int d_ncdm = findTitle(ptdat->titles, "d_ncdm[0]", ptdat->n_functions);
    int d_g = findTitle(ptdat->titles, "d_g", ptdat->n_functions);

    /* The baryon fraction */
    const double f_b = m->Omega_b / (m->Omega_c + m->Omega_b);

    /* Pointers to the corresponding arrays (k_size * tau_size) */
    double *d_cdm_array = ptdat->delta + ptdat->tau_size * ptdat->k_size * d_cdm;
    double *d_b_array = ptdat->delta + ptdat->tau_size * ptdat->k_size * d_b;
    double *d_ncdm_array = ptdat->delta + ptdat->tau_size * ptdat->k_size * d_ncdm;
    double *d_g_array = ptdat->delta + ptdat->tau_size * ptdat->k_size * d_g;

    /* Create an array of the transfer function ratio */
    double *ratio_dnu_dcb = malloc(ptdat->tau_size * ptdat->k_size * sizeof(double));
    double *ratio_dg_dcb = malloc(ptdat->tau_size * ptdat->k_size * sizeof(double));
    double *D_cb = malloc(ptdat->tau_size * ptdat->k_size * sizeof(double));
    for (int i=0; i<ptdat->tau_size * ptdat->k_size; i++) {
        ratio_dnu_dcb[i] = d_ncdm_array[i] / (f_b * d_b_array[i] + (1.0 - f_b) * d_cdm_array[i]);
        ratio_dg_dcb[i] = d_g_array[i] / (f_b * d_b_array[i] + (1.0 - f_b) * d_cdm_array[i]);
        D_cb[i] = (f_b * d_b_array[i] + (1.0 - f_b) * d_cdm_array[i]);
    }

    /* Normalize D_cb */
    for (int tau_index = 0; tau_index < ptdat->tau_size; tau_index++) {
        for (int k_index = 0; k_index < ptdat->k_size; k_index++) {
            int i = ptdat->k_size * tau_index + k_index;
            int i0 = ptdat->k_size * (ptdat->tau_size - 1) + k_index;
            D_cb[i] /= D_cb[i0];
        }
    }

    /* The wavenumbers and redshifts in the perturbation vector */
    double *kvec = ptdat->k;
    double *zvec = ptdat->redshift;

    /* Create a scale factor spline for the cosmological tables */
    struct strooklat spline_tab = {tab->avec, tab->size};
    init_strooklat_spline(&spline_tab, 100);

    /* The scale factors in the perturbation vector */
    double *avec = malloc(ptdat->tau_size * sizeof(double));
    for (int i=0; i<ptdat->tau_size; i++) {
       avec[i] = 1.0 / (1.0 + zvec[i]);
    }

    /* Create a scale factor spline for the perturbation factor */
    struct strooklat spline_a = {avec, ptdat->tau_size};
    init_strooklat_spline(&spline_a, 100);

    /* Create a wavenumber spline for the perturbation vector */
    struct strooklat spline_k = {kvec, ptdat->k_size};
    init_strooklat_spline(&spline_k, 100);

    /* Prepare the parameters for the fluid ODEs */
    struct ode_params_2 odep;
    odep.spline = &spline_tab;
    odep.pt_spline_a = &spline_a;
    odep.pt_spline_k = &spline_k;
    odep.tab = tab;
    odep.f_b = f_b;
    odep.ratio_dnu_dcb = ratio_dnu_dcb;
    odep.ratio_dg_dcb = ratio_dg_dcb;
    odep.D_cb = D_cb;

    /* Prepare integrating the asymptotic first order growth factor */
    double tol_1 = 1e-15;
    double hstart_1 = 1e-15;
    gsl_odeiv2_system sys_1 = {ode_1st_order, NULL, 2, &odep};
    gsl_odeiv2_driver *d_1 = gsl_odeiv2_driver_alloc_y_new(&sys_1, gsl_odeiv2_step_rk8pd, hstart_1, tol_1, tol_1);

    /* Allocate memory for the first order growth factor and the g-function */
    double *D_asymp = malloc(tab->size * sizeof(double));
    double *g_asymp = malloc(tab->size * sizeof(double));

    /* Compute normalization factor for g */
    const double H_0 = m->h * 100 * KM_METRES / MPC_METRES * us->UnitTimeSeconds;
    const double Omega_cb = m->Omega_c + m->Omega_b;
    const double B_0 = H_0 * H_0 * Omega_cb;
    const double f_nu_tot_0 = strooklat_interp(&spline_tab, tab->f_nu_tot, 1.);
    
    /* Start in the EdS limit */
    double inv_sqrt_g_start = 0.25 * (sqrt(1.0 + 24 * (1.0 - f_nu_tot_0)) - 1.0) / sqrt(1.0 - f_nu_tot_0);
    double D_dot_start = inv_sqrt_g_start * pow(tab->avec[0], -1.5) * sqrt(B_0);
    double H_start = tab->Hvec[0];
    printf("g_start = %.10g\n", 1.0 / pow(inv_sqrt_g_start,2));
    printf("H_start = %.10g %.10g\n", H_start, tab->Hvec[0]);
    double y_1[2] = {1.0, -D_dot_start/H_start};

    /* Start integrating at the beginning of the cosmological table */
    double log_a = log(tab->avec[0]);
    
    /* Integrate up to the scale factor of each row in the cosmological table */
    for (int i=0; i<tab->size; i++) {
        double loga_final = log(tab->avec[i]);

        gsl_odeiv2_driver_apply(d_1, &log_a, loga_final, y_1);

        /* Compute g = (D^2 / D_dot^2) / a^3 B_0 */
        double a = tab->avec[i];
        double H = tab->Hvec[i];
        double D = y_1[0];
        double D_dot = -y_1[1] * H;
        double g = (D * D) / (D_dot * D_dot) / (a * a * a) * B_0;

        /* Store the asymptotic values of D and g */
        D_asymp[i] = D;
        g_asymp[i] = g;
    }

    /* Finalise the integration */
    gsl_odeiv2_driver_free(d_1);

    /* Normalize the growth factor by the present-day value */
    double D_asymp_today = strooklat_interp(&spline_tab, D_asymp, 1.0);
    for (int i=0; i<tab->size; i++) {
        D_asymp[i] /= D_asymp_today;
    }

    /* Create a growth factor spline for the cosmological tables */
    struct strooklat spline_D = {D_asymp, tab->size};
    init_strooklat_spline(&spline_D, 100);

    /* Pass on references to the g-function array and growth factor spline */
    odep.g_asymp = g_asymp;
    odep.spline_D = &spline_D;

    /* Minimum and maximum wavenumbers for the second order kernel */
    double log_k_min = log(k_min);
    double log_k_max = log(k_max);

    /* Allocate memory for the second order kernel table in (k,k1,k2)-space */
    gfac2->nk = nk;
    gfac2->k = malloc(nk * sizeof(double));
    gfac2->D2_A = malloc(nk * nk * nk * sizeof(double));
    gfac2->D2_B = malloc(nk * nk * nk * sizeof(double));
    gfac2->D2_C1 = malloc(nk * nk * nk * sizeof(double));
    gfac2->D2_C2 = malloc(nk * nk * nk * sizeof(double));

    /* Initialize the wavenumbers at which to compute the second order kernel */
    for (int i=0; i<nk; i++) {
        gfac2->k[i] = k_min * exp(i * (log_k_max - log_k_min) / nk);
    }

    /* Perform the second-order growth factor calculation */
    /* We loop over values of the arguments D_2(k, k1, k2) */
    for (int i=0; i<nk; i++) { //k-loop
        for (int j1=0; j1<nk; j1++) { //k1-loop
            for (int j2=0; j2<nk; j2++) { //k2-loop

                /* Magnitude of wavevectors k1, k2, and k = k1 + k2 */
                double k1 = gfac2->k[j1];
                double k2 = gfac2->k[j2];
                double k = gfac2->k[i]; // | k | = | k1 + k2 |
                double k1_dot_k2 = 0.5 * (k*k - k1*k1 - k2*k2); // dot product

                /* Skip extreme angles that are not needed */
                // double ak1 = k1_dot_k2 / (k1*k1);
                // double ak2 = k1_dot_k2 / (k2*k2);
                // if (fabs(ak1) > 270. || fabs(ak2) > 270.) {
                //     gfac2->D2_A[i * nk * nk + j1 * nk + j2] = 1.;
                //     gfac2->D2_B[i * nk * nk + j1 * nk + j2] = 1.;
                //     continue;
                // }

                /* Skip impossible configurations using |k1|^2 + |k2|^2 -
                 * 2|k1||k2| <= |k1 + k2| |k1|^2 + |k2|^2 + 2|k1||k2|) */
                if (k*k < 0.8 * (k1*k1 + k2*k2 - 2*k1*k2) ||
                    k*k > 1.2 * (k1*k1 + k2*k2 + 2*k1*k2)) {
                    gfac2->D2_A[i * nk * nk + j1 * nk + j2] = 1.;
                    gfac2->D2_B[i * nk * nk + j1 * nk + j2] = 1.;
                    gfac2->D2_C1[i * nk * nk + j1 * nk + j2] = 0.;
                    gfac2->D2_C2[i * nk * nk + j1 * nk + j2] = 0.;
                    continue;
                }

                // k1 = k2 = 10;
                // k1 = 10.0;
                // k = sqrt(k1*k1 + k2*k2 + 0.2*k1*k2);

                /* Pass the wavenumbers on to the ODE integrator */
                odep.k = k;
                odep.k1 = k1;
                odep.k2 = k2;

                /* Integrate from the start of the cosmological table up to
                 * the growth factor corresponding to a_final */
                double D_start = D_asymp[0];
                double D_final = strooklat_interp(&spline_tab, D_asymp, a_final);

                /* Compute steady state solution at early times in asymptotic limit */
                double g_start = strooklat_interp(&spline_tab, g_asymp, tab->avec[0]);
                double E_theory = (21.0/2.0) * g_start / (6. + (9./2.) * g_start);
                                
                /* Prepare the initial conditions */
                double D2_EdS = 3./7. * E_theory * D_start * D_start;
                double D2_EdS_dot = -(6./7.) * E_theory * D_start;
                double y[12] = {D2_EdS, D2_EdS_dot, D2_EdS, D2_EdS_dot, 0, 0, 0, 0, D_start, -1, D_start, -1};

                double D = D_start;
                double D_next = D_start;

                /* Integrate */
                double tol = 1e-12;
                double hstart = 1e-12;
                gsl_odeiv2_system sys = {ode_2nd_order, NULL, 12, &odep};
                gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd, hstart, tol, tol);

                /* Number of intermediate steps */
                int steps = 1;

                for (int n = 0; n < steps; n++) {
                    D_next *= exp((log(D_final) - log(D_start)) / steps);
                    D_next = fmin(D_next, D_final);

                    gsl_odeiv2_driver_apply(d, &D, D_next, y);
                    D2_EdS = (3. / 7.) * y[8] * y[10];

                    // printf("%g %g %g %g %.8g %.8g %.8g %.8g %.8g\n", k, k1, k2, D_next, y[0] / D2_EdS, y[2] / D2_EdS, y[4] / D2_EdS, y[6] / D2_EdS, strooklat_interp(&spline_D, g_asymp, D_next));
                }

                // exit(1);

                gsl_odeiv2_driver_free(d);

                gfac2->D2_A[i * nk * nk + j1 * nk + j2] = y[0] / D2_EdS;
                gfac2->D2_B[i * nk * nk + j1 * nk + j2] = y[2] / D2_EdS;
                gfac2->D2_C1[i * nk * nk + j1 * nk + j2] = y[4] / D2_EdS;
                gfac2->D2_C2[i * nk * nk + j1 * nk + j2] = y[6] / D2_EdS;

                printf("%g %g (%.3f%%) %g %g %g %g %g %g : %.8g %.8g %.8g %.8g\n", D_final, strooklat_interp(&spline_D, g_asymp, D_final), (i * nk * nk + j1 * nk + j2) * 100.0 / (nk * nk * nk), k, k1, k2, k1_dot_k2, k1_dot_k2/(k1*k1), k1_dot_k2/(k2*k2), y[0] / D2_EdS, y[2] / D2_EdS, y[4] / D2_EdS, y[6] / D2_EdS);
            }
        }
    }

    /* Free the perturbation splines */
    free_strooklat_spline(&spline_a);
    free_strooklat_spline(&spline_k);
    free_strooklat_spline(&spline_tab);
    free(avec);
    free(ratio_dnu_dcb);
    free(ratio_dg_dcb);
    free(D_cb);
}

void free_growth_factors_2(struct growth_factors_2 *gfac2) {
    free(gfac2->k);
    free(gfac2->D2_A);
    free(gfac2->D2_B);
    free(gfac2->D2_C1);
    free(gfac2->D2_C2);
}
