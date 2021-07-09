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

struct ode_params {
    struct strooklat *spline;
    struct strooklat *pt_spline_a;
    struct strooklat *pt_spline_k;
    struct cosmology_tables *tab;
    double *ratio_dnu_dcb;
    double *D_cb;
    double k, k1, k2;
    double f_b;
};

int func2(double loga, const double y[], double f[], void *params) {
    struct ode_params *p = (struct ode_params *) params;
    struct strooklat *spline = p->spline;
    struct cosmology_tables *tab = p->tab;
    struct strooklat *spline_a = p->pt_spline_a;
    struct strooklat *spline_k = p->pt_spline_k;

    double a = exp(loga);
    double k1 = p->k1;
    double k2 = p->k2;
    double k = p->k; // | k1 + k2 | != | k1 | + | k2 |
    double A = strooklat_interp(spline, tab->Avec, a);
    double B = strooklat_interp(spline, tab->Bvec, a);
    double H = strooklat_interp(spline, tab->Hvec, a);
    double f_nu_nr = strooklat_interp(spline, tab->f_nu_nr, a);
    double ratio_dnu_dcb_12 = strooklat_interp_2d(spline_a, spline_k, p->ratio_dnu_dcb, a, k);
    double ratio_dnu_dcb_1 = strooklat_interp_2d(spline_a, spline_k, p->ratio_dnu_dcb, a, k1);
    double ratio_dnu_dcb_2 = strooklat_interp_2d(spline_a, spline_k, p->ratio_dnu_dcb, a, k2);
    double D_cb_1 = strooklat_interp_2d(spline_a, spline_k, p->D_cb, a, k1);
    double D_cb_2 = strooklat_interp_2d(spline_a, spline_k, p->D_cb, a, k2);

    double B_12 = B * ((1.0 - f_nu_nr) + f_nu_nr * ratio_dnu_dcb_12);
    double B_1 = B * ((1.0 - f_nu_nr) + f_nu_nr * ratio_dnu_dcb_1);
    double B_2 = B * ((1.0 - f_nu_nr) + f_nu_nr * ratio_dnu_dcb_2);

    f[0] = -y[1];
    f[1] = A * y[1] + B_12 * (y[0] + D_cb_1 * D_cb_2);
    f[2] = -y[3];
    f[3] = A * y[3] + B_12 * y[2] + (B_1 + B_2 - B_12) * (D_cb_1 * D_cb_2);

    return GSL_SUCCESS;
}

void integrate_fluid_equations_2(struct model *m, struct units *us,
                                 struct cosmology_tables *tab,
                                 struct perturb_data *ptdat,
                                 struct growth_factors *gfac,
                                 struct growth_factors_2 *gfac2,
                                 double a_start, double a_final) {

    /* Find the necessary titles in the perturbation vector */
    int d_cdm = findTitle(ptdat->titles, "d_cdm", ptdat->n_functions);
    int d_b = findTitle(ptdat->titles, "d_b", ptdat->n_functions);
    int d_ncdm = findTitle(ptdat->titles, "d_ncdm[0]", ptdat->n_functions);

    /* The baryon fraction */
    const double f_b = m->Omega_b / (m->Omega_c + m->Omega_b);

    /* Pointers to the corresponding arrays (k_size * tau_size) */
    double *d_cdm_array = ptdat->delta + ptdat->tau_size * ptdat->k_size * d_cdm;
    double *d_b_array = ptdat->delta + ptdat->tau_size * ptdat->k_size * d_b;
    double *d_ncdm_array = ptdat->delta + ptdat->tau_size * ptdat->k_size * d_ncdm;



    /* Create an array of the transfer function ratio */
    double *ratio_dnu_dcb = malloc(ptdat->tau_size * ptdat->k_size * sizeof(double));
    double *D_cb = malloc(ptdat->tau_size * ptdat->k_size * sizeof(double));
    for (int i=0; i<ptdat->tau_size * ptdat->k_size; i++) {
        ratio_dnu_dcb[i] = d_ncdm_array[i] / (f_b * d_b_array[i] + (1.0 - f_b) * d_cdm_array[i]);
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

    /* We will differentiate the density perturbations at a_start */
    double log_a_start = log(a_start);

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
    struct ode_params odep;
    odep.spline = &spline_tab;
    odep.pt_spline_a = &spline_a;
    odep.pt_spline_k = &spline_k;
    odep.tab = tab;
    odep.f_b = f_b;
    odep.ratio_dnu_dcb = ratio_dnu_dcb;
    odep.D_cb = D_cb;

    /* Initialize the vector of wavenumbers */
    int nk = 10;
    gfac2->nk = nk;
    gfac2->k = malloc(nk * sizeof(double));
    gfac2->D2_A = malloc(nk * nk * nk * sizeof(double));
    gfac2->D2_B = malloc(nk * nk * nk * sizeof(double));

    double k_min = 1e-5;
    double k_max = 10;
    double log_k_min = log(k_min);
    double log_k_max = log(k_max);

    for (int i=0; i<nk; i++) {
        gfac2->k[i] = k_min * exp(i * (log_k_max - log_k_min) / nk);
    }

    for (int i=0; i<nk; i++) {
        /* The sum of wavenumbers */
        double k = gfac2->k[i];

        for (int j1=0; j1<nk; j1++) {
            /* The first wavenumber */
            double k1 = gfac2->k[j1];

            for (int j2=0; j2<nk; j2++) {
                /* The second wavenumber */
                double k2 =  gfac2->k[j2];

                odep.k = k;
                odep.k1 = k1;
                odep.k2 = k2;

                /* Find the values at the starting redshift and normalize */
                double Dc = strooklat_interp_2d(&spline_a, &spline_k, d_cdm_array, a_start, k1);
                double Db = strooklat_interp_2d(&spline_a, &spline_k, d_b_array, a_start, k1);
                double Dn = strooklat_interp_2d(&spline_a, &spline_k, d_ncdm_array, a_start, k1);

                /* Prepare the initial conditions */
                double y[4] = {0., 0., 0., 0.};
                double loga = log(a_start);
                double loga_final = log(a_final);

                /* Integrate */
                double tol = 1e-12;
                double hstart = 1e-12;
                gsl_odeiv2_system sys = {func2, NULL, 4, &odep};
                gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd, hstart, tol, tol);
                gsl_odeiv2_driver_apply(d, &loga, loga_final, y);
                gsl_odeiv2_driver_free(d);

                /* Extract the result */
                double U = y[0];
                double V = y[1];

                gfac2->D2_A[i * nk * nk + j1 * nk + j2] = y[0] / (3. / 7.);
                gfac2->D2_B[i * nk * nk + j1 * nk + j2] = y[2] / (3. / 7.);

                // printf("%g %g %g %g %g %g %g %g\n", k, k1, k2, y[0] / (3./7.), y[2] / (3./7.), y[1], y[3], strooklat_interp_2d(&spline_a, &spline_k, D_cb, a_start, k1));
            }
        }
    }

    /* Free the perturbation splines */
    free_strooklat_spline(&spline_a);
    free_strooklat_spline(&spline_k);
    free_strooklat_spline(&spline_tab);
    free(avec);
    free(ratio_dnu_dcb);
    free(D_cb);
}
