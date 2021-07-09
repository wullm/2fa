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
    struct cosmology_tables *tab;
    double k;
    double f_b;
    double c_s;
};

int func (double loga, const double y[], double f[], void *params) {
    struct ode_params *p = (struct ode_params *) params;
    struct strooklat *spline = p->spline;
    struct cosmology_tables *tab = p->tab;

    double a = exp(loga);
    double A = strooklat_interp(spline, tab->Avec, a);
    double B = strooklat_interp(spline, tab->Bvec, a);
    double H = strooklat_interp(spline, tab->Hvec, a);
    double f_nu_nr = strooklat_interp(spline, tab->f_nu_nr, a);

    double c_s = p->c_s / a;
    double k = p->k;
    double k_fs2 = -B * H * H / (c_s * c_s) * (a * a);
    double f_b = p->f_b;
    double D_cb = (1.0 - f_b) * y[0] + f_b * y[2];

    f[0] = -y[1];
    f[1] = A * y[1] + B * ((1.0 - f_nu_nr) * D_cb + f_nu_nr * y[4]);
    f[2] = -y[3];
    f[3] =  A * y[3] + B * ((1.0 - f_nu_nr) * D_cb + f_nu_nr * y[4]);
    f[4] = -y[5];
    f[5] = A * y[5] + B * ((1.0 - f_nu_nr) * D_cb + (f_nu_nr - (k*k)/k_fs2)*y[4]);

    return GSL_SUCCESS;
}

void integrate_fluid_equations(struct model *m, struct units *us,
                               struct cosmology_tables *tab,
                               struct perturb_data *ptdat,
                               struct growth_factors *gfac,
                               double a_start, double a_final) {

    /* Find the necessary titles in the perturbation vector */
    int d_cdm = findTitle(ptdat->titles, "d_cdm", ptdat->n_functions);
    int d_b = findTitle(ptdat->titles, "d_b", ptdat->n_functions);
    int d_ncdm = findTitle(ptdat->titles, "d_ncdm[0]", ptdat->n_functions);

    /* Pointers to the corresponding arrays (k_size * tau_size) */
    double *d_cdm_array = ptdat->delta + ptdat->tau_size * ptdat->k_size * d_cdm;
    double *d_b_array = ptdat->delta + ptdat->tau_size * ptdat->k_size * d_b;
    double *d_ncdm_array = ptdat->delta + ptdat->tau_size * ptdat->k_size * d_ncdm;

    /* The wavenumbers and redshifts in the perturbation vector */
    double *kvec = ptdat->k;
    double *zvec = ptdat->redshift;

    /* We will differentiate the density perturbations at a_start */
    double log_a_start = log(a_start);
    double delta_log_a = 0.002;

    /* Nearby scale factors */
    double a_mm = exp(log_a_start - 2.0 * delta_log_a);
    double a_m = exp(log_a_start - 1.0 * delta_log_a);
    double a_p = exp(log_a_start + 1.0 * delta_log_a);
    double a_pp = exp(log_a_start + 2.0 * delta_log_a);

    /* Create a scale factor spline for the cosmological tables */
    struct strooklat spline_tab = {tab->avec, tab->size};
    init_strooklat_spline(&spline_tab, 100);

    /* Compute the Hubble ratio */
    double H_start = strooklat_interp(&spline_tab, tab->Hvec, a_start);
    double H_final = strooklat_interp(&spline_tab, tab->Hvec, a_final);

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
    odep.tab = tab;
    odep.f_b = m->Omega_b / (m->Omega_c + m->Omega_b);
    odep.c_s = 134.423 / m->M_nu[0] * KM_METRES / us->UnitLengthMetres * us->UnitTimeSeconds;

    /* Allocate tables for the results */
    gfac->k = malloc(ptdat->k_size * sizeof(double));
    gfac->Dc = malloc(ptdat->k_size * sizeof(double));
    gfac->Db = malloc(ptdat->k_size * sizeof(double));
    gfac->Dn = malloc(ptdat->k_size * sizeof(double));
    gfac->Tc = malloc(ptdat->k_size * sizeof(double));
    gfac->Tb = malloc(ptdat->k_size * sizeof(double));
    gfac->Tn = malloc(ptdat->k_size * sizeof(double));
    gfac->gc = malloc(ptdat->k_size * sizeof(double));
    gfac->gb = malloc(ptdat->k_size * sizeof(double));
    gfac->gn = malloc(ptdat->k_size * sizeof(double));
    gfac->nk = ptdat->k_size;

    /* Copy the wavenumbers from the perturbation file */
    memcpy(gfac->k, ptdat->k, ptdat->k_size * sizeof(double));

    /* For each wavenumber */
    for (int i=0; i<10; i++) {
    // for (int i=0; i<ptdat->k_size; i++) {
        /* The wavenumber of interest */
        double k = kvec[i];
        odep.k = k;

        /* Find the values at the starting redshift and normalize */
        double Dc = strooklat_interp_2d(&spline_a, &spline_k, d_cdm_array, a_start, k);
        double Db = strooklat_interp_2d(&spline_a, &spline_k, d_b_array, a_start, k);
        double Dn = strooklat_interp_2d(&spline_a, &spline_k, d_ncdm_array, a_start, k);

        /* Initial conditions */
        double beta_c = Dc / Dc;
        double beta_b = Db / Dc;
        double beta_n = Dn / Dc;


        /* Compute the derivatives */
        double dDc_dloga = 0;
        dDc_dloga += strooklat_interp_2d(&spline_a, &spline_k, d_cdm_array, a_mm, k);
        dDc_dloga -= strooklat_interp_2d(&spline_a, &spline_k, d_cdm_array, a_m, k) * 8.0;
        dDc_dloga += strooklat_interp_2d(&spline_a, &spline_k, d_cdm_array, a_p, k) * 8.0;
        dDc_dloga -= strooklat_interp_2d(&spline_a, &spline_k, d_cdm_array, a_pp, k);
        dDc_dloga /= 12.0 * delta_log_a;
        double dDb_dloga = 0;
        dDb_dloga += strooklat_interp_2d(&spline_a, &spline_k, d_b_array, a_mm, k);
        dDb_dloga -= strooklat_interp_2d(&spline_a, &spline_k, d_b_array, a_m, k) * 8.0;
        dDb_dloga += strooklat_interp_2d(&spline_a, &spline_k, d_b_array, a_p, k) * 8.0;
        dDb_dloga -= strooklat_interp_2d(&spline_a, &spline_k, d_b_array, a_pp, k);
        dDb_dloga /= 12.0 * delta_log_a;
        double dDn_dloga = 0;
        dDn_dloga += strooklat_interp_2d(&spline_a, &spline_k, d_ncdm_array, a_mm, k);
        dDn_dloga -= strooklat_interp_2d(&spline_a, &spline_k, d_ncdm_array, a_m, k) * 8.0;
        dDn_dloga += strooklat_interp_2d(&spline_a, &spline_k, d_ncdm_array, a_p, k) * 8.0;
        dDn_dloga -= strooklat_interp_2d(&spline_a, &spline_k, d_ncdm_array, a_pp, k);
        dDn_dloga /= 12.0 * delta_log_a;

        /* Growth rates at a_start */
        double gc = dDc_dloga / Dc;
        double gb = dDb_dloga / Db;
        double gn = dDn_dloga / Dn;

        /* Prepare the initial conditions */
        double y[6] = {beta_c, -gc * beta_c, beta_b, -gb * beta_b, beta_n, -gn * beta_n};
        double loga = log(a_start);
        double loga_final = log(a_final);

        /* Integrate */
        double tol = 1e-12;
        double hstart = 1e-12;
        gsl_odeiv2_system sys = {func, NULL, 6, &odep};
        gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rk8pd, hstart, tol, tol);
        gsl_odeiv2_driver_apply(d, &loga, loga_final, y);
        gsl_odeiv2_driver_free(d);

        /* Extract the result */
        double Dc_final = y[0];
        double Db_final = y[2];
        double Dn_final = y[4];
        double gc_final = -y[1]/y[0];
        double gb_final = -y[3]/y[2];
        double gn_final = -y[5]/y[4];

        /* Store the results (relative density and velocity transfer functions)*/
        gfac->Dc[i] = beta_c / Dc_final;
        gfac->Db[i] = beta_b / Db_final;
        gfac->Dn[i] = beta_n / Dn_final;
        gfac->Tc[i] = (gc / gc_final) * (H_start * a_start) / (H_final * a_final) * gfac->Dc[i];
        gfac->Tb[i] = (gb / gb_final) * (H_start * a_start) / (H_final * a_final) * gfac->Db[i];
        gfac->Tn[i] = (gn / gn_final) * (H_start * a_start) / (H_final * a_final) * gfac->Dn[i];
        gfac->gc[i] = gc;
        gfac->gb[i] = gb;
        gfac->gn[i] = gn;
    }

    /* Free the perturbation splines */
    free_strooklat_spline(&spline_a);
    free_strooklat_spline(&spline_k);
    free_strooklat_spline(&spline_tab);
    free(avec);
}


void free_growth_factors(struct growth_factors *gfac) {
    free(gfac->k);
    free(gfac->Dc);
    free(gfac->Db);
    free(gfac->Dn);
    free(gfac->Tc);
    free(gfac->Tb);
    free(gfac->Tn);
    free(gfac->gc);
    free(gfac->gb);
    free(gfac->gn);
}
