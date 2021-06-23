/*******************************************************************************
 * This file is part of 3fa.
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
#include "../include/write_transfers.h"
#include "../include/titles.h"
#include "../include/strooklat.h"

void write_transfer_functions(struct model *m, struct units *us,
                              struct cosmology_tables *tab,
                              struct perturb_data *ptdat,
                              struct growth_factors *gfac,
                              double a_start, double a_final,
                              char fname[100]) {

    /* Find the necessary titles in the perturbation vector */
    int d_cdm = findTitle(ptdat->titles, "d_cdm", ptdat->n_functions);
    int d_b = findTitle(ptdat->titles, "d_b", ptdat->n_functions);
    int d_ncdm = findTitle(ptdat->titles, "d_ncdm[0]", ptdat->n_functions);
    int d_g = findTitle(ptdat->titles, "d_g", ptdat->n_functions);
    int d_ur = findTitle(ptdat->titles, "d_ur", ptdat->n_functions);
    int d_tot = findTitle(ptdat->titles, "d_tot", ptdat->n_functions);
    int t_cdm = findTitle(ptdat->titles, "t_cdm", ptdat->n_functions);
    int t_b = findTitle(ptdat->titles, "t_b", ptdat->n_functions);
    int t_ncdm = findTitle(ptdat->titles, "t_ncdm[0]", ptdat->n_functions);

    /* Pointers to the corresponding arrays (k_size * tau_size) */
    double *d_cdm_array = ptdat->delta + ptdat->tau_size * ptdat->k_size * d_cdm;
    double *d_b_array = ptdat->delta + ptdat->tau_size * ptdat->k_size * d_b;
    double *d_ncdm_array = ptdat->delta + ptdat->tau_size * ptdat->k_size * d_ncdm;
    double *d_g_array = ptdat->delta + ptdat->tau_size * ptdat->k_size * d_g;
    double *d_ur_array = ptdat->delta + ptdat->tau_size * ptdat->k_size * d_ur;
    double *d_tot_array = ptdat->delta + ptdat->tau_size * ptdat->k_size * d_tot;
    double *t_cdm_array = ptdat->delta + ptdat->tau_size * ptdat->k_size * t_cdm;
    double *t_b_array = ptdat->delta + ptdat->tau_size * ptdat->k_size * t_b;
    double *t_ncdm_array = ptdat->delta + ptdat->tau_size * ptdat->k_size * t_ncdm;

    /* The wavenumbers and redshifts in the perturbation vector */
    double *kvec = ptdat->k;
    double *zvec = ptdat->redshift;

    /* The scale factors in the perturbation vector */
    double *avec = malloc(ptdat->tau_size * sizeof(double));
    for (int i=0; i<ptdat->tau_size; i++) {
       avec[i] = 1.0 / (1.0 + zvec[i]);
    }

    /* Create a scale factor spline for the cosmological tables */
    struct strooklat spline_tab = {tab->avec, tab->size};
    init_strooklat_spline(&spline_tab, 100);

    /* Create a scale factor spline for the perturbation factor */
    struct strooklat spline_a = {avec, ptdat->tau_size};
    init_strooklat_spline(&spline_a, 100);

    /* Create a wavenumber spline for the perturbation vector */
    struct strooklat spline_k = {kvec, ptdat->k_size};
    init_strooklat_spline(&spline_k, 100);

    /* Fraction of non-relativistic massive neutrinos in matter density */
    const double a_today = 1.0;
    const double f_nu_nr_0 = strooklat_interp(&spline_tab, tab->f_nu_nr, a_today);
    const double f_b = m->Omega_b / (m->Omega_c + m->Omega_b);

    /* Hubble rate at the starting and final redshifts */
    const double H_start = strooklat_interp(&spline_tab, tab->Hvec, a_start);
    const double H_final = strooklat_interp(&spline_tab, tab->Hvec, a_final);

    /* Find a time slightly before the final time */
    const double delta_log_a = 0.002;
    const double a_m = a_final * exp(-1.0 * delta_log_a);

    /* Determine the asymptotic growth rates for k --> infinity */
    double Dm_sum = 0;
    double gm_sum = 0;
    double gm_final_sum = 0;
    int count = 0;
    for (int i=0; i<ptdat->k_size; i++) {
        /* The wavenumber of interest */
        double k = gfac->k[i];

        /* Only include small scale modes (cut-off at 1/Mpc) */
        if (k < 1.0 * MPC_METRES / us->UnitLengthMetres) continue;

        /* Find the back-scaled growth factors and rates */
        double Dc = gfac->Dc[i];
        double Db = gfac->Db[i];
        double Dn = gfac->Dn[i];
        double gc = gfac->gc[i];
        double gb = gfac->gb[i];
        double gn = gfac->gn[i];

        /* Find the growth factors at the final time from the perturbation vector */
        double Dc_final = strooklat_interp_2d(&spline_a, &spline_k, d_cdm_array, a_final, k);
        double Db_final = strooklat_interp_2d(&spline_a, &spline_k, d_b_array, a_final, k);
        double Dn_final = strooklat_interp_2d(&spline_a, &spline_k, d_ncdm_array, a_final, k);
        /* Find the values slightly before the final redshift */
        double Dc_m = strooklat_interp_2d(&spline_a, &spline_k, d_cdm_array, a_m, k);
        double Db_m = strooklat_interp_2d(&spline_a, &spline_k, d_b_array, a_m, k);
        double Dn_m = strooklat_interp_2d(&spline_a, &spline_k, d_ncdm_array, a_m, k);
        /* Compute the backward-difference derivatives at a_final */
        double dDc_dloga = (Dc_final - Dc_m) / delta_log_a;
        double dDb_dloga = (Db_final - Db_m) / delta_log_a;
        double dDn_dloga = (Dn_final - Dn_m) / delta_log_a;
        /* Compute the growth rates at a_final */
        double gc_final = dDc_dloga / Dc_final;
        double gb_final = dDb_dloga / Db_final;
        double gn_final = dDn_dloga / Dn_final;

        /* Use the weighted average growth factor for the total density */
        double Dcb = (1.0 - f_b) * Dc + f_b * Db;
        double Dcb_final = (1.0 - f_b) * Dc_final + f_b * Db_final;
        double Dm = (1.0 - f_nu_nr_0) * Dcb + f_nu_nr_0 * Dn;
        double Dm_final = (1.0 - f_nu_nr_0) * Dcb_final + f_nu_nr_0 * Dn_final;

        /* Use the weighted average of the growth rate */
        double gcb = ((1.0 - f_b) * gc * Dc + f_b * gb * Db) / Dcb;
        double gcb_final = ((1.0 - f_b) * gc_final * Dc_final + f_b * gb_final * Db_final) / Dcb_final;
        double gm = ((1.0 - f_nu_nr_0) * gcb * Dcb + f_nu_nr_0 * gn * Dn) / Dm;
        double gm_final = ((1.0 - f_nu_nr_0) * gcb_final * Dcb_final + f_nu_nr_0 * gn_final * Dn_final) / Dm_final;

        /* Accumulate */
        Dm_sum += Dm;
        gm_sum += gm;
        gm_final_sum += gm_final;
        count++;
    }

    /* The asymptotic growth factor and rate, given by the mean values for k > 1/Mpc */
    const double Dm_asymptotic = Dm_sum / count;
    const double gm_asymptotic = gm_sum / count;
    const double gm_final_asymptotic = gm_final_sum / count;

    /* Convert to h/Mpc? */
    const int convert_to_CAMB_units = 1;
    const double h = m->h;

    /* Open output file */
    FILE *f = fopen(fname, "w");

    /* Print a header row */
    fprintf(f, "#Transfer functions at z = %.10g, a = %.10g, f_nu_nr_0 = %.10g, f_b/(f_b + f_c) = %.10g\n", 1./a_start - 1., a_start, f_nu_nr_0, f_b);
    fprintf(f, "#k delta_cdm delta_b delta_g delta_ur delta_ncdm delta_tot dummy dummy theta_ncdm theta_cdm theta_b theta_bc\n");

    /* For each wavenumber */
    for (int i=0; i<ptdat->k_size; i++) {
        /* The wavenumber of interest */
        double k = gfac->k[i];

        /* Find the linear theory values at the starting redshift and normalize */
        double Dc = strooklat_interp_2d(&spline_a, &spline_k, d_cdm_array, a_start, k);
        double Db = strooklat_interp_2d(&spline_a, &spline_k, d_b_array, a_start, k);
        double Dn = strooklat_interp_2d(&spline_a, &spline_k, d_ncdm_array, a_start, k);
        double Dg = strooklat_interp_2d(&spline_a, &spline_k, d_g_array, a_start, k);
        double Dur = strooklat_interp_2d(&spline_a, &spline_k, d_ur_array, a_start, k);
        double Dtot = strooklat_interp_2d(&spline_a, &spline_k, d_tot_array, a_start, k);
        double Tc = strooklat_interp_2d(&spline_a, &spline_k, t_cdm_array, a_start, k);
        double Tb = strooklat_interp_2d(&spline_a, &spline_k, t_b_array, a_start, k);
        double Tn = strooklat_interp_2d(&spline_a, &spline_k, t_ncdm_array, a_start, k);

        /* Find the linear theory values at the final redshift */
        double Dc_final = strooklat_interp_2d(&spline_a, &spline_k, d_cdm_array, a_final, k);
        double Db_final = strooklat_interp_2d(&spline_a, &spline_k, d_b_array, a_final, k);
        double Dn_final = strooklat_interp_2d(&spline_a, &spline_k, d_ncdm_array, a_final, k);
        double Dg_final = strooklat_interp_2d(&spline_a, &spline_k, d_g_array, a_final, k);
        double Dur_final = strooklat_interp_2d(&spline_a, &spline_k, d_ur_array, a_final, k);
        double Tc_final = strooklat_interp_2d(&spline_a, &spline_k, t_cdm_array, a_final, k);
        double Tb_final = strooklat_interp_2d(&spline_a, &spline_k, t_b_array, a_final, k);
        double Tn_final = strooklat_interp_2d(&spline_a, &spline_k, t_ncdm_array, a_final, k);

        /* Rescale back to a_start using the growth factors from the 3-fluid calculation */
        double Dc_start = Dc_final * gfac->Dc[i];
        double Db_start = Db_final * gfac->Db[i];
        double Dn_start = Dn_final * gfac->Dn[i];
        // double Tc_start = Tc_final * gfac->Tc[i];
        // double Tb_start = Tb_final * gfac->Tb[i];
        // double Tn_start = Tn_final * gfac->Tn[i];

        /* The non-rescaled energy fluxes at a_start */
        double Tc_start = Tc;
        double Tb_start = Tb;
        double Tn_start = Tn;

        /* Wavenumber in h/Mpc, dimensionless densities and energy fluxes */
        if (convert_to_CAMB_units) {
            k /= MPC_METRES / us->UnitLengthMetres * h;

            /* Export the dimensionless quantity theta / (g*a*H) */
            Tc /= gm_asymptotic * H_start * a_start;
            Tb /= gm_asymptotic * H_start * a_start;
            Tn /= gm_asymptotic * H_start * a_start;

            Tc_start /= gm_asymptotic * H_start * a_start;
            Tb_start /= gm_asymptotic * H_start * a_start;
            Tn_start /= gm_asymptotic * H_start * a_start;

            Tc_final /= gm_final_asymptotic * H_final * a_final;
            Tb_final /= gm_final_asymptotic * H_final * a_final;
            Tn_final /= gm_final_asymptotic * H_final * a_final;
        }

        /* Compute the weighted average of the back-scaled quantities */
        double Dcb_start = (1.0 - f_b) * Dc_start + f_b * Db_start;
        double Dm_start = (1.0 - f_nu_nr_0) * Dcb_start + f_nu_nr_0 * Dn_start;
        double Tcb_start = (1.0 - f_b) * Tc_start + f_b * Tb_start;
        double Tm_start = (1.0 - f_nu_nr_0) * Tcb_start + f_nu_nr_0 * Tn_start;

        /* Now scale forward with the asymptotic total matter growth factor */
        double Dm_rescaled = Dm_start / Dm_asymptotic;
        double Dn_rescaled = Dn_start / Dm_asymptotic;
        double Dcb_rescaled = Dcb_start / Dm_asymptotic;
        double Tm_rescaled = Tm_start / Dm_asymptotic;
        double Tn_rescaled = Tn_start / Dm_asymptotic;
        double Tcb_rescaled = Tcb_start / Dm_asymptotic;

        // not used, but computed for consistency
        (void) Tm_start;
        (void) Tm_rescaled;

        /* Use the compensated mode at the target redshift to set d_b and d_c */
        double Dbc_final = Db_final - Dc_final;
        double Dc_rescaled = Dcb_rescaled - f_b * Dbc_final;
        double Db_rescaled = Dcb_rescaled + (1.0 - f_b) * Dbc_final;

        /* Do the same for the velocities */
        double Tbc_final = -(Tb_final - Tc_final); //confusing sign convention, TODO: check
        double Tc_rescaled = Tcb_rescaled - f_b * Tbc_final;
        double Tb_rescaled = Tcb_rescaled + (1.0 - f_b) * Tbc_final;

        /* Do we want to export the original (not rescaled) transfer functions at z_start? */
        int export_original = 0;
        if (export_original) {
            fprintf(f, "%.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g\n", k, Dc, Db, Dg, Dur, Dn, Dtot, 0.0, 0.0, Tn, Tc, Tb, Tb - Tc);
        } else {
            // fprintf(f, "%.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g\n", k, Dc_start, Db_start, Dg, Dur, Dn_start, Dm_start, 0.0, 0.0, Tn_start, Tc_start, Tb_start, Tb_start - Tc_start);
            fprintf(f, "%.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g\n", k, Dc_rescaled, Db_rescaled, Dg_final, Dur_final, Dn_rescaled, Dm_rescaled, 0.0, 0.0, Tn_rescaled, Tc_rescaled, Tb_rescaled, Tb_rescaled - Tc_rescaled);
        }
    }

    fclose(f);

    printf("Transfer functions written to '%s'.\n", fname);

    /* Convert the Hubble rate to km/s/Mpc */
    const double H_kms_Mpc = H_start / (KM_METRES / MPC_METRES * us->UnitTimeSeconds);
    const double H_final_kms_Mpc = H_final / (KM_METRES / MPC_METRES * us->UnitTimeSeconds);

    printf("\n");
    printf("Asymptotic values as k --> infinity:\n");
    printf("Dm(a_start) = %.10g\n", Dm_asymptotic);
    printf("gm(a_start) = %.10g = dlog(Dm)/dlog(a)\n", gm_asymptotic);
    printf("H(a_start)  = %.10g 1/U_T\n", H_start);
    printf("            = %.10g km/s/Mpc\n", H_kms_Mpc);
    printf("gm(a_final) = %.10g = dlog(Dm)/dlog(a)\n", gm_final_asymptotic);
    printf("H(a_final)  = %.10g 1/U_T\n", H_final);
    printf("            = %.10g km/s/Mpc\n", H_final_kms_Mpc);
    printf("\n");
    printf("f_nu_nr_0     = %.10g\n", f_nu_nr_0);
    printf("f_b/(f_b+f_c) = %.10g\n", f_b);

    /* Free the perturbation splines */
    free_strooklat_spline(&spline_a);
    free_strooklat_spline(&spline_k);
    free_strooklat_spline(&spline_tab);
    free(avec);
}
