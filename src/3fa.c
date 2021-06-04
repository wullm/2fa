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

#include "../include/3fa.h"
#include "../include/strooklat.h"

int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("No parameter file specified.\n");
        return 0;
    }

    /* Read options */
    const char *fname = argv[1];
    printf("3FA initial conditions.\n");
    
    /* Timer */
    struct timeval time_stop, time_start;
    gettimeofday(&time_start, NULL);
    
    /* 3FA structures */
    struct params pars;
    struct units us;
    struct perturb_data ptdat;
    struct perturb_params ptpars;
    
    /* Read parameter file for parameters, units */
    readParams(&pars, fname);
    readUnits(&us, fname);
    
    /* Read the perturbation data file */
    readPerturb(&pars, &us, &ptdat);
    readPerturbParams(&pars, &us, &ptpars);
    
    /* Our choice of neutrino masses */
    double M_nu[1] = {0.067666666};
    double deg_nu[1] = {3.0};
    
    /* Specify the cosmological model */
    struct model m;
    m.h = 0.6771;
    m.Omega_b = 0.0495;
    m.Omega_c = 0.2491464152;
    m.Omega_k = 0.;
    m.N_ur = 0.00441;
    m.N_nu = 1;
    m.M_nu = M_nu;
    m.deg_nu = deg_nu;
    m.T_nu_0 = 1.951757805;
    m.T_CMB_0 = 2.7255;
    m.w0 = -1.0;
    m.wa = 0.0;
    
    printf("Integrating cosmological tables.\n");
    
    /* Integrate the cosmological tables */
    struct cosmology_tables tab;
    integrate_cosmology_tables(&m, &us, &tab, 1000);
    
    printf("Integrating fluid equations.\n");
    
    /* Integrate the fluid equations */
    double z_start = 31.0;
    double a_start = 1.0 / (1 + z_start);
    double a_final = 1.0;
    struct growth_factors gfac;
    integrate_fluid_equations(&m, &us, &tab, &ptdat, &gfac, a_start, a_final); 
    
    /* Print the results */
    printf("Relative scaling of:\n");
    printf("delta_c delta_b delta_n theta_c theta_b theta_n:\n");
    for (int i=0; i<gfac.nk; i++) {
        printf("%g %g %g %g %g %g %g\n", gfac.k[i], gfac.Dc[i], gfac.Db[i], gfac.Dn[i], gfac.gc[i], gfac.gb[i], gfac.gn[i]);
    }
    
    /* Free the results */
    free_growth_factors(&gfac);   
        
    // /* Also find the cdm density at z = 0 */
    // double d_c_0 = strooklat_interp_2d(&spline_a, &spline_k, d_ncdm_array, 1.0, k);
    // /* The power spectrum at this scale is */
    // const double A_s = 2.087599637e-09;
    // const double n_s = 0.97774;
    // const double k_pivot = 0.05;
    // double Pk = primordialPower(k, A_s, n_s, k_pivot) * d_c_0 * d_c_0;
    // printf("Pk = %g %g\n", Pk, d_c_0);
    
    /* Free the cosmological tables */
    free_cosmology_tables(&tab);
    
    /* Clean up */
    cleanParams(&pars);
    cleanPerturb(&ptdat);
    cleanPerturbParams(&ptpars);
        
    /* Timer */
    gettimeofday(&time_stop, NULL);
    long unsigned microsec = (time_stop.tv_sec - time_start.tv_sec) * 1000000
                           + time_stop.tv_usec - time_start.tv_usec;
    printf("\nTime elapsed: %.5f s\n", microsec/1e6);
}
