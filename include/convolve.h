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

#ifndef CONVOLVE_H
#define CONVOLVE_H

#include "../include/fluid_equations.h"

void convolve(int N, double boxlen, const double *phi, double *out,
              struct growth_factors_2 *gfac2, double k_cutoff,
              double D2_asymp, int X_min, int X_max, int verbose);
void convolve_fft(int N, double boxlen, const double *phi, double *out);

#endif
