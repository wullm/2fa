/*******************************************************************************
 * This file is part of 3FA.
 * Copyright (c) 2020 Willem Elbers (whe@willemelbers.com)
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

#include <math.h>
#include "../include/primordial.h"

/* The bare primordial power spectrum, without transfer functions */
/* This is k^4 * P(k); the transfer functions are -T(k)/k^2 */
double primordialPower(double k, double A_s, double n_s, double k_pivot) {
    if (k == 0) return 0;

    return A_s * pow(k/k_pivot, n_s - 1.) * k * (2. * M_PI * M_PI);
}
