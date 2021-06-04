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

#ifndef WRITE_TRANSFERS_H
#define WRITE_TRANSFERS_H

#include "../include/perturb_data.h"
#include "../include/input.h"
#include "../include/fluid_equations.h"

void write_transfer_functions(struct model *m, struct units *us,
                              struct cosmology_tables *tab,
                              struct perturb_data *ptdat,
                              struct growth_factors *gfac,
                              double a_start, double a_final,
                              char fname[100]);

#endif
