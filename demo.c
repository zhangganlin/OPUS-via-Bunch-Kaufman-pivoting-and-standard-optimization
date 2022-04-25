/* An implementation of the Particle Swarm Optimization algorithm

   === DEMONSTRATION CODE ===

   Copyright 2010 Kyriakos Kentzoglanakis

   This program is free software: you can redistribute it and/or
   modify it under the terms of the GNU General Public License version
   3 as published by the Free Software Foundation.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see
   <http://www.gnu.org/licenses/>.
*/


#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h> // for printf
#include "opus.h"


//==============================================================
//                  BENCHMARK FUNCTIONS
//==============================================================

double opus_sphere(double *vec, int dim, void *params) {

    double sum = 0;
    int i;
    for (i=0; i<dim; i++)
        sum += pow(vec[i], 2);

    return sum;
}



double opus_rosenbrock(double *vec, int dim, void *params) {

    double sum = 0;
    int i;
    for (i=0; i<dim-1; i++)
        sum += 100 * pow((vec[i+1] - pow(vec[i], 2)), 2) +	\
            pow((1 - vec[i]), 2);

    return sum;

}


double opus_griewank(double *vec, int dim, void *params) {

    double sum = 0.;
    double prod = 1.;
    int i;
    for (i=0; i<dim;i++) {
        sum += pow(vec[i], 2);
        prod *= cos(vec[i] / sqrt(i+1));
    }

    return sum / 4000 - prod + 1;

}


//==============================================================

int main(int argc, char **argv) {

    opus_settings_t *settings = NULL;
    opus_obj_fun_t obj_fun = NULL;

    // parse command line argument (function name)
    if (argc == 2) {
        if (strcmp(argv[1], "rosenbrock") == 0) {
            obj_fun = opus_rosenbrock;
            settings = opus_settings_new(30, -2.048, 2.048);
            printf("Optimizing function: rosenbrock (dim=%d, swarm size=%d)\n",
                   settings->dim, settings->size);
        } else if (strcmp(argv[1], "griewank") == 0) {
            obj_fun = opus_griewank;
            settings = opus_settings_new(30, -600, 600);
            printf("Optimizing function: griewank (dim=%d, swarm size=%d)\n",
                   settings->dim, settings->size);
        } else if (strcmp(argv[1], "sphere") == 0) {
            obj_fun = opus_sphere;
            settings = opus_settings_new(30, -100, 100);
            printf("Optimizing function: sphere (dim=%d, swarm size=%d)\n",
                   settings->dim, settings->size);
        } else {
            printf("Unsupported objective function: %s", argv[1]);
            return 1;
        }
    } else if (argc > 2) {
        printf("Usage: demo [PROBLEM], where problem is optional with values [sphere|rosenbrock|griewank]\n ");
        return 1;
    }

    // handle the default case (no argument given)
    if (obj_fun == NULL || settings == NULL) {
        obj_fun = opus_sphere;
        settings = opus_settings_new(30, -100, 100);
        printf("Optimizing function: sphere (dim=%d, swarm size=%d)\n",
                   settings->dim, settings->size);
    }

    // set some general PSO settings
    settings->goal = 1e-5;
    settings->size = 30;
    settings->w_strategy = OPUS_W_LIN_DEC;

    // initialize GBEST solution
    opus_result_t solution;
    // allocate memory for the best position buffer
    solution.gbest = (double *)malloc(settings->dim * sizeof(double));

    // run optimization algorithm
    opus_solve(obj_fun, NULL, &solution, settings);

    // free the gbest buffer
    free(solution.gbest);

    // free the settings
    opus_settings_free(settings);

    return 0;

}
