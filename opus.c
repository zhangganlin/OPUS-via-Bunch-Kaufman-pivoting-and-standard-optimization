/* An implementation of the Particle Swarm Optimization algorithm

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


#include <stdlib.h> // for rand() stuff
#include <stdio.h> // for printf
#include <time.h> // for time()
#include <math.h> // for cos(), pow(), sqrt() etc.
#include <float.h> // for DBL_MAX
#include <string.h> // for mem*

#include "opus.h"

// generates a double between (0, 1)
#define RNG_UNIFORM() (rand()/(double)RAND_MAX)

// generate an int between 0 and s (exclusive)
#define RNG_UNIFORM_INT(s) (rand()%s)

// function type for the different inertia calculation functions
typedef double (*inertia_fun_t)(int step, opus_settings_t *settings);


//==============================================================
// calulate swarm size based on dimensionality
int opus_calc_swarm_size(int dim) {
    int size = 10. + 2. * sqrt(dim);
    return (size > OPUS_MAX_SIZE ? OPUS_MAX_SIZE : size);
}


//==============================================================
//          INERTIA WEIGHT UPDATE STRATEGIES
//==============================================================
// calculate linearly decreasing inertia weight
double calc_inertia_lin_dec(int step, opus_settings_t *settings) {

    int dec_stage = 3 * settings->steps / 4;
    if (step <= dec_stage)
        return settings->w_min + (settings->w_max - settings->w_min) *	\
            (dec_stage - step) / dec_stage;
    else
        return settings->w_min;
}


//==============================================================
// create pso settings
opus_settings_t *opus_settings_new(int dim, double range_lo, double range_hi) {
    opus_settings_t *settings = (opus_settings_t *)malloc(sizeof(opus_settings_t));
    if (settings == NULL) { return NULL; }

    // set some default values
    settings->dim = dim;
    settings->goal = 1e-5;

    // set up the range arrays
    settings->range_lo = (double *)malloc(settings->dim * sizeof(double));
    if (settings->range_lo == NULL) { free(settings); return NULL; }

    settings->range_hi = (double *)malloc(settings->dim * sizeof(double));
    if (settings->range_hi == NULL) { free(settings); free(settings->range_lo); return NULL; }

    for (int i=0; i<settings->dim; i++) {
        settings->range_lo[i] = range_lo;
        settings->range_hi[i] = range_hi;
    }

    settings->size = opus_calc_swarm_size(settings->dim);
    settings->print_every = 1000;
    settings->steps = 100000;
    settings->c1 = 1.496;
    settings->c2 = 1.496;
    settings->w_max = OPUS_INERTIA;
    settings->w_min = 0.3;

    settings->w_strategy = OPUS_W_LIN_DEC;

    return settings;
}

// destroy OPUS settings
void opus_settings_free(opus_settings_t *settings) {
    free(settings->range_lo);
    free(settings->range_hi);
    free(settings);
}


double **opus_matrix_new(int size, int dim) {
    double **m = (double **)malloc(size * sizeof(double *));
    for (int i=0; i<size; i++) {
        m[i] = (double *)malloc(dim * sizeof(double));
    }
    return m;
}

void opus_matrix_free(double **m, int size) {
    for (int i=0; i<size; i++) {
        free(m[i]);
    }
    free(m);
}


//==============================================================
//                     PSO ALGORITHM
//==============================================================
void opus_solve(opus_obj_fun_t obj_fun, void *obj_fun_params,
	       opus_result_t *solution, opus_settings_t *settings)
{
    // Particles
    double **pos = opus_matrix_new(settings->size, settings->dim); // position matrix
    double **vel = opus_matrix_new(settings->size, settings->dim); // velocity matrix
    double **pos_b = opus_matrix_new(settings->size, settings->dim); // best position matrix
    double *fit = (double *)malloc(settings->size * sizeof(double));
    double *fit_b = (double *)malloc(settings->size * sizeof(double));

    int i, d, step;
    double a, b; // for matrix initialization
    double rho1, rho2; // random numbers (coefficients)
    // initialize omega using standard value
    double w = OPUS_INERTIA;
    inertia_fun_t calc_inertia_fun = NULL; // inertia weight update function

    // initialize random seed
    srand(time(NULL));

    // SELECT APPROPRIATE INERTIA WEIGHT UPDATE FUNCTION
    switch (settings->w_strategy)
        {
            /* case PSO_W_CONST : */
            /*     calc_inertia_fun = calc_inertia_const; */
            /*     break; */
        case OPUS_W_LIN_DEC :
            calc_inertia_fun = calc_inertia_lin_dec;
            break;
        }

    // INITIALIZE SOLUTION
    solution->error = DBL_MAX;

    // SWARM INITIALIZATION
    // for each particle
    for (i=0; i<settings->size; i++) {
        // for each dimension
        for (d=0; d<settings->dim; d++) {
            // generate two numbers within the specified range
            a = settings->range_lo[d] + (settings->range_hi[d] - settings->range_lo[d]) * \
                RNG_UNIFORM();
            b = settings->range_lo[d] + (settings->range_hi[d] - settings->range_lo[d]) *	\
                RNG_UNIFORM();
            // initialize position
            pos[i][d] = a;
            // best position is the same
            pos_b[i][d] = a;
            // initialize velocity
            vel[i][d] = (a-b) / 2.;
        }
        // update particle fitness
        fit[i] = obj_fun(pos[i], settings->dim, obj_fun_params);
        fit_b[i] = fit[i]; // this is also the personal best
        // update gbest??
        if (fit[i] < solution->error) {
            // update best fitness
            solution->error = fit[i];
            // copy particle pos to gbest vector
            memmove((void *)solution->gbest, (void *)pos[i],
                    sizeof(double) * settings->dim);
        }

    }

    // RUN ALGORITHM
    for (step=0; step<settings->steps; step++) {
        // update current step
        settings->step = step;
        // update inertia weight
        // do not bother with calling a calc_w_const function
        if (calc_inertia_fun != NULL) {
            w = calc_inertia_fun(step, settings);
        }
        // check optimization goal
        if (solution->error <= settings->goal) {
            // SOLVED!!
            if (settings->print_every)
                printf("Goal achieved @ step %d (error=%.3e) :-)\n", step, solution->error);
            break;
        }

        // the value of improved was just used; reset it

        // update all particles
        for (i=0; i<settings->size; i++) {
            // for each dimension
            for (d=0; d<settings->dim; d++) {
                // calculate stochastic coefficients
                rho1 = settings->c1 * RNG_UNIFORM();
                rho2 = settings->c2 * RNG_UNIFORM();
                // update velocity
                vel[i][d] = w * vel[i][d] +	\
                    rho1 * (pos_b[i][d] - pos[i][d]) +	\
                    rho2 * (solution->gbest[d] - pos[i][d]);
                // update position
                pos[i][d] += vel[i][d];
                // clamp position within bounds
                if (pos[i][d] < settings->range_lo[d]) {
                    pos[i][d] = settings->range_lo[d];
                    vel[i][d] = 0;
                } else if (pos[i][d] > settings->range_hi[d]) {
                    pos[i][d] = settings->range_hi[d];
                    vel[i][d] = 0;
                }
                

            }

            // update particle fitness
            fit[i] = obj_fun(pos[i], settings->dim, obj_fun_params);
            // update personal best position?
            if (fit[i] < fit_b[i]) {
                fit_b[i] = fit[i];
                // copy contents of pos[i] to pos_b[i]
                memmove((void *)pos_b[i], (void *)pos[i],
                        sizeof(double) * settings->dim);
            }
            // update gbest??
            if (fit[i] < solution->error) {
                // update best fitness
                solution->error = fit[i];
                // copy particle pos to gbest vector
                memmove((void *)solution->gbest, (void *)pos[i],
                        sizeof(double) * settings->dim);
            }
        }

        if (settings->print_every && (step % settings->print_every == 0))
            printf("Step %d (w=%.2f) :: min err=%.5e\n", step, w, solution->error);

    }

    // free resources
    opus_matrix_free(pos, settings->size);
    opus_matrix_free(vel, settings->size);
    opus_matrix_free(pos_b, settings->size);
    free(fit);
    free(fit_b);
}
