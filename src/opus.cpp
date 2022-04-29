
#include <stdlib.h> // for rand() stuff
#include <stdio.h> // for printf
#include <time.h> // for time()
#include <math.h> // for cos(), pow(), sqrt() etc.
#include <float.h> // for DBL_MAX
#include <string.h> // for mem*

#include "opus.h"

#include "randomlhs.hpp"

// generates a double between (0, 1)
#define RNG_UNIFORM() (rand()/(double)RAND_MAX)

// generate an int between 0 and s (exclusive)
#define RNG_UNIFORM_INT(s) (rand()%s)

// function type for the different inertia calculation functions
typedef double (*inertia_fun_t)(int step, opus_settings_t *settings);



int fz_compare(const void *a, const void *b){
    if (*(double*)a>*(double*)b) return 1;
    else return -1;
}

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
// create opus settings
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
    settings->k_size = settings->size*2;
    settings->r = 10;
    settings->delta = (range_hi-range_lo)/100.0;

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

double ** opus_matrix_extend(int old_size, int dim, double** matrix) {
    matrix = (double **)realloc(matrix,2*old_size * sizeof(double *));
    for (int i=old_size; i<2*old_size; i++) {
        matrix[i] = (double *)malloc(dim * sizeof(double));
    }
    return matrix;
}

void opus_matrix_free(double **m, int size) {
    for (int i=0; i<size; i++) {
        free(m[i]);
    }
    free(m);
}


//==============================================================
//                     OPUS ALGORITHM
//==============================================================
void opus_solve(opus_obj_fun_t obj_fun, void *obj_fun_params,
	       opus_result_t *solution, opus_settings_t *settings)
{
    // Particles
    double **pos_z = opus_matrix_new(settings->k_size, settings->dim);
    double **pos = opus_matrix_new(settings->size, settings->dim); // position matrix
    double **temp_pos = opus_matrix_new(settings->r, settings->dim);
    double **temp_vel = opus_matrix_new(settings->r, settings->dim);
    double **vel = opus_matrix_new(settings->size, settings->dim); // velocity matrix
    double **pos_b = opus_matrix_new(settings->size, settings->dim); // best position matrix
    fz_t *fit_z = (fz_t *)malloc(settings->k_size * sizeof(fz_t));
    double *fit = (double *)malloc(settings->size * sizeof(double));
    double *fit_b = (double *)malloc(settings->size * sizeof(double));
    double *temp_result = (double *)malloc(settings->r * sizeof(double));
    double *x_optimized = (double *)malloc(settings->dim * sizeof(double));

    int epsilon_size = 100;
    double **epsilon = opus_matrix_new(epsilon_size,settings->dim); // may need to extend the size, remember to also update epsilon size!!!!! 
    int valid_epsilon_size;


    int i, d, step, l, temp_idx, j;
    double u;
    double temp_res_max;
    double min_dist, temp_dist;
    double f_opt;
    double rho1, rho2; // random numbers (coefficients)
    // initialize omega using standard value
    double w = OPUS_INERTIA;
    inertia_fun_t calc_inertia_fun = NULL; // inertia weight update function

    // initialize random seed
    srand(time(NULL));

    // SELECT APPROPRIATE INERTIA WEIGHT UPDATE FUNCTION
    calc_inertia_fun = calc_inertia_lin_dec;

    // INITIALIZE SOLUTION
    solution->error = DBL_MAX;

    // Step 1-4 ------------------------------------------------------------------------------------
    // SWARM INITIALIZATION
    // for each particle
    randomLHS(settings->k_size,settings->dim,pos_z,*settings->range_lo,*settings->range_hi);
    for(i=0; i<settings->k_size;i++){
        fit_z[i] = (fz_t){obj_fun(pos_z[i], settings->dim, obj_fun_params),i};
    }
    qsort(fit_z,settings->k_size,sizeof(fz_t),fz_compare); // fit_z[0] with smallest f value


    for (i=0; i<settings->size; i++) {
        // for each dimension
        for (d=0; d<settings->dim; d++) {
            // generate one number within the specified range
            u = settings->range_lo[d] + (settings->range_hi[d] - settings->range_lo[d]) * \
                RNG_UNIFORM();
            // initialize position
            pos[i][d] = pos_z[fit_z[i].index][d];
            // best position is the same
            pos_b[i][d] = pos_z[fit_z[i].index][d];
            // initialize velocity
            vel[i][d] = (u-pos[i][d]) / 2.;
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
    valid_epsilon_size = 0;
    //------------------------------------------------------------------------------------------------


    // RUN ALGORITHM
    for (step=0; step<settings->steps; step++) {
        // update current step
        settings->step = step;
        // update inertia weight
        // do not bother with calling a calc_w_const function
        if (calc_inertia_fun != NULL) {
            w = calc_inertia_fun(step, settings);   // i(t) in the paper
        }
        // check optimization goal
        if (solution->error <= settings->goal) {
            // SOLVED!!
            if (settings->print_every)
                printf("Goal achieved @ step %d (error=%.3e) :-)\n", step, solution->error);
            break;
        }

        // step 5: fit surrogate--------------------------------------------------------------
        // TODO


        // -----------------------------------------------------------------------------------


        // update all particles
        for (i=0; i<settings->size; i++) {
            // step 6-----------------------------------------------------------------------------
            // 6a
            for(l=0; l<settings->r; l++){
                for (d=0; d<settings->dim; d++) {
                    // calculate stochastic coefficients
                    rho1 = settings->c1 * RNG_UNIFORM();
                    rho2 = settings->c2 * RNG_UNIFORM();
                    // update velocity
                    temp_vel[l][d] = w * vel[i][d] +	\
                        rho1 * (pos_b[i][d] - pos[i][d]) +	\
                        rho2 * (solution->gbest[d] - pos[i][d]);
                    // update position
                    temp_pos[l][d] = pos[i][d] + temp_vel[l][d];
                    // clamp position within bounds
                    if (temp_pos[l][d] < settings->range_lo[d]) {
                        temp_pos[l][d] = settings->range_lo[d];
                        temp_vel[l][d] = 0;
                    } else if (temp_pos[l][d] > settings->range_hi[d]) {
                        temp_pos[l][d] = settings->range_hi[d];
                        temp_pos[l][d] = 0;
                    }
                }
            }
            //6b TODO need to fuse the surrogate code
            // // using surrogate model here
            // surrogate_model(temp_pos,temp_result);
            temp_idx = 0;
            temp_res_max = temp_result[temp_idx];
            for(l = 0; l < settings->r; l++){
                if(temp_result[l]>temp_res_max){
                    temp_idx = l;
                    temp_res_max = temp_result[l];
                }
            }
            memmove((void *)pos[i], (void *)temp_pos[temp_idx],
                        sizeof(double) * settings->dim);
            memmove((void *)vel[i], (void *)temp_vel[temp_idx],
                        sizeof(double) * settings->dim);
            // -----------------------------------------------------------------------------------


            // step 7-8 ---------------------------------------------------------------------------
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
            // -----------------------------------------------------------------------------------


            // step 9 Refit surrogate-------------------------------------------------------------
            // TODO

            // -----------------------------------------------------------------------------------


            // step 10----------------------------------------------------------------------------
            // TODO

            // -----------------------------------------------------------------------------------

            // step 11----------------------------------------------------------------------------
            min_dist = 0;
            for(j = 0; j < valid_epsilon_size; j++){
                temp_dist = 0;
                for(d = 0; d < settings->dim; d++){
                    temp_dist += (x_optimized[d] - epsilon[j][d])*(x_optimized[d] - epsilon[j][d]);
                }
                min_dist = min_dist<temp_dist? min_dist:temp_dist;
            }

            if(min_dist<settings->delta*settings->delta){
                f_opt = obj_fun(x_optimized,settings->dim,obj_fun_params);
                if(f_opt<solution->error){
                    solution->error = f_opt;
                    memmove((void *)solution->gbest, (void *)x_optimized,
                        sizeof(double) * settings->dim);
                }
                if(valid_epsilon_size>=epsilon_size){
                    epsilon = opus_matrix_extend(epsilon_size,settings->dim,epsilon);
                    epsilon_size += epsilon_size;
                }
                memmove((void *)epsilon[valid_epsilon_size], (void *)x_optimized,
                        sizeof(double) * settings->dim);
                valid_epsilon_size ++;
            }
            
            // -----------------------------------------------------------------------------------


        }

        if (settings->print_every && (step % settings->print_every == 0))
            printf("Step %d (w=%.2f) :: min err=%.5e\n", step, w, solution->error);

    }

    // free resources
    opus_matrix_free(pos_z, settings->size);
    opus_matrix_free(pos, settings->size);
    opus_matrix_free(vel, settings->size);
    opus_matrix_free(pos_b, settings->size);
    opus_matrix_free(epsilon, epsilon_size);
    opus_matrix_free(temp_pos,settings->r);
    opus_matrix_free(temp_vel,settings->r);


    free(fit_z);
    free(fit);
    free(fit_b);
    free(temp_result);
    free(x_optimized);
}