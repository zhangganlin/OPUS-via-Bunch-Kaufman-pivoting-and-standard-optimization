
#ifndef OPUS_H_
#define OPUS_H_


// CONSTANTS
#define OPUS_MAX_SIZE 100 // max swarm size
#define OPUS_INERTIA 0.7298 // default value of w (see clerc02)


// === INERTIA WEIGHT UPDATE FUNCTIONS ===
#define OPUS_W_CONST 0
#define OPUS_W_LIN_DEC 1

// OPUS SOLUTION -- Initialized by the user
typedef struct {

    double error;
    double *gbest; // should contain DIM elements!!

} opus_result_t;



// OBJECTIVE FUNCTION TYPE
typedef double (*opus_obj_fun_t)(double *, int, void *);



// PSO SETTINGS
typedef struct {

    int dim; // problem dimensionality
    double *range_lo; // lower range limit (array of length DIM)
    double *range_hi; // higher range limit (array of length DIM)
    double goal; // optimization goal (error threshold)

    int size; // swarm size (number of particles)
    int k_size; // number of space-filling design z
    int print_every; // ... N steps (set to 0 for no output)
    int steps; // maximum number of iterations
    int step; // current PSO step
    double c1; // cognitive coefficient
    double c2; // social coefficient
    double w_max; // max inertia weight value
    double w_min; // min inertia weight value

} opus_settings_t;

opus_settings_t *opus_settings_new(int dim, double range_lo, double range_hi);
void opus_settings_free(opus_settings_t *settings);

// return the swarm size based on dimensionality
int opus_calc_swarm_size(int dim);


// minimize the provided obj_fun using OPUS with the specified settings
// and store the result in *solution
void opus_solve(opus_obj_fun_t obj_fun, void *obj_fun_params,
	       opus_result_t *solution, opus_settings_t *settings);

int double_compare(const void *a, const void *b){
    if (*(double*)a>*(double*)b) return -1;
    else return 1;
}

#endif // OPUS_H_
