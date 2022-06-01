#ifndef SURROGATE_H_
#define SURROGATE_H_

#include <iostream>
#include <Eigen/Dense>
#include "LinearSolver.h"

void build_surrogate(double* points, double* f, int N, int d, double* lambda_c);

void evaluate_surrogate_gt( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output);

void evaluate_surrogate_unroll_8_sqrt_sample_vec_optimize_load( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output);

#endif