#ifndef SURROGATE_H_
#define SURROGATE_H_

#include <iostream>
#include <Eigen/Dense>
#include "LinearSolver.h"

void build_surrogate(double* points, double* f, int N, int d, double* lambda_c);

double evaluate_surrogate( double* x,  double* points,  double* lambda_c, int N, int d);

void evaluate_surrogate_batch( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output);

#endif