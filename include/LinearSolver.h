#ifndef LINEARSOLVER_H_
#define LINEARSOLVER_H_

#include <math.h>
#include <stdio.h>


void LUdecomp(double* A, double* b, double* sol, int N, int func_dim);

void BunchKaufman(double* A, double* L, int* P, int* pivot, int M);

void solve_lower(double* L, double* x, double* b, int n);

void solve(double*A, double*x, double*b, int n);

#endif