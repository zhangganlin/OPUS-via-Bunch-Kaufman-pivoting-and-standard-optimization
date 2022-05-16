#ifndef LINEARSOLVER_H_
#define LINEARSOLVER_H_

#include <math.h>
#include <stdio.h>


void LUdecomp(double* A, double* b, double* sol, int N, int func_dim);

void BunchKaufman(double* A, double* L, int* P, int* pivot, int M);

void solve_lower(double* L, double* x, double* b, int n);

void solve_upper(double* L, double* x, double* b, int n);

void solve_diag(double* D, int* P, double* x, double* b, int n);

void solve_BunchKaufman(double* A, double* x, double* b, int n);

#endif