#ifndef LINEARSOLVER_H_
#define LINEARSOLVER_H_

#include <math.h>
#include <stdio.h>
#include "blockbk.h"

void solve_lower(double* L, double* x, double* b, int n);

void solve_upper(double* L, double* x, double* b, int n);

void solve_diag(double* D, int* P, double* x, double* b, int n);

void solve_BunchKaufman(double* A, double* x, double* b, int n);

#endif