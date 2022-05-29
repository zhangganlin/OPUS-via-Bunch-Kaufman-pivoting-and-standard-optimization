// #pragma once

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <iostream>
#include <fstream>
#include <sstream>

#include <iomanip>

using namespace std;

void read_matrix(double* A, int m, int n, string path);

void generate_random_dense(double* arr, int m, int n);

void generate_random_l(double* arr, int n);

void generate_random_symmetry(double* arr, int n);


void generate_random_d(double* arr, int n, int* pivot);

// A: m*n, B:n*p
void matrix_mul(double*A, double*B, double* res, int m, int n, int p);


void matrix_transpose(double*A, double*At, int n);

void matrix_transpose(double*A, double*At, int m, int n);
//A[start_row:start_row+n_row, start_col:start_col+n_col] = x
void matrix_set_block(double*A, int start_row, int start_col, int n_row, int n_col, double*x, int n);

//x = A[start_row:start_row+n_row, start_col:start_col+n_col]
void matrix_get_block(double*A, int start_row, int start_col, int n_row, int n_col, double*x, int n);


void print_matrix(double* A, int m, int n, int precision);

void print_vector(int* b, int n);

#endif