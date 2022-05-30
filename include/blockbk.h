#ifndef BLOCKBK_H
#define BLOCKBK_H


#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

void permute(double* A, int* P, int b_row_start, int b_row_end, int b_col_start, int b_col_end, int n, int dim);

void BunchKaufman_noblock(double* A, double* L, int* P, int* pivot, int M);

void BunchKaufman_subblock(double* A, double* L, int* P, int* pivot, int* pivot_idx, int M, int b_start, int b_size);

void BunchKaufman_block(double* A, double* L, int* P, int* pivot, int n, int r);

void block_solve_XLtB_inplace_L(double* A, double* L, int b_col, int b_size, int b_row, int n_row, int n);

void block_solve_column_LD(double* A, double* L, int b_col, int b_size, int n);


/* 
	L saves LD 
	X * D = B
	D[i,j] = D[b_col*b_size+i, b_col*b_size+j]
	X[i,j] = L[b_row*b_size+i, b_col*b_size+j]
	b_row > b_col

	result save in L
*/
void block_solve_XDB(double* L, double* D, int b_col, int b_size, int b_row, int n_row, int n, int* pivot);

void block_solve_column_L(double* L, double* D, int b_col, int b_size, int n, int* pivot);

void block_solve_column(double* A, double* L, double* D,int b_col, int b_size, int n, int* pivot);



//k = r, 2r, ... (n // r) * r
void matrix_update(double* mat_A, double* mat_D, double* mat_L, int* vec_ind, int n, int k, int r);

#endif