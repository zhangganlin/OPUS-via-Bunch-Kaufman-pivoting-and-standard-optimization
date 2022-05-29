#include <iostream>
#include <cmath>
#include <cstring>
#include "test_utils.h"
#include "blockbk.h"


using namespace std;

void block_solve_XLtB_inplace_L(double* A, double* L, int b_col, int b_size, int b_row, int n_row, int n){
	for(int k = 0; k < n_row; k++){
		double s;
		for(int i = 0; i < b_size; i++){
			s = 0;
			for(int j = 0; j < i; j++) {
				s = s + L[(b_col*b_size+i) * n + b_col*b_size+j] * L[(b_row*b_size+k)*n+b_col*b_size+j];
			}
			L[(b_row*b_size+k)*n+b_col*b_size+i] = (A[(b_row*b_size+k)*n+b_col*b_size+i] - s) ;
		}
	}
}

void block_solve_column_LD(double* A, double* L, int b_col, int b_size, int n){
	int b_row;
	for(b_row = b_col+1; (b_row+1)*b_size < n; b_row++){
		block_solve_XLtB_inplace_L(A,L,b_col,b_size,b_row,b_size,n);
	}
	if (b_row*b_size == n){
		return;
	}else{
		int remaining_row = n - b_row*b_size;
		block_solve_XLtB_inplace_L(A,L,b_col,b_size,b_row,remaining_row,n);
	}

}


/* 
	L saves LD 
	X * D = B
	D[i,j] = D[b_col*b_size+i, b_col*b_size+j]
	X[i,j] = L[b_row*b_size+i, b_col*b_size+j]
	b_row > b_col

	result save in L
*/
void block_solve_XDB(double* L, double* D, int b_col, int b_size, int b_row, int n_row, int n, int* pivot){
	int i;
    double s;
	for(int k = 0; k < n_row; k++){
		for(i = 0; i < b_size; i++){
			if(pivot[b_col*b_size+i]==1){
				if(D[(b_col*b_size+i)*n + b_col*b_size+i]==0){
					printf("D is singular!\n");
				}
				L[(b_row*b_size+k)*n + b_col*b_size+i] = L[(b_row*b_size+k)*n + b_col*b_size+i]/
														 D[(b_col*b_size+i)*n + b_col*b_size+i];
				// X[k*n + i] = B[k*n + i]/D[i*n+i];
			}
			else if(pivot[b_col*b_size+i]==2){
				double a = D[(b_col*b_size+i)*n + b_col*b_size+i];
				double tb = D[(b_col*b_size+i)*n + b_col*b_size+i+1];
				double c = D[(b_col*b_size+i+1)*n + b_col*b_size+i+1];
				double d1 = L[(b_row*b_size+k)*n + b_col*b_size+i];
				double d2 = L[(b_row*b_size+k)*n + b_col*b_size+i+1];

				if((c*a-tb*tb)==0){
					printf("D is singular!\n");
				}

				L[(b_row*b_size+k)*n + b_col*b_size+i] = (c*d1-tb*d2)/(c*a-tb*tb);
				L[(b_row*b_size+k)*n + b_col*b_size+i+1] = (d1*tb-a*d2)/(tb*tb-a*c);
				
			}else if(pivot[b_col*b_size+i]==0){
				continue;
			}
		}
	}
}

void block_solve_column_L(double* L, double* D, int b_col, int b_size, int n, int* pivot){
	int b_row;
	for(b_row = b_col+1; (b_row+1)*b_size < n; b_row++){
		block_solve_XDB(L,D,b_col,b_size,b_row,b_size,n,pivot);
	}
	if (b_row*b_size == n){
		return;
	}else{
		int remaining_row = n - b_row*b_size;
		block_solve_XDB(L,D,b_col,b_size,b_row,remaining_row,n,pivot);
	}
}

void block_solve_column(double* A, double* L, double* D,int b_col, int b_size, int n, int* pivot){
	block_solve_column_LD(A,L,b_col,b_size,n);
	block_solve_column_L(L,D,b_col,b_size,n,pivot);
}



//k = r, 2r, ... (n // r) * r
void matrix_update(double* mat_A, double* mat_D, double* mat_L, int* vec_ind, int n, int k, int r){
    for(int j = k; j < n; j += r){
        for(int i = j; i < n; i += r){
            int remaining_row_num = min(r, n-i);
            int remaining_col_num = min(r, n-j);
            for(int s = 0; s < remaining_row_num; s++){
                for(int t = 0; t < remaining_col_num; t++){
                    double d_0 = 0 , d_n1 = 0 , d_1  = 0;
                    for(int l = 0; l < r; l++){
                        //sum_{l}(L[t,l] * d[ll] * L[t,l])
                        d_0 += mat_L[(i+s)*n + k-r+l] * mat_D[(l+k-r)*n + l+k-r] * mat_L[(j+t)*n + k-r+l];
                    }
                    for(int u = 0; u < r - 1; u++){
                        //sum_{u}(L[s,u+1] * d[u+1,u] * L[t,u])
                        d_n1 += mat_L[(i+s)*n + k-r+u+1] * mat_D[(u+1+k-r)*n + u+k-r] * mat_L[(j+t)*n + k-r+u];
                    }
                    for(int p = 1; p < r; p++){
                        //sum_{p}(L[s,p-1] * d[p-1,p] * L[t,p])
                        d_1 += mat_L[(i+s)*n + k-r+p-1] * mat_D[(p-1+k-r)*n + p+k-r] * mat_L[(j+t)*n + k-r+p];
                    }
                    
                    mat_A[(i+s)*n + (j+t)] = mat_A[(i+s)*n + (j+t)] - d_0 - d_n1 - d_1;
                }
            }
        }
    }
}



void permute(double* A, int* P, int b_row_start, int b_row_end, int b_col_start, int b_col_end, int n, int dim){
    // dim = 0 => P * A
    // dim = 1 => A * PT
    // P base 0
    b_row_end = min(b_row_end, n);
    int b_row_size = b_row_end - b_row_start, b_col_size = b_col_end - b_col_start;
    double* A_temp = (double*)malloc(b_row_size * b_col_size * sizeof(double));
    for(int i = 0; i < b_row_end - b_row_start; i++){
        memcpy(A_temp + i * b_col_size, A + (b_row_start + i)*n + b_col_start, b_col_size * sizeof(double));
    }
    if(dim == 0){
        for(int i = b_row_start; i < b_row_end; i++){
            for(int j = b_col_start; j < b_col_end; j++){
                A[i * n + j] = A_temp[(P[i] -  b_row_start) * b_col_size + j - b_col_start];
            }
        }
    } else {
        for(int i = b_row_start; i < b_row_end; i++){
            for(int j = b_col_start; j < b_col_end; j++){
                A[i * n + j] = A_temp[(i - b_row_start) * b_col_size + P[j] - b_col_start];
            }
        }
    }
    free(A_temp);
}

void BunchKaufman_noblock(double* A, double* L, int* P, int* pivot, int M){
    const double alpha = (1+sqrt(17))/8;
    int r, i, j, tmp_i;
    int k = 0;
    double w1, wr;
    double tmp_d, A_kk;
    double detE, invE_11, invE_22, invE_12, invE_21;
    //Initialize Matrices
    for (i = 0; i < M; i++) {
        for (j = 0; j < M; j++){
            if (j == i) {
                L[i * M + i] = 1.0;
            }
            else{
                L[i * M + j] = 0.0;
            }
        }
        P[i] = i+1;
        pivot[i] = 0.0;
    }
    while (k < M-1){
        w1 = 0.0;
        for (i = k + 1; i < M; i++){
            tmp_d = abs(A[i * M + k]);
            //find the column 1 max magnitude of subdiagonal
            if (w1 < tmp_d){
                w1 = tmp_d;
                r = i;
            }
        }
        if (abs(A[k * M + k]) >= alpha * w1){
            A_kk = A[k * M + k];
            for (i = k + 1; i < M; i++)  L[i * M + k] = A[i * M + k] / A_kk;
            for (i = k + 1; i < M; i++){
                tmp_d = A[i * M + k];
                for (j = i; j < M; j++)
                    A[j * M + i] -=  L[j * M + k] * tmp_d;
            }
            
            for (i = k+1; i < M; i++) A[i * M + k] = 0.0;
            for (i = k+1; i < M; i++){
                A[k * M + i] = 0.0;
            }
            pivot[k] = 1;
            k = k + 1;
        }
        else{
            wr = 0.0;
            //Step 4
            //max of off-diagonal in row r
            for (i = k; i < r; i++){
                tmp_d = abs(A[r * M + i]);
                if (tmp_d > wr) wr = tmp_d;
            }
            // max of off-dinagonal in row r
            if (r < M){
                for (i = r + 1; i < M; i++){
                    tmp_d = abs(A[i * M + r]);
                //find the column 1 max magnitude of subdiagonal
                    if (tmp_d > wr) wr = tmp_d;
                }
            }
            if (abs(A[k * M + k]) * wr >= alpha * w1 * w1){
                A_kk = A[k * M + k];
                for (i = k + 1; i < M; i++)  L[i * M + k] = A[i * M + k] / A_kk;
                for (i = k + 1; i < M; i++){
                    tmp_d = A[i * M + k];
                    for (j = i; j < M; j++)
                        A[j * M + i] -=  L[j * M + k] * tmp_d; //##TODO: need to be checked
                }

                for (i = k+1; i < M; i++) A[i * M + k] = 0.0;
                for (i = k+1; i < M; i++){
                     A[k * M + i] = 0.0;
                }
                pivot[k] = 1;
                k = k + 1;
            } else if (abs(A[r * M + r]) >= alpha * wr){
                tmp_i = P[k];
                P[k] = P[r];
                P[r] = tmp_i;
                tmp_d = A[k * M + k];
                A[k * M + k] = A[r * M + r];
                A[r * M + r] = tmp_d;
                for (i = r + 1; i < M; i++){
                    tmp_d = A[i * M + r];
                    A[i * M + r] = A[i * M + k];
                    A[i * M + k] = tmp_d;
                }
                for (i = k + 1; i < r; i++){
                    tmp_d = A[i * M + k];
                    A[i * M + k] = A[r * M + i];
                    A[r * M + i] = tmp_d;
                }
                if (k > 0){
                    for (j = 0; j < k; j++){
                        tmp_d = L[k * M + j];
                        L[k * M + j] = L[r * M + j];
                        L[r * M + j] = tmp_d;
                    }
                }
                A_kk = A[k * M + k];
                for (i = k + 1; i < M; i++)  L[i * M + k] = A[i * M + k] / A_kk;
                for (i = k + 1; i < M; i++){
                    tmp_d = A[i * M + k];
                    for (j = i; j < M; j++){
                        A[j * M + i] -=  L[j * M + k] * tmp_d; //##TODO: need to be checked
                    }
                }
                
                for (i = k+1; i < M; i++){
                     A[i * M + k] = 0.0;
                }
                for (i = k+1; i < M; i++){
                     A[k * M + i] = 0.0;
                }
                pivot[k] = 1;
                k = k + 1;
            } else {
                tmp_i = P[k+1];
                P[k+1] = P[r];
                P[r] = tmp_i;
                tmp_d = A[(k+1) * M + (k+1)];
                A[(k+1) * M + (k+1)] = A[r * M + r];
                A[r * M + r] = tmp_d;
                for (i = r + 1; i < M; i++){
                    tmp_d = A[i * M + k+1];
                    A[i * M + k+1] = A[i * M + r];
                    A[i * M + r] = tmp_d;
                }
                tmp_d = A[(k+1) * M + k];
                A[(k+1) * M + k] = A[r * M + k];
                A[r * M + k] = tmp_d;
                for (i = k+2; i < r; i++){
                    tmp_d = A[i * M + k+1];
                    A[i * M + k+1] = A[r * M + i];
                    A[r * M + i] = tmp_d;
                }
                if (k > 0){
                    for (i = 0; i < k; i++){
                        tmp_d = L[(k+1) * M + i];
                        L[(k+1) * M + i] = L[r * M + i];
                        L[r * M + i] = tmp_d;
                    }
                }

                detE = A[k * M + k] * A[(k+1) * M + k+1] - A[(k+1) * M + k] * A[(k+1) * M + k];
                invE_11 = A[(k+1) * M + k+1] / detE;
                invE_22 = A[k * M + k] / detE;
                invE_12 = - A[(k+1) * M + k] / detE;
                invE_21 = invE_12;
                for (i = k + 2; i < M; i++){
                    L[i * M + k] = A[i * M + k] * invE_11 + A[i * M + k+1] * invE_21;
                    L[i * M + k+1] = A[i * M + k] * invE_12 + A[i * M + k+1] * invE_22;
                }

                for (j = k+2; j < M; j++){
                    for(i = j; i < M ; i++){
                        A[i * M + j] -= L[i * M + k] * A[j * M + k] + L[i * M + k+1] * A[j * M + k+1];
                    }
                }
                for (i = k+2; i < M; i++){
                    A[i * M + k] = 0.0;
                    A[i * M + k+1] = 0.0;
                }
                for (i = k+2; i < M; i++){
                     A[k * M + i] = 0.0;
                     A[(k+1) * M + i] = 0.0;
                 }
                pivot[k] = 2;
                k = k + 2;

                for (i = 0; i < M-1; i++){
                    A[i*M + i+1] = A[(i+1)*M +i];
                }
            }
        }
    }
    if (pivot[M-1] == 0)
        if(pivot[M-2] != 2)
            pivot[M-1] = 1;
}

// we change the permutation to base 0
void BunchKaufman_subblock(double* A, double* L, int* P, int* pivot, int M, int b_start, int b_size){
    // [b_start, b_end) interval
    int b_end = min(b_start + b_size, M);
    b_size = b_end - b_start;

    const double alpha = (1+sqrt(17))/8;
    int r, i, j, tmp_i;
    int k = b_start;
    double w1, wr;
    double tmp_d, A_kk;
    double detE, invE_11, invE_22, invE_12, invE_21;
    //Initialize Matrices
    for (i = b_start; i < b_end; i++) {
        for (j = b_start; j < b_end; j++){
            if (j == i) {
                L[i * M + i] = 1.0;
            }
            else{
                L[i * M + j] = 0.0;
            }
        }
        P[i] = i;
        pivot[i] = 0.0;
    }
    while (k < b_end-1){
        w1 = 0.0;
        for (i = k + 1; i < b_end; i++){
            tmp_d = abs(A[i * M + k]);
            //find the column 1 max magnitude of subdiagonal
            if (w1 < tmp_d){
                w1 = tmp_d;
                r = i;
            }
        }
        if (abs(A[k * M + k]) >= alpha * w1){
            A_kk = A[k * M + k];
            for (i = k + 1; i < b_end; i++)  L[i * M + k] = A[i * M + k] / A_kk;
            for (i = k + 1; i < b_end; i++){
                tmp_d = A[i * M + k];
                for (j = i; j < b_end; j++)
                    A[j * M + i] -=  L[j * M + k] * tmp_d;
            }
            
            for (i = k+1; i < b_end; i++) A[i * M + k] = 0.0;
            for (i = k+1; i < b_end; i++){
                A[k * M + i] = 0.0;
            }
            pivot[k] = 1;
            k = k + 1;
        }
        else{
            wr = 0.0;
            //Step 4
            //max of off-diagonal in row r
            for (i = k; i < r; i++){
                tmp_d = abs(A[r * M + i]);
                if (tmp_d > wr) wr = tmp_d;
            }
            // max of off-dinagonal in row r
            if (r < M){
                for (i = r + 1; i < b_end; i++){
                    tmp_d = abs(A[i * M + r]);
                //find the column 1 max magnitude of subdiagonal
                    if (tmp_d > wr) wr = tmp_d;
                }
            }
            if (abs(A[k * M + k]) * wr >= alpha * w1 * w1){
                A_kk = A[k * M + k];
                for (i = k + 1; i < b_end; i++)  L[i * M + k] = A[i * M + k] / A_kk;
                for (i = k + 1; i < b_end; i++){
                    tmp_d = A[i * M + k];
                    for (j = i; j < b_end; j++)
                        A[j * M + i] -=  L[j * M + k] * tmp_d; //##TODO: need to be checked
                }

                for (i = k+1; i < b_end; i++) A[i * M + k] = 0.0;
                for (i = k+1; i < b_end; i++){
                     A[k * M + i] = 0.0;
                }
                pivot[k] = 1;
                k = k + 1;
            } else if (abs(A[r * M + r]) >= alpha * wr){
                tmp_i = P[k];
                P[k] = P[r];
                P[r] = tmp_i;
                tmp_d = A[k * M + k];
                A[k * M + k] = A[r * M + r];
                A[r * M + r] = tmp_d;
                for (i = r + 1; i < b_end; i++){
                    tmp_d = A[i * M + r];
                    A[i * M + r] = A[i * M + k];
                    A[i * M + k] = tmp_d;
                }
                for (i = k + 1; i < r; i++){
                    tmp_d = A[i * M + k];
                    A[i * M + k] = A[r * M + i];
                    A[r * M + i] = tmp_d;
                }
                if (k > 0){
                    for (j = b_start; j < k; j++){
                        tmp_d = L[k * M + j];
                        L[k * M + j] = L[r * M + j];
                        L[r * M + j] = tmp_d;
                    }
                }
                A_kk = A[k * M + k];
                for (i = k + 1; i < b_end; i++)  L[i * M + k] = A[i * M + k] / A_kk;
                for (i = k + 1; i < b_end; i++){
                    tmp_d = A[i * M + k];
                    for (j = i; j < b_end; j++){
                        A[j * M + i] -=  L[j * M + k] * tmp_d; //##TODO: need to be checked
                    }
                }
                
                for (i = k+1; i < b_end; i++){
                     A[i * M + k] = 0.0;
                }
                for (i = k+1; i < b_end; i++){
                     A[k * M + i] = 0.0;
                }
                pivot[k] = 1;
                k = k + 1;
            } else {
                tmp_i = P[k+1];
                P[k+1] = P[r];
                P[r] = tmp_i;
                tmp_d = A[(k+1) * M + (k+1)];
                A[(k+1) * M + (k+1)] = A[r * M + r];
                A[r * M + r] = tmp_d;
                for (i = r + 1; i < b_end; i++){
                    tmp_d = A[i * M + k+1];
                    A[i * M + k+1] = A[i * M + r];
                    A[i * M + r] = tmp_d;
                }
                tmp_d = A[(k+1) * M + k];
                A[(k+1) * M + k] = A[r * M + k];
                A[r * M + k] = tmp_d;
                for (i = k+2; i < r; i++){
                    tmp_d = A[i * M + k+1];
                    A[i * M + k+1] = A[r * M + i];
                    A[r * M + i] = tmp_d;
                }
                if (k > 0){
                    for (i = b_start; i < k; i++){
                        tmp_d = L[(k+1) * M + i];
                        L[(k+1) * M + i] = L[r * M + i];
                        L[r * M + i] = tmp_d;
                    }
                }

                detE = A[k * M + k] * A[(k+1) * M + k+1] - A[(k+1) * M + k] * A[(k+1) * M + k];
                invE_11 = A[(k+1) * M + k+1] / detE;
                invE_22 = A[k * M + k] / detE;
                invE_12 = - A[(k+1) * M + k] / detE;
                invE_21 = invE_12;
                for (i = k + 2; i < b_end; i++){
                    L[i * M + k] = A[i * M + k] * invE_11 + A[i * M + k+1] * invE_21;
                    L[i * M + k+1] = A[i * M + k] * invE_12 + A[i * M + k+1] * invE_22;
                }

                for (j = k+2; j < b_end; j++){
                    for(i = j; i < b_end ; i++){
                        A[i * M + j] -= L[i * M + k] * A[j * M + k] + L[i * M + k+1] * A[j * M + k+1];
                    }
                }
                for (i = k+2; i < b_end; i++){
                    A[i * M + k] = 0.0;
                    A[i * M + k+1] = 0.0;
                }
                for (i = k+2; i < b_end; i++){
                     A[k * M + i] = 0.0;
                     A[(k+1) * M + i] = 0.0;
                 }
                pivot[k] = 2;
                k = k + 2;

                for (i = b_start; i < b_end-1; i++){
                    A[i*M + i+1] = A[(i+1)*M +i];
                }
            }
        }
    }
    if (pivot[b_end-1] == 0)
        if(pivot[b_end-2] != 2)
            pivot[b_end-1] = 1;
}

/* output is L,A,P,pivot, D will be saved in A*/
void BunchKaufman_block(double* A, double* L, int* P, int* pivot, int n, int r){
 
    if(n <= r){
        BunchKaufman_noblock(A, L, P, pivot, n);
        return;
    }
    int num_block = ceil((double)n / (double)r);
    BunchKaufman_subblock(A, L, P, pivot, n, 0, r);
    print_vector(pivot,n);

    // store D in A, may have bugs here
    permute(A, P, r, n, 0, r, n, 1);
    block_solve_column(A, L, A, 0, r, n, pivot);
    
    for(int b_start = r; b_start < n; b_start += r){
        matrix_update(A, A, L, pivot, n, b_start, r);
        BunchKaufman_subblock(A, L, P, pivot, n, b_start, r);
        permute(A, P, b_start + r, n, b_start, b_start + r, n, 1);
        block_solve_column(A, L, A, b_start/r, r, n, pivot);
        permute(L, P, b_start, b_start + r, 0, b_start, n, 0);
    }
}

