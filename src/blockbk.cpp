#include <iostream>
#include <cmath>
#include <cstring>
#include "test_utils.h"
#include "blockbk.h"
#include <immintrin.h>


using namespace std;

// L[i,i]=1
// X * L.T = B
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
void matrix_update_sparse_d_unroll_rename_vec_tail(double* mat_A, double* mat_D, double* mat_L, int* vec_ind, int n, int k, int r){
    int remaining_col_num_j, remaining_row_num, k_n = k*n, r_n = r*n, n_n = n*n, k_r = k-r, j_t, l_id;
    int n_2 = 2*n , n_3 = 3*n, n_4 = 4*n, n_5 = 5*n, n_6 = 6*n, n_7 = 7*n, n_8 = 8*n;
    int k_r_l, t_j_n, k_r_l_n,k_r_l_1, l;
    int s_i_n_k_r_l, t_j_n_k_r_l, k_r_l_n_k_r_l ;
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, dij, dij_1;
    __m256d A_vec, d_vec, d_vec_00, d_vec_04, d_vec_10, d_vec_14,d_vec_20, d_vec_24,d_vec_30, d_vec_34, d_vec_0, d_vec_1 ,d_vec_2 ,d_vec_3;
    __m256d L_s_i_n_k_r_l_vec_0, L_s_i_n_k_r_l_vec_4, D_k_r_l_n_k_r_l_vec_0, D_k_r_l_n_k_r_l_vec_4;
    __m256d L_t_j_n_k_r_l_vec_00, L_t_j_n_k_r_l_vec_04, L_t_j_n_k_r_l_vec_10, L_t_j_n_k_r_l_vec_14, L_t_j_n_k_r_l_vec_20, L_t_j_n_k_r_l_vec_24, L_t_j_n_k_r_l_vec_30, L_t_j_n_k_r_l_vec_34;
    __m256i jump_idx = _mm256_set_epi64x(n_3+3, n_2+2, n+1, 0);
    __m256i jump_idx_n = _mm256_set_epi64x(n_3, n_2, n, 0);
    double d_res[4];

    for(int j = k, j_n = k_n; j < n; j += r, j_n += r_n){
        remaining_col_num_j = min(r + j, n);
        for(int i = j, i_n = j_n; i_n < n_n; i += r, i_n += r_n){
            remaining_row_num = min(r, n-i);
            for(int s = 0, s_i_n = i_n; s < remaining_row_num; s++, s_i_n += n){
                
                for(j_t = j, t_j_n = j_n; j_t + 3 < remaining_col_num_j; t_j_n += n_4, j_t += 4){
                    A_vec = _mm256_loadu_pd((double*)(mat_A + s_i_n + j_t));
                    d_vec_00 = d_vec_04 = d_vec_10 = d_vec_14 = d_vec_20 = d_vec_24 = d_vec_30 = d_vec_34 = _mm256_set1_pd(0);

                    for(l = 0, k_r_l = k_r, k_r_l_n = ( k_r + l) * n; l + 7 < r; l+=8, k_r_l += 8, k_r_l_n += n_8){
                        s_i_n_k_r_l = s_i_n + k_r_l;
                        t_j_n_k_r_l = t_j_n + k_r_l;
                        k_r_l_n_k_r_l = k_r_l_n + k_r_l;

                        L_s_i_n_k_r_l_vec_0 = _mm256_loadu_pd(mat_L+s_i_n_k_r_l);
                        L_s_i_n_k_r_l_vec_4 = _mm256_loadu_pd(mat_L+s_i_n_k_r_l+4);
                        D_k_r_l_n_k_r_l_vec_0 = _mm256_i64gather_pd((double*)( mat_D + k_r_l_n_k_r_l), jump_idx, sizeof(double));
                        D_k_r_l_n_k_r_l_vec_4 = _mm256_i64gather_pd((double*)( mat_D + k_r_l_n_k_r_l + n_4 + 4), jump_idx, sizeof(double));   
                        
                        L_t_j_n_k_r_l_vec_00 = _mm256_loadu_pd((double*)(mat_L+t_j_n_k_r_l));
                        L_t_j_n_k_r_l_vec_04 = _mm256_loadu_pd((double*)(mat_L+t_j_n_k_r_l + 4));
                        L_t_j_n_k_r_l_vec_10 = _mm256_loadu_pd((double*)(mat_L+t_j_n_k_r_l + n));
                        L_t_j_n_k_r_l_vec_14 = _mm256_loadu_pd((double*)(mat_L+t_j_n_k_r_l + n + 4));
                        L_t_j_n_k_r_l_vec_20 = _mm256_loadu_pd((double*)(mat_L+t_j_n_k_r_l + n_2));
                        L_t_j_n_k_r_l_vec_24 = _mm256_loadu_pd((double*)(mat_L+t_j_n_k_r_l + n_2 + 4));
                        L_t_j_n_k_r_l_vec_30 = _mm256_loadu_pd((double*)(mat_L+t_j_n_k_r_l + n_3));
                        L_t_j_n_k_r_l_vec_34 = _mm256_loadu_pd((double*)(mat_L+t_j_n_k_r_l + n_3 + 4));

                        L_s_i_n_k_r_l_vec_0 = _mm256_mul_pd(L_s_i_n_k_r_l_vec_0, D_k_r_l_n_k_r_l_vec_0);
                        L_s_i_n_k_r_l_vec_4 = _mm256_mul_pd(L_s_i_n_k_r_l_vec_4, D_k_r_l_n_k_r_l_vec_4);


                        d_vec_00 = _mm256_fmadd_pd(L_s_i_n_k_r_l_vec_0, L_t_j_n_k_r_l_vec_00, d_vec_00);
                        d_vec_04 = _mm256_fmadd_pd(L_s_i_n_k_r_l_vec_4, L_t_j_n_k_r_l_vec_04, d_vec_04);
                        d_vec_10 = _mm256_fmadd_pd(L_s_i_n_k_r_l_vec_0, L_t_j_n_k_r_l_vec_10, d_vec_10);
                        d_vec_14 = _mm256_fmadd_pd(L_s_i_n_k_r_l_vec_4, L_t_j_n_k_r_l_vec_14, d_vec_14);
                        d_vec_20 = _mm256_fmadd_pd(L_s_i_n_k_r_l_vec_0, L_t_j_n_k_r_l_vec_20, d_vec_20);
                        d_vec_24 = _mm256_fmadd_pd(L_s_i_n_k_r_l_vec_4, L_t_j_n_k_r_l_vec_24, d_vec_24);
                        d_vec_30 = _mm256_fmadd_pd(L_s_i_n_k_r_l_vec_0, L_t_j_n_k_r_l_vec_30, d_vec_30);
                        d_vec_34 = _mm256_fmadd_pd(L_s_i_n_k_r_l_vec_4, L_t_j_n_k_r_l_vec_34, d_vec_34);
                        
                    }
                    d_vec_0 = _mm256_add_pd(d_vec_00, d_vec_04);
                    d_vec_1 = _mm256_add_pd(d_vec_10, d_vec_14);
                    d_vec_2 = _mm256_add_pd(d_vec_20, d_vec_24);
                    d_vec_3 = _mm256_add_pd(d_vec_30, d_vec_34); 

                    

                    for( ; l + 3 < r; l+=4, k_r_l +=4, k_r_l_n += n_4){
                        s_i_n_k_r_l = s_i_n + k_r_l;
                        t_j_n_k_r_l = t_j_n + k_r_l;
                        k_r_l_n_k_r_l = k_r_l_n + k_r_l;

                        L_s_i_n_k_r_l_vec_0 = _mm256_loadu_pd(mat_L+s_i_n_k_r_l);
                        D_k_r_l_n_k_r_l_vec_0 = _mm256_i64gather_pd((double*)( mat_D + k_r_l_n_k_r_l), jump_idx, sizeof(double));
                        
                        L_t_j_n_k_r_l_vec_00 = _mm256_loadu_pd((double*)(mat_L+t_j_n_k_r_l));
                        L_t_j_n_k_r_l_vec_10 = _mm256_loadu_pd((double*)(mat_L+t_j_n_k_r_l + n));
                        L_t_j_n_k_r_l_vec_20 = _mm256_loadu_pd((double*)(mat_L+t_j_n_k_r_l + n_2));
                        L_t_j_n_k_r_l_vec_30 = _mm256_loadu_pd((double*)(mat_L+t_j_n_k_r_l + n_3));
                        
                        L_s_i_n_k_r_l_vec_0 = _mm256_mul_pd(L_s_i_n_k_r_l_vec_0, D_k_r_l_n_k_r_l_vec_0);

                        d_vec_0 = _mm256_fmadd_pd(L_s_i_n_k_r_l_vec_0, L_t_j_n_k_r_l_vec_00, d_vec_0);
                        d_vec_1 = _mm256_fmadd_pd(L_s_i_n_k_r_l_vec_0, L_t_j_n_k_r_l_vec_10, d_vec_1);
                        d_vec_2 = _mm256_fmadd_pd(L_s_i_n_k_r_l_vec_0, L_t_j_n_k_r_l_vec_20, d_vec_2);
                        d_vec_3 = _mm256_fmadd_pd(L_s_i_n_k_r_l_vec_0, L_t_j_n_k_r_l_vec_30, d_vec_3);
                        
                    }

                    d_vec_0 = _mm256_permute4x64_pd(_mm256_hadd_pd(d_vec_0, d_vec_2), 0b11011000);
                    d_vec_2 = _mm256_permute4x64_pd(_mm256_hadd_pd(d_vec_1, d_vec_3), 0b11011000);
                    d_vec = _mm256_hadd_pd(d_vec_0, d_vec_2);

                    double temp;
                    __m256d temp_vec;
                    for( ; l < r; l++, k_r_l ++, k_r_l_n += n){
                        L_s_i_n_k_r_l_vec_0 =  _mm256_i64gather_pd((double*)(mat_L + t_j_n + k_r_l), jump_idx_n, sizeof(double));
                        temp = mat_L[s_i_n + k_r_l] * mat_D[k_r_l_n + k_r_l];
                        temp_vec =  _mm256_set1_pd(temp);
                        d_vec = _mm256_fmadd_pd(temp_vec, L_s_i_n_k_r_l_vec_0, d_vec);
                    }
                    
                    A_vec = _mm256_sub_pd(A_vec, d_vec);
                    _mm256_storeu_pd((double*)(mat_A + s_i_n + j_t), A_vec);
                }

                for(; j_t < remaining_col_num_j; t_j_n += n, j_t ++){
                    d_vec_00 = d_vec_04 = _mm256_set1_pd(0);
                    for(l = 0, k_r_l = k_r, k_r_l_n = ( k_r + l) * n; l + 7 < r; l+=8, k_r_l += 8, k_r_l_n += n_8){
                        s_i_n_k_r_l = s_i_n + k_r_l;
                        t_j_n_k_r_l = t_j_n + k_r_l;
                        k_r_l_n_k_r_l = k_r_l_n + k_r_l;

                        L_s_i_n_k_r_l_vec_0 = _mm256_loadu_pd(mat_L+s_i_n_k_r_l);
                        L_s_i_n_k_r_l_vec_4 = _mm256_loadu_pd(mat_L+s_i_n_k_r_l+4);
                        D_k_r_l_n_k_r_l_vec_0 = _mm256_i64gather_pd((double*)( mat_D + k_r_l_n_k_r_l), jump_idx, sizeof(double));
                        D_k_r_l_n_k_r_l_vec_4 = _mm256_i64gather_pd((double*)( mat_D + k_r_l_n_k_r_l + n_4 + 4), jump_idx, sizeof(double));   
                        
                        L_t_j_n_k_r_l_vec_00 = _mm256_loadu_pd((double*)(mat_L+t_j_n_k_r_l));
                        L_t_j_n_k_r_l_vec_04 = _mm256_loadu_pd((double*)(mat_L+t_j_n_k_r_l + 4));

                        L_s_i_n_k_r_l_vec_0 = _mm256_mul_pd(L_s_i_n_k_r_l_vec_0, D_k_r_l_n_k_r_l_vec_0);
                        L_s_i_n_k_r_l_vec_4 = _mm256_mul_pd(L_s_i_n_k_r_l_vec_4, D_k_r_l_n_k_r_l_vec_4);
                        
                        d_vec_00 = _mm256_fmadd_pd(L_s_i_n_k_r_l_vec_0, L_t_j_n_k_r_l_vec_00, d_vec_00);
                        d_vec_04 = _mm256_fmadd_pd(L_s_i_n_k_r_l_vec_4, L_t_j_n_k_r_l_vec_04, d_vec_04);

                    }

                    for( ; l + 3 < r; l+=4, k_r_l +=4, k_r_l_n += n_4){
                        s_i_n_k_r_l = s_i_n + k_r_l;
                        t_j_n_k_r_l = t_j_n + k_r_l;
                        k_r_l_n_k_r_l = k_r_l_n + k_r_l;

                        L_s_i_n_k_r_l_vec_0 = _mm256_loadu_pd(mat_L+s_i_n_k_r_l);
                        D_k_r_l_n_k_r_l_vec_0 = _mm256_i64gather_pd((double*)( mat_D + k_r_l_n_k_r_l), jump_idx, sizeof(double));
                        
                        L_t_j_n_k_r_l_vec_00 = _mm256_loadu_pd((double*)(mat_L+t_j_n_k_r_l));
                        
                        L_s_i_n_k_r_l_vec_0 = _mm256_mul_pd(L_s_i_n_k_r_l_vec_0, D_k_r_l_n_k_r_l_vec_0);

                        d_vec_00 = _mm256_fmadd_pd(L_s_i_n_k_r_l_vec_0, L_t_j_n_k_r_l_vec_00, d_vec_00);
                    }
                    d_vec_0 = _mm256_add_pd(d_vec_00, d_vec_04);
                    d_vec_0 = _mm256_hadd_pd(d_vec_0, d_vec_0);
                    _mm256_storeu_pd(d_res, d_vec_0);
                    d_0 = d_res[0] + d_res[3];

                    for( ; l < r; l++, k_r_l ++, k_r_l_n += n){
                        d_0 +=  mat_L[s_i_n + k_r_l] * mat_D[k_r_l_n + k_r_l] * mat_L[t_j_n + k_r_l];
                    }
                    
                    mat_A[s_i_n + j_t] = mat_A[s_i_n + j_t] - d_0;
                }

                for(j_t = j, t_j_n = j_n; j_t < remaining_col_num_j; t_j_n += n, j_t += 1){
                    d_1  = 0;

                    for(l_id = 1; l_id <= vec_ind[0]; l_id++){
                        k_r_l = k_r + vec_ind[l_id];
                        dij = mat_D[k_r_l * n + n + k_r_l];
                        d_1 += dij * (mat_L[s_i_n+ k_r_l+1] * mat_L[t_j_n + k_r_l] + mat_L[s_i_n + k_r_l] *  mat_L[t_j_n + k_r_l+1]);
                    }
                    
                    mat_A[s_i_n + j_t] = mat_A[s_i_n + j_t] - d_1;
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
void BunchKaufman_subblock(double* A, double* L, int* P, int* pivot, int* pivot_idx, int M, int b_start, int b_size){
    // [b_start, b_end) interval
    int b_end = min(b_start + b_size, M);
    b_size = b_end - b_start;

    const double alpha = (1+sqrt(17))/8;
    int r, i, j, tmp_i;
    int k = b_start;
    double w1, wr;
    double tmp_d, A_kk;
    double detE, invE_11, invE_22, invE_12, invE_21;
    int pivot_cnt = 0;
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
                pivot_idx[++pivot_cnt] = k;
                
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
    pivot_idx[0] = pivot_cnt;
}

/* output is L,A,P,pivot, D will be saved in A*/
void BunchKaufman_block(double* A, double* L, int* P, int* pivot, int n, int r){
    int* pivot_idx = (int*)malloc((r + 1) * sizeof(int));
    if(n <= r){
        BunchKaufman_noblock(A, L, P, pivot, n);
        return;
    }
    int num_block = ceil((double)n / (double)r);
    BunchKaufman_subblock(A, L, P, pivot, pivot_idx, n, 0, r);
    // store D in A, may have bugs here
    permute(A, P, r, n, 0, r, n, 1);
    block_solve_column(A, L, A, 0, r, n, pivot);
    
    for(int b_start = r; b_start < n; b_start += r){
        // matrix_update(A, A, L, pivot, n, b_start, r);
        matrix_update_sparse_d_unroll_rename_vec_tail(A, A, L, pivot_idx, n, b_start, r);

        BunchKaufman_subblock(A, L, P, pivot, pivot_idx, n, b_start, r);
        permute(A, P, b_start + r, n, b_start, b_start + r, n, 1);
        block_solve_column(A, L, A, b_start/r, r, n, pivot);
        permute(L, P, b_start, b_start + r, 0, b_start, n, 0);
    }
    free(pivot_idx);
}




