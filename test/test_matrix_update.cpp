#include <iostream>
#include "test_utils.h"
#include "tsc_x86.h"
#include <time.h>
#include <immintrin.h>
#include <vector>
#include <string>

// #define FLOP_COUNTER

using namespace std;

void matrix_update_gt(double* mat_A, double* mat_D, double* mat_L, int* vec_ind, int n, int k, int r){
    for(int j = k; j < n; j += r){
        for(int i = j; i < n; i += r){                                                                                                                                                                           
            int remaining_row_num = min(r, n-i);
            int remaining_col_num = min(r, n-j);
            for(int s = 0; s < remaining_row_num; s++){
                for(int t = 0; t < remaining_col_num; t++){
                    double d_0 = 0 , d_n1 = 0 , d_1 = 0;
                    for(int l = 0; l < r; l++){
                        //sum_{l}(L[t,l] * d[ll] * L[t,l])
                        d_0 += mat_L[(i+s)*n + k-r+l] * mat_D[(l+k-r)*n + l+k-r] * mat_L[(j+t)*n + k-r+l];

                        #ifdef FLOP_COUNTER
                            flops()+=3;
                        #endif  
                    }
                    for(int u = 0; u < r - 1; u++){
                        //sum_{u}(L[s,u+1] * d[u+1,u] * L[t,u])
                        d_n1 += mat_L[(i+s)*n + k-r+u+1] * mat_D[(u+1+k-r)*n + u+k-r] * mat_L[(j+t)*n + k-r+u];

                        #ifdef FLOP_COUNTER
                            flops()+=3;
                        #endif
                    }
                    for(int p = 1; p < r; p++){
                        //sum_{p}(L[s,p-1] * d[p-1,p] * L[t,p])
                        d_1 += mat_L[(i+s)*n + k-r+p-1] * mat_D[(p-1+k-r)*n + p+k-r] * mat_L[(j+t)*n + k-r+p];

                        #ifdef FLOP_COUNTER
                            flops()+=3;
                        #endif
                    }
                    
                    mat_A[(i+s)*n + (j+t)] = mat_A[(i+s)*n + (j+t)] - d_0 - d_n1 - d_1;
                    #ifdef FLOP_COUNTER
                        flops()+=3;
                    #endif
                }
            }
        }
    }
}

void matrix_update_ijts(double* mat_A, double* mat_D, double* mat_L, int* vec_ind, int n, int k, int r){
    
    for(int i = k; i < n; i += r){
        for(int j = k; j <= i; j += r){
            int remaining_row_num = min(r, n-i);
            int remaining_col_num = min(r, n-j);
            for(int t = 0; t < remaining_col_num; t++){
                for(int s = 0; s < remaining_row_num; s++){
                    double d_0 = 0 , d_n1 = 0 , d_1 = 0;
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
                    #ifdef FLOP_COUNTER
                        flops()+=r*3+(r-1)*3+(r-1)*3+3;
                    #endif
                    
                }
            }
        }
    }
}

void matrix_update_sparse_d(double* mat_A, double* mat_D, double* mat_L, int* vec_ind, int n, int k, int r){
    int remaining_col_num_j, remaining_row_num, k_n = k*n, r_n = r*n, n_n = n*n, k_r = k-r, j_t;
    int k_r_l, t_j_n;
    double d_0, d_1, dij;
    for(int j = k, j_n = k_n; j < n; j += r, j_n += r_n){
        remaining_col_num_j = min(r + j, n);
        for(int i = j, i_n = j_n; i_n < n_n; i += r, i_n += r_n){
            remaining_row_num = min(r, n-i);
            for(int s = 0, s_i_n = i_n; s < remaining_row_num; s++, s_i_n += n){

                for(j_t = j, t_j_n = j_n; j_t < remaining_col_num_j; t_j_n += n, j_t ++){
                    d_0 = 0 , d_1  = 0;
                    for(int l = 0, k_r_l_n = ( k_r + l) * n; l < r; l++, k_r_l_n += n){
                        k_r_l = k_r + l;
                        d_0 +=  mat_L[s_i_n + k_r_l] * mat_D[k_r_l_n + k_r_l] * mat_L[t_j_n + k_r_l];
                        #ifdef FLOP_COUNTER
                            flops()+=3;
                        #endif
                    }
                    for(int l_id = 1; l_id <= vec_ind[0]; l_id++){
                        k_r_l = vec_ind[l_id];
                        dij = mat_D[k_r_l * n + n + k_r_l];
                        d_1 += dij * (mat_L[s_i_n+ k_r_l+1] * mat_L[t_j_n + k_r_l] + mat_L[s_i_n + k_r_l] *  mat_L[t_j_n + k_r_l+1]);
                    }
                    mat_A[s_i_n + j_t] = mat_A[s_i_n + j_t] - d_0 - d_1;
                    #ifdef FLOP_COUNTER
                            flops()+=vec_ind[0]*5 + 2;
                    #endif
                }
            }
        }
    }
}

void matrix_update_sparse_d_unroll(double* mat_A, double* mat_D, double* mat_L, int* vec_ind, int n, int k, int r){
    int remaining_col_num_j, remaining_row_num, k_n = k*n, r_n = r*n, n_n = n*n, k_r = k-r, j_t, l_id;
    int n_2 = 2*n, n_3 = 3*n, n_4 = 4*n, n_5 = 5*n, n_6 = 6*n, n_7 = 7*n, n_8 = 8*n;
    int k_r_l, t_j_n, k_r_l_n,k_r_l_1, l;
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, dij, dij_1;
    for(int j = k, j_n = k_n; j < n; j += r, j_n += r_n){
        remaining_col_num_j = min(r + j, n);
        for(int i = j, i_n = j_n; i_n < n_n; i += r, i_n += r_n){
            remaining_row_num = min(r, n-i);
            for(int s = 0, s_i_n = i_n; s < remaining_row_num; s++, s_i_n += n){

                for(j_t = j, t_j_n = j_n; j_t < remaining_col_num_j; t_j_n += n, j_t ++){
                    d_0 =  d_1 = d_2 = d_3 = d_4 = d_5 = d_6 = d_7 = 0;

                    for(l = 0, k_r_l = k_r, k_r_l_n = ( k_r + l) * n; l + 7 < r; l+=8, k_r_l += 8, k_r_l_n += n_8){
                        d_0 +=  mat_L[s_i_n + k_r_l    ] * mat_D[k_r_l_n + k_r_l] * mat_L[t_j_n + k_r_l];
                        d_1 +=  mat_L[s_i_n + k_r_l + 1] * mat_D[k_r_l_n + n   + k_r_l + 1] * mat_L[t_j_n + k_r_l + 1];
                        d_2 +=  mat_L[s_i_n + k_r_l + 2] * mat_D[k_r_l_n + n_2 + k_r_l + 2] * mat_L[t_j_n + k_r_l + 2];
                        d_3 +=  mat_L[s_i_n + k_r_l + 3] * mat_D[k_r_l_n + n_3 + k_r_l + 3] * mat_L[t_j_n + k_r_l + 3];
                        d_4 +=  mat_L[s_i_n + k_r_l + 4] * mat_D[k_r_l_n + n_4 + k_r_l + 4] * mat_L[t_j_n + k_r_l + 4];
                        d_5 +=  mat_L[s_i_n + k_r_l + 5] * mat_D[k_r_l_n + n_5 + k_r_l + 5] * mat_L[t_j_n + k_r_l + 5];
                        d_6 +=  mat_L[s_i_n + k_r_l + 6] * mat_D[k_r_l_n + n_6 + k_r_l + 6] * mat_L[t_j_n + k_r_l + 6];
                        d_7 +=  mat_L[s_i_n + k_r_l + 7] * mat_D[k_r_l_n + n_7 + k_r_l + 7] * mat_L[t_j_n + k_r_l + 7];
                        #ifdef FLOP_COUNTER
                            flops()+=24;
                        #endif
                    }
                    
                    for( ; l + 3 < r; l+=4, k_r_l +=4, k_r_l_n += n_4){
                        d_0 +=  mat_L[s_i_n + k_r_l] * mat_D[k_r_l_n + k_r_l] * mat_L[t_j_n + k_r_l];
                        d_1 +=  mat_L[s_i_n + k_r_l + 1] * mat_D[k_r_l_n + n   + k_r_l + 1] * mat_L[t_j_n + k_r_l + 1];
                        d_2 +=  mat_L[s_i_n + k_r_l + 2] * mat_D[k_r_l_n + n_2 + k_r_l + 2] * mat_L[t_j_n + k_r_l + 2];
                        d_3 +=  mat_L[s_i_n + k_r_l + 3] * mat_D[k_r_l_n + n_3 + k_r_l + 3] * mat_L[t_j_n + k_r_l + 3];

                        #ifdef FLOP_COUNTER
                            flops()+=12;
                        #endif
                    }
                    d_0 += (d_1 + d_2 + d_3 + d_4 + d_5 + d_6 + d_7);
                    #ifdef FLOP_COUNTER
                        flops()+=7;
                    #endif

                    for( ; l < r; l++, k_r_l ++, k_r_l_n += n){
                        d_0 +=  mat_L[s_i_n + k_r_l] * mat_D[k_r_l_n + k_r_l] * mat_L[t_j_n + k_r_l];

                        #ifdef FLOP_COUNTER
                            flops()+=3;
                        #endif
                    }
                    
                    d_1  = 0;
                    for(l_id = 1; l_id <= vec_ind[0]; l_id++){
                        k_r_l = vec_ind[l_id];
                        dij = mat_D[k_r_l * n + n + k_r_l];
                        d_1 += dij * (mat_L[s_i_n+ k_r_l+1] * mat_L[t_j_n + k_r_l] + mat_L[s_i_n + k_r_l] *  mat_L[t_j_n + k_r_l+1]);

                        #ifdef FLOP_COUNTER
                            flops()+=5;
                        #endif
                    }
                    
                    mat_A[s_i_n + j_t] = mat_A[s_i_n + j_t] - d_0 - d_1;
                    #ifdef FLOP_COUNTER
                        flops()+=2;
                    #endif
                }
            }
        }
    }
}

void matrix_update_sparse_d_unroll_rename(double* mat_A, double* mat_D, double* mat_L, int* vec_ind, int n, int k, int r){
    int remaining_col_num_j, remaining_row_num, k_n = k*n, r_n = r*n, n_n = n*n, k_r = k-r, j_t, l_id;
    int n_2 = 2*n , n_3 = 3*n, n_4 = 4*n, n_5 = 5*n, n_6 = 6*n, n_7 = 7*n, n_8 = 8*n;
    int k_r_l, t_j_n, k_r_l_n,k_r_l_1, l;
    int s_i_n_k_r_l, t_j_n_k_r_l, k_r_l_n_k_r_l ;
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, dij, dij_1;
    for(int j = k, j_n = k_n; j < n; j += r, j_n += r_n){
        remaining_col_num_j = min(r + j, n);
        for(int i = j, i_n = j_n; i_n < n_n; i += r, i_n += r_n){
            remaining_row_num = min(r, n-i);
            for(int s = 0, s_i_n = i_n; s < remaining_row_num; s++, s_i_n += n){

                for(j_t = j, t_j_n = j_n; j_t < remaining_col_num_j; t_j_n += n, j_t ++){
                    d_0 =  d_1 = d_2 = d_3 = d_4 = d_5 = d_6 = d_7 = 0;

                    for(l = 0, k_r_l = k_r, k_r_l_n = ( k_r + l) * n; l + 7 < r; l+=8, k_r_l += 8, k_r_l_n += n_8){
                        s_i_n_k_r_l = s_i_n + k_r_l;
                        t_j_n_k_r_l = t_j_n + k_r_l;
                        k_r_l_n_k_r_l = k_r_l_n + k_r_l;
                        d_0 +=  mat_L[s_i_n_k_r_l    ] * mat_D[k_r_l_n_k_r_l] * mat_L[t_j_n_k_r_l];
                        d_1 +=  mat_L[s_i_n_k_r_l + 1] * mat_D[k_r_l_n_k_r_l + n + 1] * mat_L[t_j_n_k_r_l + 1];
                        d_2 +=  mat_L[s_i_n_k_r_l + 2] * mat_D[k_r_l_n_k_r_l + n_2 + 2] * mat_L[t_j_n_k_r_l + 2];
                        d_3 +=  mat_L[s_i_n_k_r_l + 3] * mat_D[k_r_l_n_k_r_l + n_3 + 3] * mat_L[t_j_n_k_r_l + 3];
                        d_4 +=  mat_L[s_i_n_k_r_l + 4] * mat_D[k_r_l_n_k_r_l + n_4 + 4] * mat_L[t_j_n_k_r_l + 4];
                        d_5 +=  mat_L[s_i_n_k_r_l + 5] * mat_D[k_r_l_n_k_r_l + n_5 + 5] * mat_L[t_j_n_k_r_l + 5];
                        d_6 +=  mat_L[s_i_n_k_r_l + 6] * mat_D[k_r_l_n_k_r_l + n_6 + 6] * mat_L[t_j_n_k_r_l + 6];
                        d_7 +=  mat_L[s_i_n_k_r_l + 7] * mat_D[k_r_l_n_k_r_l + n_7 + 7] * mat_L[t_j_n_k_r_l + 7];
                        #ifdef FLOP_COUNTER
                            flops()+=24;
                        #endif
                    }
                    
                    for( ; l + 3 < r; l+=4, k_r_l +=4, k_r_l_n += n_4){
                        s_i_n_k_r_l = s_i_n + k_r_l;
                        t_j_n_k_r_l = t_j_n + k_r_l;
                        k_r_l_n_k_r_l = k_r_l_n + k_r_l;
                        d_0 +=  mat_L[s_i_n_k_r_l] * mat_D[k_r_l_n_k_r_l] * mat_L[t_j_n + k_r_l];
                        d_1 +=  mat_L[s_i_n_k_r_l + 1] * mat_D[k_r_l_n_k_r_l + n   + 1] * mat_L[t_j_n_k_r_l + 1];
                        d_2 +=  mat_L[s_i_n_k_r_l + 2] * mat_D[k_r_l_n_k_r_l + n_2 + 2] * mat_L[t_j_n_k_r_l + 2];
                        d_3 +=  mat_L[s_i_n_k_r_l + 3] * mat_D[k_r_l_n_k_r_l + n_3 + 3] * mat_L[t_j_n_k_r_l + 3];
                        #ifdef FLOP_COUNTER
                            flops()+=12;
                        #endif
                    }
                    d_0 += (d_1 + d_2 + d_3 + d_4 + d_5 + d_6 + d_7);
                    #ifdef FLOP_COUNTER
                        flops()+=7;
                    #endif

                    for( ; l < r; l++, k_r_l ++, k_r_l_n += n){
                        d_0 +=  mat_L[s_i_n + k_r_l] * mat_D[k_r_l_n + k_r_l] * mat_L[t_j_n + k_r_l];
                        #ifdef FLOP_COUNTER
                            flops()+=3;
                        #endif
                    }
                    
                    d_1  = 0;
                    for(l_id = 1; l_id <= vec_ind[0]; l_id++){
                        k_r_l = vec_ind[l_id];
                        dij = mat_D[k_r_l * n + n + k_r_l];
                        d_1 += dij * (mat_L[s_i_n+ k_r_l+1] * mat_L[t_j_n + k_r_l] + mat_L[s_i_n + k_r_l] *  mat_L[t_j_n + k_r_l+1]);
                    }
                    
                    mat_A[s_i_n + j_t] = mat_A[s_i_n + j_t] - d_0 - d_1;
                    #ifdef FLOP_COUNTER
                            flops()+=vec_ind[0]*5 + 2;
                    #endif
                }
            }
        }
    }
}

void matrix_update_sparse_d_unroll_rename_vec(double* mat_A, double* mat_D, double* mat_L, int* vec_ind, int n, int k, int r){
    int remaining_col_num_j, remaining_row_num, k_n = k*n, r_n = r*n, n_n = n*n, k_r = k-r, j_t, l_id;
    int n_2 = 2*n , n_3 = 3*n, n_4 = 4*n, n_5 = 5*n, n_6 = 6*n, n_7 = 7*n, n_8 = 8*n;
    int k_r_l, t_j_n, k_r_l_n,k_r_l_1, l;
    int s_i_n_k_r_l, t_j_n_k_r_l, k_r_l_n_k_r_l ;
    double d_0, d_1, d_2, d_3, d_4, d_5, d_6, d_7, dij, dij_1;
    __m256d d_vec, d_vec_00, d_vec_04, d_vec_10, d_vec_14,d_vec_20, d_vec_24,d_vec_30, d_vec_34, d_vec_0, d_vec_1 ,d_vec_2 ,d_vec_3;
    __m256d L_s_i_n_k_r_l_vec_0, L_s_i_n_k_r_l_vec_4, D_k_r_l_n_k_r_l_vec_0, D_k_r_l_n_k_r_l_vec_4;
    __m256d L_t_j_n_k_r_l_vec_00, L_t_j_n_k_r_l_vec_04, L_t_j_n_k_r_l_vec_10, L_t_j_n_k_r_l_vec_14, L_t_j_n_k_r_l_vec_20, L_t_j_n_k_r_l_vec_24, L_t_j_n_k_r_l_vec_30, L_t_j_n_k_r_l_vec_34;
    __m256i jump_idx = _mm256_set_epi64x(n_3+3, n_2+2, n+1, 0);
    double d_res[4];

    for(int j = k, j_n = k_n; j < n; j += r, j_n += r_n){
        remaining_col_num_j = min(r + j, n);
        for(int i = j, i_n = j_n; i_n < n_n; i += r, i_n += r_n){
            remaining_row_num = min(r, n-i);
            for(int s = 0, s_i_n = i_n; s < remaining_row_num; s++, s_i_n += n){

                for(j_t = j, t_j_n = j_n; j_t + 3 < remaining_col_num_j; t_j_n += n_4, j_t += 4){
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
                        
                        // d_0 +=  mat_L[s_i_n_k_r_l    ] * mat_D[k_r_l_n_k_r_l] * mat_L[t_j_n_k_r_l];
                        // d_1 +=  mat_L[s_i_n_k_r_l + 1] * mat_D[k_r_l_n_k_r_l + n + 1] * mat_L[t_j_n_k_r_l + 1];
                        // d_2 +=  mat_L[s_i_n_k_r_l + 2] * mat_D[k_r_l_n_k_r_l + n_2 + 2] * mat_L[t_j_n_k_r_l + 2];
                        // d_3 +=  mat_L[s_i_n_k_r_l + 3] * mat_D[k_r_l_n_k_r_l + n_3 + 3] * mat_L[t_j_n_k_r_l + 3];
                        // d_4 +=  mat_L[s_i_n_k_r_l + 4] * mat_D[k_r_l_n_k_r_l + n_4 + 4] * mat_L[t_j_n_k_r_l + 4];
                        // d_5 +=  mat_L[s_i_n_k_r_l + 5] * mat_D[k_r_l_n_k_r_l + n_5 + 5] * mat_L[t_j_n_k_r_l + 5];
                        // d_6 +=  mat_L[s_i_n_k_r_l + 6] * mat_D[k_r_l_n_k_r_l + n_6 + 6] * mat_L[t_j_n_k_r_l + 6];
                        // d_7 +=  mat_L[s_i_n_k_r_l + 7] * mat_D[k_r_l_n_k_r_l + n_7 + 7] * mat_L[t_j_n_k_r_l + 7];

                        #ifdef FLOP_COUNTER
                            flops()+=18 * 4;
                        #endif
                    }
                    d_vec_0 = _mm256_add_pd(d_vec_00, d_vec_04);
                    d_vec_1 = _mm256_add_pd(d_vec_10, d_vec_14);
                    d_vec_2 = _mm256_add_pd(d_vec_20, d_vec_24);
                    d_vec_3 = _mm256_add_pd(d_vec_30, d_vec_34); 

                    d_vec_0 = _mm256_permute4x64_pd(_mm256_hadd_pd(d_vec_0, d_vec_2), 0b11011000);
                    d_vec_2 = _mm256_permute4x64_pd(_mm256_hadd_pd(d_vec_1, d_vec_3), 0b11011000);
                    d_vec = _mm256_hadd_pd(d_vec_0, d_vec_2);
                    _mm256_storeu_pd(d_res, d_vec);

                    #ifdef FLOP_COUNTER
                        flops()+= 7 * 4;
                    #endif  

                    // for( ; l + 3 < r; l+=4, k_r_l +=4, k_r_l_n += n_4){
                    //     s_i_n_k_r_l = s_i_n + k_r_l;
                    //     t_j_n_k_r_l = t_j_n + k_r_l;
                    //     k_r_l_n_k_r_l = k_r_l_n + k_r_l;
                    //     d_0 +=  mat_L[s_i_n_k_r_l] * mat_D[k_r_l_n_k_r_l] * mat_L[t_j_n + k_r_l];
                    //     d_1 +=  mat_L[s_i_n_k_r_l + 1] * mat_D[k_r_l_n_k_r_l + n   + 1] * mat_L[t_j_n_k_r_l + 1];
                    //     d_2 +=  mat_L[s_i_n_k_r_l + 2] * mat_D[k_r_l_n_k_r_l + n_2 + 2] * mat_L[t_j_n_k_r_l + 2];
                    //     d_3 +=  mat_L[s_i_n_k_r_l + 3] * mat_D[k_r_l_n_k_r_l + n_3 + 3] * mat_L[t_j_n_k_r_l + 3];
                    // }

                    // d_0 += (d_1 + d_2 + d_3 + d_4 + d_5 + d_6 + d_7);
                    double temp;
                    for( ; l < r; l++, k_r_l ++, k_r_l_n += n){
                        temp = mat_L[s_i_n + k_r_l] * mat_D[k_r_l_n + k_r_l];
                        d_res[0] +=  temp * mat_L[t_j_n + k_r_l];
                        d_res[1] +=  temp * mat_L[t_j_n + n + k_r_l];
                        d_res[2] +=  temp * mat_L[t_j_n + n_2 + k_r_l];
                        d_res[3] +=  temp * mat_L[t_j_n + n_3 + k_r_l];

                        #ifdef FLOP_COUNTER
                            flops()+=9;
                        #endif
                    }
                    
                    
                    mat_A[s_i_n + j_t] = mat_A[s_i_n + j_t] - d_res[0];
                    mat_A[s_i_n + j_t + 1] = mat_A[s_i_n + j_t + 1] - d_res[1];
                    mat_A[s_i_n + j_t + 2] = mat_A[s_i_n + j_t + 2] - d_res[2];
                    mat_A[s_i_n + j_t + 3] = mat_A[s_i_n + j_t + 3] - d_res[3];
                    #ifdef FLOP_COUNTER
                        flops()+=4;
                    #endif
                }

                for(; j_t < remaining_col_num_j; t_j_n += n, j_t ++){
                    d_0 =  d_1 = d_2 = d_3 = d_4 = d_5 = d_6 = d_7 = 0;

                    for(l = 0, k_r_l = k_r, k_r_l_n = ( k_r + l) * n; l + 7 < r; l+=8, k_r_l += 8, k_r_l_n += n_8){
                        s_i_n_k_r_l = s_i_n + k_r_l;
                        t_j_n_k_r_l = t_j_n + k_r_l;
                        k_r_l_n_k_r_l = k_r_l_n + k_r_l;
                        d_0 +=  mat_L[s_i_n_k_r_l    ] * mat_D[k_r_l_n_k_r_l] * mat_L[t_j_n_k_r_l];
                        d_1 +=  mat_L[s_i_n_k_r_l + 1] * mat_D[k_r_l_n_k_r_l + n + 1] * mat_L[t_j_n_k_r_l + 1];
                        d_2 +=  mat_L[s_i_n_k_r_l + 2] * mat_D[k_r_l_n_k_r_l + n_2 + 2] * mat_L[t_j_n_k_r_l + 2];
                        d_3 +=  mat_L[s_i_n_k_r_l + 3] * mat_D[k_r_l_n_k_r_l + n_3 + 3] * mat_L[t_j_n_k_r_l + 3];
                        d_4 +=  mat_L[s_i_n_k_r_l + 4] * mat_D[k_r_l_n_k_r_l + n_4 + 4] * mat_L[t_j_n_k_r_l + 4];
                        d_5 +=  mat_L[s_i_n_k_r_l + 5] * mat_D[k_r_l_n_k_r_l + n_5 + 5] * mat_L[t_j_n_k_r_l + 5];
                        d_6 +=  mat_L[s_i_n_k_r_l + 6] * mat_D[k_r_l_n_k_r_l + n_6 + 6] * mat_L[t_j_n_k_r_l + 6];
                        d_7 +=  mat_L[s_i_n_k_r_l + 7] * mat_D[k_r_l_n_k_r_l + n_7 + 7] * mat_L[t_j_n_k_r_l + 7];

                        #ifdef FLOP_COUNTER
                            flops()+=24;
                        #endif
                    }
                    
                    for( ; l + 3 < r; l+=4, k_r_l +=4, k_r_l_n += n_4){
                        s_i_n_k_r_l = s_i_n + k_r_l;
                        t_j_n_k_r_l = t_j_n + k_r_l;
                        k_r_l_n_k_r_l = k_r_l_n + k_r_l;
                        d_0 +=  mat_L[s_i_n_k_r_l] * mat_D[k_r_l_n_k_r_l] * mat_L[t_j_n + k_r_l];
                        d_1 +=  mat_L[s_i_n_k_r_l + 1] * mat_D[k_r_l_n_k_r_l + n   + 1] * mat_L[t_j_n_k_r_l + 1];
                        d_2 +=  mat_L[s_i_n_k_r_l + 2] * mat_D[k_r_l_n_k_r_l + n_2 + 2] * mat_L[t_j_n_k_r_l + 2];
                        d_3 +=  mat_L[s_i_n_k_r_l + 3] * mat_D[k_r_l_n_k_r_l + n_3 + 3] * mat_L[t_j_n_k_r_l + 3];
                        #ifdef FLOP_COUNTER
                            flops()+=12;
                        #endif
                    }
                    d_0 += (d_1 + d_2 + d_3 + d_4 + d_5 + d_6 + d_7);

                    #ifdef FLOP_COUNTER
                        flops()+=7;
                    #endif

                    for( ; l < r; l++, k_r_l ++, k_r_l_n += n){
                        d_0 +=  mat_L[s_i_n + k_r_l] * mat_D[k_r_l_n + k_r_l] * mat_L[t_j_n + k_r_l];
                        #ifdef FLOP_COUNTER
                            flops()+=3;
                        #endif
                    }
                    
                    mat_A[s_i_n + j_t] = mat_A[s_i_n + j_t] - d_0;
                    #ifdef FLOP_COUNTER
                        flops()+=1;
                    #endif
                }

            }
        }
    }


    for(int j = k, j_n = k_n; j < n; j += r, j_n += r_n){
        remaining_col_num_j = min(r + j, n);
        for(int i = j, i_n = j_n; i_n < n_n; i += r, i_n += r_n){
            remaining_row_num = min(r, n-i);
            for(int s = 0, s_i_n = i_n; s < remaining_row_num; s++, s_i_n += n){
                for(j_t = j, t_j_n = j_n; j_t < remaining_col_num_j; t_j_n += n, j_t += 1){
                    d_1  = 0;

                    for(l_id = 1; l_id <= vec_ind[0]; l_id++){
                        k_r_l = vec_ind[l_id];
                        dij = mat_D[k_r_l * n + n + k_r_l];
                        d_1 += dij * (mat_L[s_i_n+ k_r_l+1] * mat_L[t_j_n + k_r_l] + mat_L[s_i_n + k_r_l] *  mat_L[t_j_n + k_r_l+1]);
                        #ifdef FLOP_COUNTER
                            flops()+=5;
                        #endif
                    }
                    
                    mat_A[s_i_n + j_t] = mat_A[s_i_n + j_t] - d_1;
                    #ifdef FLOP_COUNTER
                        flops()+=1;
                    #endif
                }
            }
        }
    }


}

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
    double* d_res = (double*)aligned_alloc(32,4*sizeof(double));

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

                        #ifdef FLOP_COUNTER
                            flops() += 2*4 + 2*8*4;
                        #endif
                        
                    }
                    d_vec_0 = _mm256_add_pd(d_vec_00, d_vec_04);
                    d_vec_1 = _mm256_add_pd(d_vec_10, d_vec_14);
                    d_vec_2 = _mm256_add_pd(d_vec_20, d_vec_24);
                    d_vec_3 = _mm256_add_pd(d_vec_30, d_vec_34); 

                    #ifdef FLOP_COUNTER
                        flops() += 4*4;
                    #endif

                    

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

                        #ifdef FLOP_COUNTER
                            flops() += 4 + 2*4*4;
                        #endif
                    }

                    d_vec_0 = _mm256_permute4x64_pd(_mm256_hadd_pd(d_vec_0, d_vec_2), 0b11011000);
                    d_vec_2 = _mm256_permute4x64_pd(_mm256_hadd_pd(d_vec_1, d_vec_3), 0b11011000);
                    d_vec = _mm256_hadd_pd(d_vec_0, d_vec_2);
                    
                    #ifdef FLOP_COUNTER
                        flops() += 4*3;
                    #endif

                    double temp;
                    __m256d temp_vec;
                    for( ; l < r; l++, k_r_l ++, k_r_l_n += n){
                        L_s_i_n_k_r_l_vec_0 =  _mm256_i64gather_pd((double*)(mat_L + t_j_n + k_r_l), jump_idx_n, sizeof(double));
                        temp = mat_L[s_i_n + k_r_l] * mat_D[k_r_l_n + k_r_l];
                        temp_vec =  _mm256_set1_pd(temp);
                        d_vec = _mm256_fmadd_pd(temp_vec, L_s_i_n_k_r_l_vec_0, d_vec);
                        #ifdef FLOP_COUNTER
                            flops() += 1+2*4;
                        #endif
                    }
                    
                    A_vec = _mm256_sub_pd(A_vec, d_vec);
                    #ifdef FLOP_COUNTER
                        flops() += 4;
                    #endif
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
                        #ifdef FLOP_COUNTER
                            flops() += 2*4 + 2*2*4;
                        #endif

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
                        #ifdef FLOP_COUNTER
                            flops() += 4 + 2*4;
                        #endif
                    }
                    d_vec_0 = _mm256_add_pd(d_vec_00, d_vec_04);
                    d_vec_0 = _mm256_hadd_pd(d_vec_0, d_vec_0);
                    _mm256_store_pd(d_res, d_vec_0);
                    d_0 = d_res[0] + d_res[3];
                    #ifdef FLOP_COUNTER
                        flops() += 2*4 + 1;
                    #endif
                    
                    for( ; l < r; l++, k_r_l ++, k_r_l_n += n){
                        d_0 +=  mat_L[s_i_n + k_r_l] * mat_D[k_r_l_n + k_r_l] * mat_L[t_j_n + k_r_l];
                        #ifdef FLOP_COUNTER
                            flops() += 3;
                        #endif
                    }
                    
                    mat_A[s_i_n + j_t] = mat_A[s_i_n + j_t] - d_0;

                    #ifdef FLOP_COUNTER
                        flops() += 1;
                    #endif

                }
                
                for(j_t = j, t_j_n = j_n; j_t < remaining_col_num_j; t_j_n += n, j_t += 1){
                    d_1  = 0;

                    for(l_id = 1; l_id <= vec_ind[0]; l_id++){
                        k_r_l = vec_ind[l_id];
                        dij = mat_D[k_r_l * n + n + k_r_l];
                        d_1 += dij * (mat_L[s_i_n+ k_r_l+1] * mat_L[t_j_n + k_r_l] + mat_L[s_i_n + k_r_l] *  mat_L[t_j_n + k_r_l+1]);
                    }
                    
                    mat_A[s_i_n + j_t] = mat_A[s_i_n + j_t] - d_1;
                    
                    #ifdef FLOP_COUNTER
                        flops() += vec_ind[0] * 5 + 1;
                    #endif
                }

            }

        }
    }
    free(d_res);
}


double test_function(void (*matrix_update)(double* mat_A, double* mat_D, double* mat_L, int* vec_ind, int n, int k, int r), 
                   int n, int repeat, int block_size){
    srand(time(NULL));
    bool correct = true;
    double speed_up = 0;

    myInt64 start, gt_time, block_time;
    gt_time = block_time = 0;

    double* A1 = (double*)malloc(n*n*sizeof(double));
    double* A2 = (double*)malloc(n*n*sizeof(double));
    double* L = (double*)malloc(n*n*sizeof(double));
    double* D = (double*)malloc(n*n*sizeof(double));
    int* pivot = (int*)malloc(n*sizeof(int));
    int* pivot_2 = (int*)malloc(block_size*sizeof(int));

    pivot_2[0] = 0;

    
    for(int i = 0; i < n; i++){
        if(i%block_size == 2 || i%block_size == 5 || i%block_size == 10){
            pivot[i] = 2;        
        }
        else if(i%block_size == 3 || i%block_size == 6 || i%block_size == 11){
            pivot[i] = 0;
        }        
        else{
            pivot[i] = 1;
        }
    }

    for(int i = 0; i < block_size; i++){
        if(i%block_size == 2 || i%block_size == 5 || i%block_size == 10){
            pivot_2[++pivot_2[0]] = i;
        }
    }
    // print_vector(pivot,n);

    // print_vector(pivot_2,n);
    // cout << pivot_2[pivot_2[0]]<< " "<<pivot_2[pivot_2[0]+1] << endl;


    for(int i = 0; i < repeat; i++){
        generate_random_symmetry(A1,n);
        generate_random_d(D,n,pivot);


        generate_random_l(L,n);

        matrix_transpose(A1,A2,n);

        start = start_tsc();
        matrix_update_gt(A1, D, L, pivot_2, n, block_size, block_size);
        gt_time += stop_tsc(start);

        start = start_tsc();
        matrix_update(A2, D, L, pivot_2, n, block_size, block_size);
        block_time += stop_tsc(start);   

        correct = compare_matrix(A1,A2, n,n);

        if(!correct){
            cout << "incorrect result!\n";
            return -1;
        }

    }
    free(A1);free(A2);free(L);free(pivot);free(D);

    speed_up = (double)gt_time/(double)block_time;
    return speed_up;
}



void compare_one(void (*matrix_update)(double* mat_A, double* mat_D, double* mat_L, int* vec_ind, int n, int k, int r),
                int n, double repeat, int block_size, double* A_template, double* A, double* L, double* D, int* pivot_2,
                vector<myInt64>& cycles_vec, vector<myInt64>& flops_vec){
    myInt64 start, time;
    time = 0;
    
    #ifdef FLOP_COUNTER
        flops()=0;
    #endif

    for(int i = 0; i < repeat; i++){
        matrix_transpose(A_template,A,n);
        #ifndef FLOP_COUNTER
        start = start_tsc();
        #endif
        matrix_update(A, D, L, pivot_2, n, block_size, block_size);
        #ifndef FLOP_COUNTER
        time += stop_tsc(start);
        #endif
    }

    #ifdef FLOP_COUNTER
        flops_vec.push_back(flops()/repeat);
    #else
        cycles_vec.push_back(time/repeat);
    #endif

}

void compare_all(int n, int block_size, double repeat, unsigned int random_seed,
                 vector<myInt64>& cycles_vec, vector<myInt64>& flops_vec){
    srand(random_seed);
    double* A_template = (double*)malloc(n*n*sizeof(double));
    double* A = (double*)malloc(n*n*sizeof(double));
    double* L = (double*)malloc(n*n*sizeof(double));
    double* D = (double*)malloc(n*n*sizeof(double));
    int* pivot = (int*)malloc(n*sizeof(int));
    int* pivot_2 = (int*)malloc(block_size*sizeof(int));

    pivot_2[0] = 0;

    
    for(int i = 0; i < n; i++){
        if(i%block_size == 2 || i%block_size == 5 || i%block_size == 10){
            pivot[i] = 2;        
        }
        else if(i%block_size == 3 || i%block_size == 6 || i%block_size == 11){
            pivot[i] = 0;
        }        
        else{
            pivot[i] = 1;
        }
    }

    for(int i = 0; i < block_size; i++){
        if(i%block_size == 2 || i%block_size == 5 || i%block_size == 10){
            pivot_2[++pivot_2[0]] = i;
        }
    }

    cycles_vec.clear();
    flops_vec.clear();
    
    generate_random_symmetry(A_template,n);
    generate_random_d(D,n,pivot);
    generate_random_l(L,n);

    compare_one(matrix_update_gt,n,repeat,block_size,A_template,A,L,D,pivot_2,cycles_vec,flops_vec);
    compare_one(matrix_update_ijts,n,repeat,block_size,A_template,A,L,D,pivot_2,cycles_vec,flops_vec);
    compare_one(matrix_update_sparse_d,n,repeat,block_size,A_template,A,L,D,pivot_2,cycles_vec,flops_vec);
    compare_one(matrix_update_sparse_d_unroll,n,repeat,block_size,A_template,A,L,D,pivot_2,cycles_vec,flops_vec);
    compare_one(matrix_update_sparse_d_unroll_rename,n,repeat,block_size,A_template,A,L,D,pivot_2,cycles_vec,flops_vec);
    compare_one(matrix_update_sparse_d_unroll_rename_vec,n,repeat,block_size,A_template,A,L,D,pivot_2,cycles_vec,flops_vec);
    compare_one(matrix_update_sparse_d_unroll_rename_vec_tail,n,repeat,block_size,A_template,A,L,D,pivot_2,cycles_vec,flops_vec);

    free(A_template);
    free(A);
    free(L);
    free(D);
    free(pivot);
    free(pivot_2);

}



int main(){
    int n_start = 100;
    int n_end = 3000;
    int n_gap = 200; 
    int n_num = 0;
    for(int i = n_start; i < n_end; i+=n_gap){
        n_num ++;
    }

    int block_size = 32;
    double repeat = 10;
    unsigned int random_seed = 2;
    vector<vector<myInt64>> cycles_vec(n_num,vector<myInt64>()); 
    vector<vector<myInt64>> flops_vec(n_num,vector<myInt64>());
    vector<string> name_vec;
    name_vec.push_back("matrix_update_gt");
    name_vec.push_back("matrix_update_ijts");
    name_vec.push_back("matrix_update_sparse_d");
    name_vec.push_back("matrix_update_sparse_d_unroll");
    name_vec.push_back("matrix_update_sparse_d_unroll_rename");
    name_vec.push_back("matrix_update_sparse_d_unroll_rename_vec");
    name_vec.push_back("matrix_update_sparse_d_unroll_rename_vec_tail");

    for(int i = 0, n=n_start; i < n_num; i++, n+=n_gap){
        compare_all(n,block_size,repeat,random_seed,cycles_vec[i],flops_vec[i]);
    }

    cout << "n_start: " << n_start << endl << "n_end: " << n_end << endl << "n_gap: " << n_gap << endl;
    cout << "block_size: " << block_size << endl;

    cout << "evaluate_func_name: [\"";
    for(int i = 0; i < name_vec.size()-1; i++){
        cout << name_vec[i] << "\", \"";
    }
    cout << name_vec[name_vec.size()-1] << "\"]\n";
    #ifdef FLOP_COUNTER
        cout << "evaluate_func_flops: [";
        for(int j = 0; j < n_num; j++){
            cout << "[";
            for(int i = 0; i < flops_vec[j].size()-1; i++){
                cout << flops_vec[j][i] << ", ";
            }
            cout << flops_vec[j][flops_vec[j].size()-1];
            cout << "]";
            if(j < n_num-1){
                cout << ",";
            }
        }
        cout <<  "]\n";
    #else
        cout << "evaluate_func_cycles: [";
        for(int j = 0; j < n_num; j++){
            cout << "[";
            for(int i = 0; i < cycles_vec[j].size()-1; i++){
                cout << cycles_vec[j][i] << ", ";
            }
            cout << cycles_vec[j][cycles_vec[j].size()-1];
            cout << "]";
            if(j < n_num-1){
                cout << ",";
            }
        }
        cout <<"]\n";
    #endif
    

    return 0;
}