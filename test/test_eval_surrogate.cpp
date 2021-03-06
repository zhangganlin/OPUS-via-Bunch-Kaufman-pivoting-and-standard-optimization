#include <iostream>
#include "LinearSolver.h"
#include <iostream>
#include "tsc_x86.h"
#include "randomlhs.hpp"
#include <time.h>
#include <immintrin.h>
#include <cstring>
#include "test_utils.h"
#include <vector>
#include <string>

#define FLOP_COUNTER

using namespace std;
static double sqrtsd (double x) {
    double r;
    __asm__ ("sqrtsd %1, %0" : "=x" (r) : "x" (x));
    return r;
}

void evaluate_surrogate_gt( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    for(int pa = 0; pa < N_x; pa++){
        double phi, error, res = 0, sq_phi;
        // flops: 3Nd + 5N
        for(int pb = 0; pb < N_points; pb++){
            phi = 0;
            // flops: 3d
            for(int j = 0; j < d; j++){
                error = x[pa * d + j] - points[pb * d + j];
                phi += error * error;
            }
            phi = sqrt(phi);            // flops: 1
            phi = phi * phi * phi;      // flops: 2
            res += phi * lambda_c[pb];   // flops: 2
            #ifdef FLOP_COUNTER
                flops()+=d*3+5;
            #endif
        }
        // flops: 2d
        for(int pb = 0; pb < d; pb++){
            res += x[pa * d + pb] * lambda_c[N_points + pb];
        }
        // flops: 1
        res += lambda_c[N_points + d];
        #ifdef FLOP_COUNTER
            flops()+=d*2+1;
        #endif

        output[pa] = res;
    }
}

void evaluate_surrogate_rename( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    for(int pa = 0; pa < N_x; pa++){
        double phi, error, res = 0, sq_phi;
        int pa_d = pa * d, pb_d;
        // flops: 3Nd + 5N
        for(int pb = 0; pb < N_points; pb++){
            phi = 0;
            pb_d = pb * d;
            // flops: 3d
            for(int j = 0; j < d; j++){
                error = x[pa_d + j] - points[pb_d + j];
                phi += error * error;
            }
            sq_phi = sqrt(phi);            // flops: 1
            phi = sq_phi * phi;      // flops: 1
            res += phi * lambda_c[pb];   // flops: 2
        }
        // flops: 2d
        for(int pb = 0; pb < d; pb++){
            res += x[pa_d + pb] * lambda_c[N_points + pb];
        }
        // flops: 1
        res += lambda_c[N_points + d];
        
        #ifdef FLOP_COUNTER
            flops()+=N_points*(d*3+4)+d*2+1;
        #endif

        output[pa] = res;
    }
}

void evaluate_surrogate_reorder_rename(double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    double* phi = (double*)malloc(sizeof(double) * N_points);
    double error, res, sq_phi, x_pa_j, sqrt_phi, temp_phi;
    int pa_d;
    for(int pa = 0, pa_d = 0; pa < N_x; pa++, pa_d += d){
        res = 0;
        memset(phi, 0, sizeof(phi));
        for(int j = 0; j < d; j++){
            x_pa_j = x[pa_d + j];
            for(int pb = 0, pb_d = 0; pb < N_points; pb++, pb_d += d){
                error = x_pa_j - points[pb_d+ j];
                phi[pb] += error * error;
                #ifdef FLOP_COUNTER
                    flops()+=3;
                #endif
            }
        }
        
        for(int pb = 0; pb < N_points; pb++){
            temp_phi = phi[pb];
            sqrt_phi = sqrt(temp_phi); 
            temp_phi = sqrt_phi * temp_phi;   
            res += temp_phi * lambda_c[pb];   
        }

        // flops: 2d
        for(int i = 0; i < d; i++){
            res += x[pa_d + i] * lambda_c[N_points + i];
        }
        // flops: 1
        res += lambda_c[N_points + d];

        #ifdef FLOP_COUNTER
            flops()+=N_points*4 + d*2 +1;
        #endif

        output[pa] = res;
    }
    free(phi);
}

void evaluate_surrogate_reorder( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    double* phi = (double*)malloc(sizeof(double) * N_points);
    double error, res, sq_phi, pa_j, sqrt_phi, temp_phi;
    for(int pa = 0; pa < N_x; pa++){
        res = 0;
        memset(phi, 0, sizeof(phi));
        for(int j = 0; j < d; j++){
            pa_j = x[pa * d + j];
            for(int pb = 0; pb < N_points; pb++){
                error = pa_j - points[pb * d + j];
                phi[pb] += error * error;
                #ifdef FLOP_COUNTER
                    flops()+=3;
                #endif
            }
        }
        for(int pb = 0; pb < N_points; pb++){
            temp_phi = phi[pb];
            sqrt_phi = sqrt(temp_phi); 
            temp_phi = sqrt_phi * temp_phi;   
            res += temp_phi * lambda_c[pb];   
        }

        // flops: 2d
        for(int i = 0; i < d; i++){
            res += x[pa * d + i] * lambda_c[N_points + i];
        }
        // flops: 1
        res += lambda_c[N_points + d];
        #ifdef FLOP_COUNTER
            flops()+=N_points*4 + d*2 +1;
        #endif
        output[pa] = res;
    }
    free(phi);
}

void evaluate_surrogate_unroll_8( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    double res, sq_phi;
    double phi, phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_res; 
    double error_0, error_1, error_2, error_3, error_4, error_5, error_6, error_7, error;
    int id, j, pa_d = 0, pb_d, pa_d_j, pb_d_j;
    for(int pa = 0; pa < N_x; pa++, pa_d += d){
        res = 0;
        // flops: 3Nd + 5N
        pb_d = 0;
        for(int pb = 0; pb < N_points; pb++, pb_d += d){
            phi_0 = phi_1 = phi_2 = phi_3 = phi_4 = phi_5 = phi_6 = phi_7 = 0; 
            j = 0;
            for(; j + 7 < d; j += 8){
                pa_d_j = pa_d + j, pb_d_j = pb_d + j;
                error_0 = x[pa_d_j] - points[pb_d + j];
                error_1 = x[pa_d_j + 1] - points[pb_d_j + 1];
                error_2 = x[pa_d_j + 2] - points[pb_d_j + 2];
                error_3 = x[pa_d_j + 3] - points[pb_d_j + 3];
                error_4 = x[pa_d_j + 4] - points[pb_d_j + 4];
                error_5 = x[pa_d_j + 5] - points[pb_d_j + 5];
                error_6 = x[pa_d_j + 6] - points[pb_d_j + 6];
                error_7 = x[pa_d_j + 7] - points[pb_d_j + 7];
                phi_0 += error_0 * error_0; 
                phi_1 += error_1 * error_1; 
                phi_2 += error_2 * error_2; 
                phi_3 += error_3 * error_3; 
                phi_4 += error_4 * error_4; 
                phi_5 += error_5 * error_5; 
                phi_6 += error_6 * error_6; 
                phi_7 += error_7 * error_7; 

                #ifdef FLOP_COUNTER
                    flops()+=24;
                #endif
            }
            phi = phi_0 + phi_1 + phi_2 + phi_3 + phi_4 + phi_5 + phi_6 + phi_7;
            #ifdef FLOP_COUNTER
                    flops()+=7;
                #endif
            phi_res = 0;
            for(; j < d; j++){
                error = x[pa_d + j] - points[pb_d + j];
                phi_res += error * error;
                #ifdef FLOP_COUNTER
                    flops()+=3;
                #endif
            }
            phi += phi_res;

            sq_phi = sqrt(phi);              // flops: 1
            phi = phi * sq_phi;                     // flops: 2
            res += phi * lambda_c[pb];               // flops: 2
            #ifdef FLOP_COUNTER
                flops()+=4;
            #endif
        }
        // flops: 2d
        for(int i = 0; i < d; i++){
            res += x[pa_d + i] * lambda_c[N_points + i];
            #ifdef FLOP_COUNTER
                flops()+=2;
            #endif
        }
        // flops: 1
        res += lambda_c[N_points + d];
        #ifdef FLOP_COUNTER
            flops()+=1;
        #endif

        output[pa] = res;
    }
}

void evaluate_surrogate_unroll_8_sqrt( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    double res, sq_phi;
    double phi, phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_res; 
    double error_0, error_1, error_2, error_3, error_4, error_5, error_6, error_7, error;
    double* history_phi = (double*)malloc(sizeof(double) * N_points);
    int id, j, pa_d = 0, pb_d, pa_d_j, pb_d_j;
    for(int pa = 0; pa < N_x; pa++, pa_d += d){
        res = 0;
        // flops: 3Nd + 5N
        pb_d = 0;
        for(int pb = 0; pb < N_points; pb++, pb_d += d){
            phi_0 = phi_1 = phi_2 = phi_3 = phi_4 = phi_5 = phi_6 = phi_7 = 0; 
            j = 0;
            for(; j + 7 < d; j += 8){
                pa_d_j = pa_d + j, pb_d_j = pb_d + j;
                error_0 = x[pa_d_j] - points[pb_d + j];
                error_1 = x[pa_d_j + 1] - points[pb_d_j + 1];
                error_2 = x[pa_d_j + 2] - points[pb_d_j + 2];
                error_3 = x[pa_d_j + 3] - points[pb_d_j + 3];
                error_4 = x[pa_d_j + 4] - points[pb_d_j + 4];
                error_5 = x[pa_d_j + 5] - points[pb_d_j + 5];
                error_6 = x[pa_d_j + 6] - points[pb_d_j + 6];
                error_7 = x[pa_d_j + 7] - points[pb_d_j + 7];
                phi_0 += error_0 * error_0; 
                phi_1 += error_1 * error_1; 
                phi_2 += error_2 * error_2; 
                phi_3 += error_3 * error_3; 
                phi_4 += error_4 * error_4; 
                phi_5 += error_5 * error_5; 
                phi_6 += error_6 * error_6; 
                phi_7 += error_7 * error_7; 
                #ifdef FLOP_COUNTER
                    flops()+= 8+16;
                #endif
            }
            phi = phi_0 + phi_1 + phi_2 + phi_3 + phi_4 + phi_5 + phi_6 + phi_7;
            #ifdef FLOP_COUNTER
                flops()+= 7;
            #endif
            phi_res = 0;
            for(; j < d; j++){
                error = x[pa_d + j] - points[pb_d + j];
                phi_res += error * error;
                #ifdef FLOP_COUNTER
                    flops()+= 3;
                #endif
            }
            phi += phi_res;
            #ifdef FLOP_COUNTER
                flops()+= 1;
            #endif
            history_phi[pb] = phi;
        }
        
        for(int pb = 0; pb < N_points; pb++){
            phi = history_phi[pb];
            sq_phi = sqrt(phi);              // flops: 1
            phi = phi * sq_phi;                     // flops: 2
            res += phi * lambda_c[pb];               // flops: 2
        }
        // flops: 2d
        for(int i = 0; i < d; i++){
            res += x[pa_d + i] * lambda_c[N_points + i];
        }
        // flops: 1
        res += lambda_c[N_points + d];
        #ifdef FLOP_COUNTER
            flops()+= N_points*4+d*2+1;
        #endif
        output[pa] = res;
    }
    free(history_phi);
}

void evaluate_surrogate_unroll_8_sqrt_sample_vec( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    __m256d phi_vec, phi_0_vec, phi_1_vec, phi_2_vec, phi_3_vec, phi_4_vec, phi_5_vec, phi_6_vec, phi_7_vec, phi_res_vec, sq_phi_vec; 
    __m256d error_0_vec, error_1_vec, error_2_vec, error_3_vec, error_4_vec, error_5_vec, error_6_vec, error_7_vec;
    __m256d x_vec_0, x_vec_1, x_vec_2, x_vec_3, x_vec_4, x_vec_5, x_vec_6, x_vec_7;
    __m256d points_vec_0, points_vec_1, points_vec_2, points_vec_3, points_vec_4, points_vec_5, points_vec_6, points_vec_7;
    __m256d lambda_c_vec, res_vec;
    __m256i jump_idx = _mm256_set_epi64x(3*d, 2*d, d, 0);

    double* history_phi_vec = (double*)malloc(sizeof(double) * N_points * 4);
    int id, j, pa_d = 0, pb_d, pa_d_j, pb_d_j, d_4 = 4*d, pa = 0;
    for(; pa + 3 < N_x; pa += 4, pa_d += d_4){
        // flops: 3Nd + 5N
        pb_d = 0;
        for(int pb = 0; pb < N_points; pb++, pb_d += d){
            phi_0_vec = phi_1_vec = phi_2_vec = phi_3_vec = phi_4_vec = phi_5_vec = phi_6_vec = phi_7_vec = _mm256_set1_pd(0);
            j = 0;
            for(; j + 7 < d; j += 8){
                pa_d_j = pa_d + j, pb_d_j = pb_d + j;
                x_vec_0 = _mm256_i64gather_pd((double*)(x + pa_d_j), jump_idx, sizeof(double));
                x_vec_1 = _mm256_i64gather_pd((double*)(x + pa_d_j + 1), jump_idx, sizeof(double));
                x_vec_2 = _mm256_i64gather_pd((double*)(x + pa_d_j + 2), jump_idx, sizeof(double));
                x_vec_3 = _mm256_i64gather_pd((double*)(x + pa_d_j + 3), jump_idx, sizeof(double));
                x_vec_4 = _mm256_i64gather_pd((double*)(x + pa_d_j + 4), jump_idx, sizeof(double));
                x_vec_5 = _mm256_i64gather_pd((double*)(x + pa_d_j + 5), jump_idx, sizeof(double));
                x_vec_6 = _mm256_i64gather_pd((double*)(x + pa_d_j + 6), jump_idx, sizeof(double));
                x_vec_7 = _mm256_i64gather_pd((double*)(x + pa_d_j + 7), jump_idx, sizeof(double));

                points_vec_0 = _mm256_broadcast_sd(points + pb_d_j);
                points_vec_1 = _mm256_broadcast_sd(points + pb_d_j + 1);
                points_vec_2 = _mm256_broadcast_sd(points + pb_d_j + 2);
                points_vec_3 = _mm256_broadcast_sd(points + pb_d_j + 3);
                points_vec_4 = _mm256_broadcast_sd(points + pb_d_j + 4);
                points_vec_5 = _mm256_broadcast_sd(points + pb_d_j + 5);
                points_vec_6 = _mm256_broadcast_sd(points + pb_d_j + 6);
                points_vec_7 = _mm256_broadcast_sd(points + pb_d_j + 7);

                error_0_vec = _mm256_sub_pd(x_vec_0, points_vec_0);
                error_1_vec = _mm256_sub_pd(x_vec_1, points_vec_1);
                error_2_vec = _mm256_sub_pd(x_vec_2, points_vec_2);
                error_3_vec = _mm256_sub_pd(x_vec_3, points_vec_3);
                error_4_vec = _mm256_sub_pd(x_vec_4, points_vec_4);
                error_5_vec = _mm256_sub_pd(x_vec_5, points_vec_5);
                error_6_vec = _mm256_sub_pd(x_vec_6, points_vec_6);
                error_7_vec = _mm256_sub_pd(x_vec_7, points_vec_7);

                phi_0_vec = _mm256_fmadd_pd(error_0_vec, error_0_vec, phi_0_vec); 
                phi_1_vec = _mm256_fmadd_pd(error_1_vec, error_1_vec, phi_1_vec); 
                phi_2_vec = _mm256_fmadd_pd(error_2_vec, error_2_vec, phi_2_vec); 
                phi_3_vec = _mm256_fmadd_pd(error_3_vec, error_3_vec, phi_3_vec); 
                phi_4_vec = _mm256_fmadd_pd(error_4_vec, error_4_vec, phi_4_vec); 
                phi_5_vec = _mm256_fmadd_pd(error_5_vec, error_5_vec, phi_5_vec); 
                phi_6_vec = _mm256_fmadd_pd(error_6_vec, error_6_vec, phi_6_vec); 
                phi_7_vec = _mm256_fmadd_pd(error_7_vec, error_7_vec, phi_7_vec); 

                #ifdef FLOP_COUNTER
                    flops()+=24 * 4;
                #endif
            }
            phi_vec = _mm256_add_pd(
                _mm256_add_pd(
                    _mm256_add_pd(phi_0_vec, phi_1_vec), 
                    _mm256_add_pd(phi_2_vec, phi_3_vec)
                ),  
                _mm256_add_pd(
                    _mm256_add_pd(phi_4_vec, phi_5_vec), 
                    _mm256_add_pd(phi_6_vec, phi_7_vec)
                )
            );
            #ifdef FLOP_COUNTER
                flops()+=7 * 4;
            #endif

            for(; j < d; j++){
                x_vec_0 = _mm256_i64gather_pd((double*)(x + pa_d + j), jump_idx, sizeof(double));
                points_vec_0 = _mm256_broadcast_sd(points + pb_d + j);
                error_0_vec = _mm256_sub_pd(x_vec_0, points_vec_0);
                phi_vec = _mm256_fmadd_pd(error_0_vec, error_0_vec, phi_vec); 

                #ifdef FLOP_COUNTER
                    flops()+=3 * 4;
                #endif
            }

            _mm256_storeu_pd((double*)(history_phi_vec + pb * 4), phi_vec);
        }
        res_vec = _mm256_set1_pd(0);
        for(int pb = 0; pb < N_points; pb++){
            phi_vec = _mm256_loadu_pd((double*)(history_phi_vec + pb*4));
            lambda_c_vec = _mm256_broadcast_sd((double*)(lambda_c + pb));
            sq_phi_vec = _mm256_sqrt_pd(phi_vec);
            phi_vec = _mm256_mul_pd(phi_vec, sq_phi_vec);
            res_vec = _mm256_fmadd_pd(phi_vec, lambda_c_vec, res_vec);

            #ifdef FLOP_COUNTER
                flops()+=4 * 4;
            #endif
        }
        // flops: 2d
        for(int i = 0; i < d; i++){
            x_vec_0 = _mm256_i64gather_pd((double*)(x + pa_d + i), jump_idx, sizeof(double));
            lambda_c_vec = _mm256_broadcast_sd((double*)(lambda_c + N_points + i));
            res_vec =  _mm256_fmadd_pd(x_vec_0, lambda_c_vec, res_vec);

            #ifdef FLOP_COUNTER
                flops()+=2 * 4;
            #endif
        }
        // flops: 1
        lambda_c_vec = _mm256_broadcast_sd((double*)(lambda_c + N_points + d));
        res_vec = _mm256_add_pd(res_vec, lambda_c_vec);
        _mm256_storeu_pd((double*)(output + pa), res_vec);
        #ifdef FLOP_COUNTER
            flops()+=1 * 4;
        #endif
    }

    double res, sq_phi, error;
    double phi, phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_res; 
    double error_0, error_1, error_2, error_3, error_4, error_5, error_6, error_7;
    double* history_phi = (double*)malloc(sizeof(double) * N_points);
    for(; pa < N_x; pa++, pa_d += d){
        res = 0;
        // flops: 3Nd + 5N
        pb_d = 0;
        for(int pb = 0; pb < N_points; pb++, pb_d += d){
            phi_0 = phi_1 = phi_2 = phi_3 = phi_4 = phi_5 = phi_6 = phi_7 = 0; 
            j = 0;
            for(; j + 7 < d; j += 8){
                pa_d_j = pa_d + j, pb_d_j = pb_d + j;
                error_0 = x[pa_d_j] - points[pb_d + j];
                error_1 = x[pa_d_j + 1] - points[pb_d_j + 1];
                error_2 = x[pa_d_j + 2] - points[pb_d_j + 2];
                error_3 = x[pa_d_j + 3] - points[pb_d_j + 3];
                error_4 = x[pa_d_j + 4] - points[pb_d_j + 4];
                error_5 = x[pa_d_j + 5] - points[pb_d_j + 5];
                error_6 = x[pa_d_j + 6] - points[pb_d_j + 6];
                error_7 = x[pa_d_j + 7] - points[pb_d_j + 7];
                phi_0 += error_0 * error_0; 
                phi_1 += error_1 * error_1; 
                phi_2 += error_2 * error_2; 
                phi_3 += error_3 * error_3; 
                phi_4 += error_4 * error_4; 
                phi_5 += error_5 * error_5; 
                phi_6 += error_6 * error_6; 
                phi_7 += error_7 * error_7; 

                #ifdef FLOP_COUNTER
                    flops()+=24;
                #endif
            }
            phi = phi_0 + phi_1 + phi_2 + phi_3 + phi_4 + phi_5 + phi_6 + phi_7;
            phi_res = 0;
            for(; j < d; j++){
                error = x[pa_d + j] - points[pb_d + j];
                phi_res += error * error;

                #ifdef FLOP_COUNTER
                    flops()+=3;
                #endif
            }
            phi += phi_res;

            #ifdef FLOP_COUNTER
                flops()+=1;
            #endif

            history_phi[pb] = phi;
        }
        
        for(int pb = 0; pb < N_points; pb++){
            phi = history_phi[pb];
            sq_phi = sqrt(phi);              // flops: 1
            phi = phi * sq_phi;                     // flops: 1
            res += phi * lambda_c[pb];               // flops: 2

            #ifdef FLOP_COUNTER
                flops()+=4;
            #endif
        }
        // flops: 2d
        for(int i = 0; i < d; i++){
            res += x[pa_d + i] * lambda_c[N_points + i];

            #ifdef FLOP_COUNTER
                flops()+=2;
            #endif
        }
        // flops: 1
        res += lambda_c[N_points + d];

        #ifdef FLOP_COUNTER
            flops()+=1;
        #endif
        output[pa] = res;
    }
    free(history_phi);
    free(history_phi_vec);
}

void evaluate_surrogate_unroll_8_sqrt_sample_vec_optimize_load( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    __m256d phi_vec, phi_0_vec, phi_1_vec, phi_2_vec, phi_3_vec, phi_01_vec, phi_23_vec, phi_00_vec, phi_04_vec, phi_10_vec, phi_14_vec, phi_20_vec, phi_24_vec, phi_30_vec, phi_34_vec, phi_res_vec, sq_phi_vec; 
    __m256d error_00_vec, error_04_vec, error_10_vec, error_14_vec, error_20_vec, error_24_vec, error_30_vec, error_34_vec;
    __m256d x_vec_00, x_vec_04, x_vec_10, x_vec_14, x_vec_20, x_vec_24, x_vec_30, x_vec_34;
    __m256d points_vec_0, points_vec_4;
    __m256d lambda_c_vec, res_vec, res_vec_0, res_vec_1, res_vec_2, res_vec_3, res_vec_01, res_vec_23;
    __m256i jump_idx = _mm256_set_epi64x(3*d, 2*d, d, 0);

    double* history_phi_vec = (double*)malloc(sizeof(double) * N_points * 4);
    int id, j, pa_d = 0, pb_d, pa_d_j, pb_d_j, d_4 = 4*d, pa = 0;
    for(; pa + 3 < N_x; pa += 4, pa_d += d_4){
        // flops: 3Nd + 5N
        pb_d = 0;
        for(int pb = 0; pb < N_points; pb++, pb_d += d){
            phi_00_vec = phi_04_vec = phi_10_vec = phi_14_vec = phi_20_vec = phi_24_vec = phi_30_vec = phi_34_vec = _mm256_set1_pd(0);
            j = 0;
            for(; j + 7 < d; j += 8){
                pa_d_j = pa_d + j, pb_d_j = pb_d + j;
                x_vec_00 = _mm256_loadu_pd((double*)(x + pa_d_j));
                x_vec_04 = _mm256_loadu_pd((double*)(x + pa_d_j + 4));
                x_vec_10 = _mm256_loadu_pd((double*)(x + pa_d_j + d));
                x_vec_14 = _mm256_loadu_pd((double*)(x + pa_d_j + d + 4));
                x_vec_20 = _mm256_loadu_pd((double*)(x + pa_d_j + 2 * d));
                x_vec_24 = _mm256_loadu_pd((double*)(x + pa_d_j + 2 * d + 4));
                x_vec_30 = _mm256_loadu_pd((double*)(x + pa_d_j + 3 * d));
                x_vec_34 = _mm256_loadu_pd((double*)(x + pa_d_j + 3 * d + 4));

                points_vec_0 = _mm256_loadu_pd(points + pb_d_j);
                points_vec_4 = _mm256_loadu_pd(points + pb_d_j + 4);

                error_00_vec = _mm256_sub_pd(x_vec_00, points_vec_0);
                error_04_vec = _mm256_sub_pd(x_vec_04, points_vec_4);
                error_10_vec = _mm256_sub_pd(x_vec_10, points_vec_0);
                error_14_vec = _mm256_sub_pd(x_vec_14, points_vec_4);
                error_20_vec = _mm256_sub_pd(x_vec_20, points_vec_0);
                error_24_vec = _mm256_sub_pd(x_vec_24, points_vec_4);
                error_30_vec = _mm256_sub_pd(x_vec_30, points_vec_0);
                error_34_vec = _mm256_sub_pd(x_vec_34, points_vec_4);

                phi_00_vec = _mm256_fmadd_pd(error_00_vec, error_00_vec, phi_00_vec); 
                phi_04_vec = _mm256_fmadd_pd(error_04_vec, error_04_vec, phi_04_vec); 
                phi_10_vec = _mm256_fmadd_pd(error_10_vec, error_10_vec, phi_10_vec); 
                phi_14_vec = _mm256_fmadd_pd(error_14_vec, error_14_vec, phi_14_vec); 
                phi_20_vec = _mm256_fmadd_pd(error_20_vec, error_20_vec, phi_20_vec); 
                phi_24_vec = _mm256_fmadd_pd(error_24_vec, error_24_vec, phi_24_vec); 
                phi_30_vec = _mm256_fmadd_pd(error_30_vec, error_30_vec, phi_30_vec); 
                phi_34_vec = _mm256_fmadd_pd(error_34_vec, error_34_vec, phi_34_vec); 

                #ifdef FLOP_COUNTER
                    flops() += 32+64;
                #endif
            }
            phi_0_vec = _mm256_add_pd(phi_00_vec, phi_04_vec);
            phi_1_vec = _mm256_add_pd(phi_10_vec, phi_14_vec);
            phi_2_vec = _mm256_add_pd(phi_20_vec, phi_24_vec);
            phi_3_vec = _mm256_add_pd(phi_30_vec, phi_34_vec); 
            #ifdef FLOP_COUNTER
                flops() += 16;
            #endif

            for(; j + 3 < d; j += 4){
                pa_d_j = pa_d + j, pb_d_j = pb_d + j;
                x_vec_00 = _mm256_loadu_pd((double*)(x + pa_d_j));
                x_vec_10 = _mm256_loadu_pd((double*)(x + pa_d_j + d));
                x_vec_20 = _mm256_loadu_pd((double*)(x + pa_d_j + 2 * d));
                x_vec_30 = _mm256_loadu_pd((double*)(x + pa_d_j + 3 * d));
                points_vec_0 = _mm256_loadu_pd(points + pb_d_j);

                error_00_vec = _mm256_sub_pd(x_vec_00, points_vec_0);
                error_10_vec = _mm256_sub_pd(x_vec_10, points_vec_0);
                error_20_vec = _mm256_sub_pd(x_vec_20, points_vec_0);
                error_30_vec = _mm256_sub_pd(x_vec_30, points_vec_0);

                phi_0_vec = _mm256_fmadd_pd(error_00_vec, error_00_vec, phi_0_vec); 
                phi_1_vec = _mm256_fmadd_pd(error_10_vec, error_10_vec, phi_1_vec); 
                phi_2_vec = _mm256_fmadd_pd(error_20_vec, error_20_vec, phi_2_vec); 
                phi_3_vec = _mm256_fmadd_pd(error_30_vec, error_30_vec, phi_3_vec); 

                #ifdef FLOP_COUNTER
                    flops() += 16 + 32;
                #endif
            }
            phi_01_vec = _mm256_permute4x64_pd(_mm256_hadd_pd(phi_0_vec, phi_2_vec), 0b11011000);
            phi_23_vec = _mm256_permute4x64_pd(_mm256_hadd_pd(phi_1_vec, phi_3_vec), 0b11011000);

            phi_vec = _mm256_hadd_pd(phi_01_vec, phi_23_vec);
            #ifdef FLOP_COUNTER
                flops() += 3 * 4;
            #endif

            for(; j < d; j++){
                x_vec_00 = _mm256_i64gather_pd((double*)(x + pa_d + j), jump_idx, sizeof(double));
                points_vec_0 = _mm256_broadcast_sd(points + pb_d + j);
                error_00_vec = _mm256_sub_pd(x_vec_00, points_vec_0);
                phi_vec = _mm256_fmadd_pd(error_00_vec, error_00_vec, phi_vec); 
                #ifdef FLOP_COUNTER
                    flops() += 4+8;
                #endif
            }
            _mm256_storeu_pd((double*)(history_phi_vec + pb * 4), phi_vec);
        }
        res_vec = _mm256_set1_pd(0);
        for(int pb = 0; pb < N_points; pb++){
            phi_vec = _mm256_loadu_pd((double*)(history_phi_vec + pb*4));
            lambda_c_vec = _mm256_broadcast_sd((double*)(lambda_c + pb));
            sq_phi_vec = _mm256_sqrt_pd(phi_vec);
            phi_vec = _mm256_mul_pd(phi_vec, sq_phi_vec);
            res_vec = _mm256_fmadd_pd(phi_vec, lambda_c_vec, res_vec);
            #ifdef FLOP_COUNTER
                    flops() += 4+4+8;
            #endif
        }
        // flops: 2d
        
        res_vec_0 = res_vec_1 = res_vec_2 = res_vec_3 = _mm256_set1_pd(0);
        int i;
        for(i = 0; i + 3 < d; i += 4){
            x_vec_00 = _mm256_loadu_pd((double*)(x + pa_d + i));
            x_vec_10 = _mm256_loadu_pd((double*)(x + pa_d + d + i));
            x_vec_20 = _mm256_loadu_pd((double*)(x + pa_d + 2 * d + i));
            x_vec_30 = _mm256_loadu_pd((double*)(x + pa_d + 3 * d + i));
            lambda_c_vec = _mm256_loadu_pd((double*)(lambda_c + N_points + i));
            res_vec_0 =  _mm256_fmadd_pd(x_vec_00, lambda_c_vec, res_vec_0);
            res_vec_1 =  _mm256_fmadd_pd(x_vec_10, lambda_c_vec, res_vec_1);
            res_vec_2 =  _mm256_fmadd_pd(x_vec_20, lambda_c_vec, res_vec_2);
            res_vec_3 =  _mm256_fmadd_pd(x_vec_30, lambda_c_vec, res_vec_3);
            #ifdef FLOP_COUNTER
                flops() += 32;
            #endif
        }
        res_vec_01 = _mm256_permute4x64_pd(_mm256_hadd_pd(res_vec_0, res_vec_2), 0b11011000);
        res_vec_23 = _mm256_permute4x64_pd(_mm256_hadd_pd(res_vec_1, res_vec_3), 0b11011000);

        res_vec = _mm256_add_pd(_mm256_hadd_pd(res_vec_01, res_vec_23), res_vec);
        #ifdef FLOP_COUNTER
            flops() += 4 * 4;
        #endif
        for(; i < d; i++){
            x_vec_00 = _mm256_i64gather_pd((double*)(x + pa_d + i), jump_idx, sizeof(double));
            lambda_c_vec = _mm256_broadcast_sd((double*)(lambda_c + N_points + i));
            res_vec =  _mm256_fmadd_pd(x_vec_00, lambda_c_vec, res_vec);
            #ifdef FLOP_COUNTER
                flops() += 8;
            #endif
        }

        // flops: 1
        lambda_c_vec = _mm256_broadcast_sd((double*)(lambda_c + N_points + d));
        res_vec = _mm256_add_pd(res_vec, lambda_c_vec);
        #ifdef FLOP_COUNTER
            flops() += 4;
        #endif
        _mm256_storeu_pd((double*)(output + pa), res_vec);
    }

    double res, sq_phi, error;
    double phi, phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_res; 
    double error_0, error_1, error_2, error_3, error_4, error_5, error_6, error_7;
    double* history_phi = (double*)malloc(sizeof(double) * N_points);
    for(; pa < N_x; pa++, pa_d += d){
        res = 0;
        // flops: 3Nd + 5N
        pb_d = 0;
        for(int pb = 0; pb < N_points; pb++, pb_d += d){
            phi_0 = phi_1 = phi_2 = phi_3 = phi_4 = phi_5 = phi_6 = phi_7 = 0; 
            j = 0;
            for(; j + 7 < d; j += 8){
                pa_d_j = pa_d + j, pb_d_j = pb_d + j;
                error_0 = x[pa_d_j] - points[pb_d + j];
                error_1 = x[pa_d_j + 1] - points[pb_d_j + 1];
                error_2 = x[pa_d_j + 2] - points[pb_d_j + 2];
                error_3 = x[pa_d_j + 3] - points[pb_d_j + 3];
                error_4 = x[pa_d_j + 4] - points[pb_d_j + 4];
                error_5 = x[pa_d_j + 5] - points[pb_d_j + 5];
                error_6 = x[pa_d_j + 6] - points[pb_d_j + 6];
                error_7 = x[pa_d_j + 7] - points[pb_d_j + 7];
                phi_0 += error_0 * error_0; 
                phi_1 += error_1 * error_1; 
                phi_2 += error_2 * error_2; 
                phi_3 += error_3 * error_3; 
                phi_4 += error_4 * error_4; 
                phi_5 += error_5 * error_5; 
                phi_6 += error_6 * error_6; 
                phi_7 += error_7 * error_7; 
                #ifdef FLOP_COUNTER
                    flops() += 8 + 16;
                #endif
            }
            phi = phi_0 + phi_1 + phi_2 + phi_3 + phi_4 + phi_5 + phi_6 + phi_7;
            #ifdef FLOP_COUNTER
                flops() += 7;
            #endif
            phi_res = 0;
            for(; j < d; j++){
                error = x[pa_d + j] - points[pb_d + j];
                phi_res += error * error;
                #ifdef FLOP_COUNTER
                    flops() += 3;
                #endif
            }
            phi += phi_res;
            #ifdef FLOP_COUNTER
                flops() += 1;
            #endif
            history_phi[pb] = phi;
        }
        
        for(int pb = 0; pb < N_points; pb++){
            phi = history_phi[pb];
            sq_phi = sqrt(phi);              // flops: 1
            phi = phi * sq_phi;                     // flops: 2
            res += phi * lambda_c[pb];               // flops: 2
            #ifdef FLOP_COUNTER
                flops() += 4;
            #endif
        }
        // flops: 2d
        for(int i = 0; i < d; i++){
            res += x[pa_d + i] * lambda_c[N_points + i];
            #ifdef FLOP_COUNTER
                flops() += 2;
            #endif
        }
        // flops: 1
        res += lambda_c[N_points + d];
        #ifdef FLOP_COUNTER
            flops() += 1;
        #endif
        output[pa] = res;
    }
    free(history_phi);
    free(history_phi_vec);
}

void evaluate_surrogate_unroll_8_sqrt_vec( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    // total flops: 3Nd + 5N + 2d + 1
    double res, error, phi_res, phi_sum, sq_phi;
    __m256d phi, phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_half_sum; 
    __m256d error_0, error_1, error_2, error_3, error_4, error_5, error_6, error_7;
    __m256d x_vec_0, x_vec_1, x_vec_2, x_vec_3, x_vec_4, x_vec_5, x_vec_6, x_vec_7;
    __m256d points_vec_0, points_vec_1, points_vec_2, points_vec_3, points_vec_4, points_vec_5, points_vec_6, points_vec_7;
    double* history_phi = (double*)malloc(sizeof(double) * N_points);
    double* phi_half = (double*)aligned_alloc(32, 4 * sizeof(double));
    int id, j, pa_d = 0, pb_d, pa_d_j, pb_d_j;
    for(int pa = 0; pa < N_x; pa++, pa_d += d){
        res = 0;
        // flops: 3Nd + 5N
        pb_d = 0;
        for(int pb = 0; pb < N_points; pb++, pb_d += d){
            phi_0 = phi_1 = phi_2 = phi_3 = phi_4 = phi_5 = phi_6 = phi_7 = _mm256_set1_pd(0);
            j = 0;
            for(; j + 31 < d; j += 32){
                pa_d_j = pa_d + j, pb_d_j = pb_d + j;
                x_vec_0 = _mm256_loadu_pd((double*)(x + pa_d_j));
                x_vec_1 = _mm256_loadu_pd((double*)(x + pa_d_j + 4));
                x_vec_2 = _mm256_loadu_pd((double*)(x + pa_d_j + 8));
                x_vec_3 = _mm256_loadu_pd((double*)(x + pa_d_j + 12));
                x_vec_4 = _mm256_loadu_pd((double*)(x + pa_d_j + 16));
                x_vec_5 = _mm256_loadu_pd((double*)(x + pa_d_j + 20));
                x_vec_6 = _mm256_loadu_pd((double*)(x + pa_d_j + 24));
                x_vec_7 = _mm256_loadu_pd((double*)(x + pa_d_j + 28));
                points_vec_0 = _mm256_loadu_pd((double*)(points + pb_d_j));
                points_vec_1 = _mm256_loadu_pd((double*)(points + pb_d_j + 4));
                points_vec_2 = _mm256_loadu_pd((double*)(points + pb_d_j + 8));
                points_vec_3 = _mm256_loadu_pd((double*)(points + pb_d_j + 12));
                points_vec_4 = _mm256_loadu_pd((double*)(points + pb_d_j + 16));
                points_vec_5 = _mm256_loadu_pd((double*)(points + pb_d_j + 20));
                points_vec_6 = _mm256_loadu_pd((double*)(points + pb_d_j + 24));
                points_vec_7 = _mm256_loadu_pd((double*)(points + pb_d_j + 28));

                error_0 = _mm256_sub_pd(x_vec_0, points_vec_0);
                error_1 = _mm256_sub_pd(x_vec_1, points_vec_1);
                error_2 = _mm256_sub_pd(x_vec_2, points_vec_2);
                error_3 = _mm256_sub_pd(x_vec_3, points_vec_3);
                error_4 = _mm256_sub_pd(x_vec_4, points_vec_4);
                error_5 = _mm256_sub_pd(x_vec_5, points_vec_5);
                error_6 = _mm256_sub_pd(x_vec_6, points_vec_6);
                error_7 = _mm256_sub_pd(x_vec_7, points_vec_7);

                phi_0 = _mm256_fmadd_pd(error_0, error_0, phi_0); 
                phi_1 = _mm256_fmadd_pd(error_1, error_1, phi_1); 
                phi_2 = _mm256_fmadd_pd(error_2, error_2, phi_2); 
                phi_3 = _mm256_fmadd_pd(error_3, error_3, phi_3); 
                phi_4 = _mm256_fmadd_pd(error_4, error_4, phi_4); 
                phi_5 = _mm256_fmadd_pd(error_5, error_5, phi_5); 
                phi_6 = _mm256_fmadd_pd(error_6, error_6, phi_6); 
                phi_7 = _mm256_fmadd_pd(error_7, error_7, phi_7); 
                #ifdef FLOP_COUNTER
                    flops() += 32+64;
                #endif
                
            }
            phi = _mm256_add_pd(
                _mm256_add_pd(
                    _mm256_add_pd(phi_0, phi_1), 
                    _mm256_add_pd(phi_2 , phi_3)
                ),  
                _mm256_add_pd(
                    _mm256_add_pd(phi_4, phi_5), 
                    _mm256_add_pd(phi_6 , phi_7)
                )
            );
            phi_half_sum = _mm256_hadd_pd(phi, phi);
            _mm256_store_pd((double*)phi_half, phi_half_sum);
            phi_sum = phi_half[0] + phi_half[2];

            #ifdef FLOP_COUNTER
                flops() += 7*4+4+1;
            #endif

            phi_res = 0;
            for(; j < d; j++){
                error = x[pa_d + j] - points[pb_d + j];
                phi_res += error * error;
                #ifdef FLOP_COUNTER
                    flops() += 3;
                #endif
            }
            phi_sum += phi_res;
            #ifdef FLOP_COUNTER
                flops() += 1;
            #endif
            history_phi[pb] = phi_sum;
        }
        
        for(int pb = 0; pb < N_points; pb++){
            phi_sum = history_phi[pb];
            sq_phi = sqrt(phi_sum);              // flops: 1
            phi_sum = phi_sum * sq_phi;                     // flops: 2
            res += phi_sum * lambda_c[pb];               // flops: 2
        }
        // flops: 2d
        for(int i = 0; i < d; i++){
            res += x[pa_d + i] * lambda_c[N_points + i];
        }
        // flops: 1
        res += lambda_c[N_points + d];
        #ifdef FLOP_COUNTER
            flops() += N_points*4+d*2+1;
        #endif
        output[pa] = res;
    }
    free(history_phi);
    free(phi_half);
}

void evaluate_surrogate_unroll_8_vec_sqrt_vec( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    // total flops: 3Nd + 5N + 2d + 1
    double res, error, phi_res, phi_sum, sq_phi;
    __m256d phi, phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_half_sum, sq_phi_vec; 
    __m256d error_0, error_1, error_2, error_3, error_4, error_5, error_6, error_7;
    __m256d x_vec_0, x_vec_1, x_vec_2, x_vec_3, x_vec_4, x_vec_5, x_vec_6, x_vec_7;
    __m256d points_vec_0, points_vec_1, points_vec_2, points_vec_3, points_vec_4, points_vec_5, points_vec_6, points_vec_7;
    __m256d lambda_c_vec, res_vec;
    double* history_phi = (double*)aligned_alloc(32, sizeof(double) * N_points);
    double* phi_half = (double*)aligned_alloc(32, 4 * sizeof(double));
    double* res_half = (double*)aligned_alloc(32, 4 * sizeof(double));
    int id, j, pa_d = 0, pb_d, pa_d_j, pb_d_j;
    for(int pa = 0; pa < N_x; pa++, pa_d += d){
        res = 0;
        // flops: 3Nd + 5N
        pb_d = 0;
        for(int pb = 0; pb < N_points; pb++, pb_d += d){
            phi_0 = phi_1 = phi_2 = phi_3 = phi_4 = phi_5 = phi_6 = phi_7 = _mm256_set1_pd(0);
            j = 0;
            for(; j + 31 < d; j += 32){
                pa_d_j = pa_d + j, pb_d_j = pb_d + j;
                x_vec_0 = _mm256_loadu_pd((double*)(x + pa_d_j));
                x_vec_1 = _mm256_loadu_pd((double*)(x + pa_d_j + 4));
                x_vec_2 = _mm256_loadu_pd((double*)(x + pa_d_j + 8));
                x_vec_3 = _mm256_loadu_pd((double*)(x + pa_d_j + 12));
                x_vec_4 = _mm256_loadu_pd((double*)(x + pa_d_j + 16));
                x_vec_5 = _mm256_loadu_pd((double*)(x + pa_d_j + 20));
                x_vec_6 = _mm256_loadu_pd((double*)(x + pa_d_j + 24));
                x_vec_7 = _mm256_loadu_pd((double*)(x + pa_d_j + 28));
                points_vec_0 = _mm256_loadu_pd((double*)(points + pb_d_j));
                points_vec_1 = _mm256_loadu_pd((double*)(points + pb_d_j + 4));
                points_vec_2 = _mm256_loadu_pd((double*)(points + pb_d_j + 8));
                points_vec_3 = _mm256_loadu_pd((double*)(points + pb_d_j + 12));
                points_vec_4 = _mm256_loadu_pd((double*)(points + pb_d_j + 16));
                points_vec_5 = _mm256_loadu_pd((double*)(points + pb_d_j + 20));
                points_vec_6 = _mm256_loadu_pd((double*)(points + pb_d_j + 24));
                points_vec_7 = _mm256_loadu_pd((double*)(points + pb_d_j + 28));

                error_0 = _mm256_sub_pd(x_vec_0, points_vec_0);
                error_1 = _mm256_sub_pd(x_vec_1, points_vec_1);
                error_2 = _mm256_sub_pd(x_vec_2, points_vec_2);
                error_3 = _mm256_sub_pd(x_vec_3, points_vec_3);
                error_4 = _mm256_sub_pd(x_vec_4, points_vec_4);
                error_5 = _mm256_sub_pd(x_vec_5, points_vec_5);
                error_6 = _mm256_sub_pd(x_vec_6, points_vec_6);
                error_7 = _mm256_sub_pd(x_vec_7, points_vec_7);

                phi_0 = _mm256_fmadd_pd(error_0, error_0, phi_0); 
                phi_1 = _mm256_fmadd_pd(error_1, error_1, phi_1); 
                phi_2 = _mm256_fmadd_pd(error_2, error_2, phi_2); 
                phi_3 = _mm256_fmadd_pd(error_3, error_3, phi_3); 
                phi_4 = _mm256_fmadd_pd(error_4, error_4, phi_4); 
                phi_5 = _mm256_fmadd_pd(error_5, error_5, phi_5); 
                phi_6 = _mm256_fmadd_pd(error_6, error_6, phi_6); 
                phi_7 = _mm256_fmadd_pd(error_7, error_7, phi_7); 

                #ifdef FLOP_COUNTER
                    flops()+=24 * 4;
                #endif
            }
            phi = _mm256_add_pd(
                _mm256_add_pd(
                    _mm256_add_pd(phi_0, phi_1), 
                    _mm256_add_pd(phi_2 , phi_3)
                ),  
                _mm256_add_pd(
                    _mm256_add_pd(phi_4, phi_5), 
                    _mm256_add_pd(phi_6 , phi_7)
                )
            );

            #ifdef FLOP_COUNTER
                flops()+=7 * 4;
            #endif

            for(; j + 3 < d; j+=4){
                x_vec_0 = _mm256_loadu_pd((double*)(x + pa_d + j));
                points_vec_0 = _mm256_loadu_pd((double*)(points + pb_d + j));
                error_0 = _mm256_sub_pd(x_vec_0, points_vec_0);
                phi = _mm256_fmadd_pd(error_0, error_0, phi); 

                #ifdef FLOP_COUNTER
                    flops()+=3 * 4;
                #endif
            }
            phi_half_sum = _mm256_hadd_pd(phi, phi); // 4
            _mm256_store_pd((double*)phi_half, phi_half_sum);
            phi_sum = phi_half[0] + phi_half[2]; // 1

            #ifdef FLOP_COUNTER
                flops()+=5;
            #endif
            
            phi_res = 0;
            for(; j < d; j++){
                error = x[pa_d + j] - points[pb_d + j];
                phi_res += error * error;

                #ifdef FLOP_COUNTER
                    flops()+=3;
                #endif
            }
            phi_sum += phi_res;
            history_phi[pb] = phi_sum;

            #ifdef FLOP_COUNTER
                flops()+=1;
            #endif
        }

        res_vec = _mm256_set1_pd(0);
        int pb;
        for(pb = 0; pb + 3 < N_points; pb+=4){
            phi = _mm256_load_pd((double*)(history_phi + pb));
            lambda_c_vec = _mm256_load_pd((double*)(lambda_c + pb));
            sq_phi_vec = _mm256_sqrt_pd(phi);
            phi = _mm256_mul_pd(phi, sq_phi_vec);
            res_vec = _mm256_fmadd_pd(phi, lambda_c_vec, res_vec);

            #ifdef FLOP_COUNTER
                flops()+=4 * 4;
            #endif
            
        }
        res_vec = _mm256_hadd_pd(res_vec, res_vec); //  4
        _mm256_store_pd(res_half,res_vec);
        res += res_half[0]+res_half[2];

        #ifdef FLOP_COUNTER
            flops()+=6;
        #endif

        for(;pb<N_points;pb++){
            phi_sum = history_phi[pb];
            sq_phi = sqrt(phi_sum);              // flops: 1
            phi_sum = phi_sum * sq_phi;                     // flops: 2
            res += phi_sum * lambda_c[pb];               // flops: 2

            #ifdef FLOP_COUNTER
                flops()+=4;
            #endif
        }
        // flops: 2d
        for(int i = 0; i < d; i++){
            res += x[pa_d + i] * lambda_c[N_points + i];

            #ifdef FLOP_COUNTER
                flops()+=2;
            #endif
        }
        // flops: 1
        res += lambda_c[N_points + d];
        output[pa] = res;

        #ifdef FLOP_COUNTER
            flops()+=1;
        #endif
    }
    free(history_phi);
    free(phi_half);
    free(res_half);
}

void generate_random(double* arr, int n){
    for(int i = 0; i < n; i++){
        arr[i] = (double)rand()/ (double)RAND_MAX;
    }
}

void test_eval1(){
    srand(time(NULL));
    myInt64 gt_start,cur_start,gt_time, cur_time,
            cur_time8,
            cur_time8_sqrt,
            cur_time8_sqrt_vec,
            cur_time8_sqrt_sample_vec;
    int N_points = 154, N_x = 154;
    int d = 41;
    int repeat = 10000;
    int warmup_iter = 1000;
    double* x = (double*)malloc(N_x * d * sizeof(double));
    double* points = (double*)malloc(N_points * d * sizeof(double));
    double* lambda_c = (double*)malloc((N_points + d + 1)*sizeof(double));
    generate_random(x, N_x * d);
    generate_random(points, N_points * d);
    generate_random(lambda_c, N_points + d + 1);

    double* result = (double*)malloc(N_x * sizeof(double));
    double* result8 = (double*)malloc(N_x * sizeof(double));
    double* result8_sqrt = (double*)malloc(N_x * sizeof(double));
    double* result8_sqrt_vec = (double*)malloc(N_x * sizeof(double));
    double* groundtruth = (double*)malloc(N_x * sizeof(double));

    //------------------------
    // test for groundtruth
    //------------------------

    for(int i = 0; i < warmup_iter; i++){
        evaluate_surrogate_gt( x, points,  lambda_c, N_x, N_points, d, groundtruth);
    }
    gt_start = start_tsc();
    for(int i = 0; i < repeat; i++){
        evaluate_surrogate_gt( x, points,  lambda_c, N_x, N_points, d, groundtruth);
    }
    gt_time = stop_tsc(gt_start);
    cout << "Check correctness: --------"<<endl;
    cout << "groundtruth: "<<groundtruth[0]<<endl; 

    //----------------------
    // test for unrolling 8
    //----------------------

    for(int i = 0; i < warmup_iter; i++){
        evaluate_surrogate_unroll_8( x, points,  lambda_c, N_x, N_points, d, result8);
    }
    cur_start = start_tsc();
    for(int i = 0; i < repeat; i++){
        evaluate_surrogate_unroll_8( x, points,  lambda_c, N_x, N_points, d, result8);
    }
    cur_time8 = stop_tsc(cur_start);
    cout << "result of unrolling-8: "<< result8[0] << endl;

    //----------------------
    // test for unrolling 8, sqrt
    //----------------------

    for(int i = 0; i < warmup_iter; i++){
        evaluate_surrogate_unroll_8_sqrt( x, points,  lambda_c, N_x, N_points, d, result8_sqrt);
    }
    cur_start = start_tsc();
    for(int i = 0; i < repeat; i++){
        evaluate_surrogate_unroll_8_sqrt( x, points,  lambda_c, N_x, N_points, d, result8_sqrt);
    }
    cur_time8_sqrt = stop_tsc(cur_start);
    cout << "result of unrolling-8-sqrt: "<< result8_sqrt[0] << endl;

     //----------------------
    // test for unrolling 8, sqrt unroll 4
    //----------------------

    for(int i = 0; i < warmup_iter; i++){
        evaluate_surrogate_unroll_8_sqrt_vec( x, points,  lambda_c, N_x, N_points, d, result8_sqrt_vec);
    }
    cur_start = start_tsc();
    for(int i = 0; i < repeat; i++){
        evaluate_surrogate_unroll_8_sqrt_vec( x, points,  lambda_c, N_x, N_points, d, result8_sqrt_vec);
    }
    cur_time8_sqrt_vec = stop_tsc(cur_start);
    cout << "result of unrolling-8-sqrt-vec: "<< result8_sqrt_vec[0] << endl;


    // print out performance
    cout << "Compare running time: -----------" << endl;
    cout << "groundtruth cycles: "<< gt_time/(double)repeat << endl;
    cout << "current cycles of unrolling-8: "<< cur_time8/(double)repeat << " and performance improve is: " <<   ((gt_time/(double)repeat)-(cur_time8/(double)repeat)) / (gt_time/(double)repeat) << endl;
    cout << "current cycles of unrolling-8-sqrt: "<< cur_time8_sqrt/(double)repeat << " and performance improve is: " <<   ((gt_time/(double)repeat)-(cur_time8_sqrt/(double)repeat)) / (gt_time/(double)repeat) << endl;
    cout << "current cycles of unrolling-8-sqrt-vec: "<< cur_time8_sqrt_vec/(double)repeat << " and performance improve is: " <<   ((gt_time/(double)repeat)-(cur_time8_sqrt_vec/(double)repeat)) / (gt_time/(double)repeat) << endl;

    free(x); free(points); free(lambda_c);
    free(result); free(result8); free(result8_sqrt); free(result8_sqrt_vec);
    free(groundtruth);
}

void test_gt(){
    srand(time(NULL));
    myInt64 gt_start, gt_time, vec_start, vec_time, start8,time8, vec_vec_time, sample_vec_time, sample_vec_optimize_load_time;

    int N_points = 1000, N_x = 154;
    int d = 4;
    int repeat = 1000;
    int warmup_iter = 100;
    double* x = (double*)malloc(N_x * d * sizeof(double));
    double* points = (double*)malloc(N_points * d * sizeof(double));
    double* lambda_c = (double*)aligned_alloc(32,(N_points + d + 1)*sizeof(double));
    generate_random(x, N_x * d);
    generate_random(points, N_points * d);
    generate_random(lambda_c, N_points + d + 1);

    // cout << x[32];
    // exit(0);
    double* groundtruth = (double*)malloc(N_x * sizeof(double));
    double* result8_sqrt_vec = (double*)malloc(N_x * sizeof(double));
    evaluate_surrogate_gt( x, points,  lambda_c, N_x, N_points, d, groundtruth);
    // cout << "groundtruth: ";
    // for(int i = 0; i < N_x; i++){
    //     cout << groundtruth[i] << " ";
    // }
    // cout << endl;
    evaluate_surrogate_unroll_8_sqrt_sample_vec_optimize_load( x, points,  lambda_c, N_x, N_points, d, result8_sqrt_vec);
    // cout << "result of unroll_8_vec_sqrt_vec: ";
    // for(int i = 0; i < N_x; i++){
    //     cout << result8_sqrt_vec[i] << " ";
    // }
    // cout << endl;
    for(int i = 0; i < N_x; i++){
        if(abs(result8_sqrt_vec[i] - groundtruth[i]) > 1e-5){
            cout << "wrong at " << i << endl;
            cout << result8_sqrt_vec[i] << " " << groundtruth[i] << endl;
            exit(0);
        }
    }
    cout << "correct\n";
    //------------------------
    // test for vec
    //------------------------

    // for(int i = 0; i < warmup_iter; i++){
    //     evaluate_surrogate_unroll_8_sqrt_vec( x, points,  lambda_c, N_x, N_points, d, groundtruth);
    // }
    // vec_start = start_tsc();
    // for(int i = 0; i < repeat; i++){
    //     evaluate_surrogate_unroll_8_sqrt_vec( x, points,  lambda_c, N_x, N_points, d, groundtruth);
    // }
    // vec_time = stop_tsc(vec_start);


    // for(int i = 0; i < warmup_iter; i++){
    //     evaluate_surrogate_unroll_8_vec_sqrt_vec( x, points,  lambda_c, N_x, N_points, d, groundtruth);
    // }
    // vec_start = start_tsc();
    // for(int i = 0; i < repeat; i++){
    //     evaluate_surrogate_unroll_8_vec_sqrt_vec( x, points,  lambda_c, N_x, N_points, d, groundtruth);
    // }
    // vec_vec_time = stop_tsc(vec_start);

    for(int i = 0; i < warmup_iter; i++){
        evaluate_surrogate_unroll_8_sqrt_sample_vec( x, points,  lambda_c, N_x, N_points, d, groundtruth);
    }
    vec_start = start_tsc();
    for(int i = 0; i < repeat; i++){
        evaluate_surrogate_unroll_8_sqrt_sample_vec( x, points,  lambda_c, N_x, N_points, d, groundtruth);
    }
    sample_vec_time = stop_tsc(vec_start);

    for(int i = 0; i < warmup_iter; i++){
        evaluate_surrogate_unroll_8_sqrt_sample_vec_optimize_load( x, points,  lambda_c, N_x, N_points, d, groundtruth);
    }
    vec_start = start_tsc();
    for(int i = 0; i < repeat; i++){
        evaluate_surrogate_unroll_8_sqrt_sample_vec_optimize_load( x, points,  lambda_c, N_x, N_points, d, groundtruth);
    }
    sample_vec_optimize_load_time = stop_tsc(vec_start);

    //------------------------
    // test for ground truth
    //------------------------

    for(int i = 0; i < warmup_iter; i++){
        evaluate_surrogate_gt( x, points,  lambda_c, N_x, N_points, d, groundtruth);
    }
    gt_start = start_tsc();
    for(int i = 0; i < repeat; i++){
        evaluate_surrogate_gt( x, points,  lambda_c, N_x, N_points, d, groundtruth);
    }
    gt_time = stop_tsc(gt_start);




    cout << "Compare running time: -----------" << endl;
    cout << "groundtruth cycles: "<< gt_time/(double)repeat << endl;
    // cout << "current cycles of unrolling-8-sqrt: "<< time8/(double)repeat << " and performance improve is: " <<   ((gt_time/(double)repeat)) / (time8/(double)repeat) << endl;
    // cout << "current cycles of unrolling-8-sqrt-vec: "<< vec_time/(double)repeat << " and performance improve is: " <<   ((gt_time/(double)repeat) / (vec_time/(double)repeat)) << endl;
    // cout << "current cycles of unrolling-8-vec-sqrt-vec: "<< vec_vec_time/(double)repeat << " and performance improve is: " <<   ((gt_time/(double)repeat)/(vec_vec_time/(double)repeat))  << endl;
    cout << "current cycles of unrolling-8-vec-sqrt-sample-vec: "<< sample_vec_time/(double)repeat << " and performance improve is: " <<   ((gt_time/(double)repeat)/(sample_vec_time/(double)repeat))  << endl;
    cout << "current cycles of unrolling-8-vec-sqrt-sample-vec-optimize-load: "<< sample_vec_optimize_load_time/(double)repeat << " and performance improve is: " <<   ((gt_time/(double)repeat)/(sample_vec_optimize_load_time/(double)repeat))  << endl;
    free(x);free(points);free(lambda_c);
    free(groundtruth); free(result8_sqrt_vec);
}


double test_function(void (*eval)(double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output), 
                   int N_points, int N_x, int d, int repeat, int warmup){
    srand(time(NULL));
    myInt64 start,time;
    double* x = (double*)malloc(N_x * d * sizeof(double));
    double* points = (double*)malloc(N_points * d * sizeof(double));
    double* lambda_c = (double*)aligned_alloc(32,(N_points + d + 1)*sizeof(double));
    double* result = (double*)malloc(N_x * sizeof(double));

    generate_random(x, N_x * d);
    generate_random(points, N_points * d);
    generate_random(lambda_c, N_points + d + 1);

    eval( x, points,  lambda_c, N_x, N_points, d, result);

    for(int i = 0; i < warmup; i++){
        eval( x, points,  lambda_c, N_x, N_points, d, result);
    }

    start = start_tsc();
    for(int i = 0; i < repeat; i++){
        eval( x, points,  lambda_c, N_x, N_points, d, result);
    }
    time = stop_tsc(start);

    free(x);
    free(points);
    free(lambda_c);
    free(result);

    return time / (double)repeat;
}


void compare_one(void (*eval)(double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output),
                int N_points, int N_x, int d, double repeat, int warmup, double*x, double*points, double*lambda_c, double* result,
                vector<myInt64>& cycles_vec, vector<myInt64>& flops_vec){
    myInt64 start;
    for(int i = 0; i < warmup; i++){
        eval(x,points,lambda_c,N_x,N_points,d,result);
    }
    #ifdef FLOP_COUNTER
    flops() = 0;
    #else
    start = start_tsc();
    #endif
    for(int i = 0; i < repeat; i++){
        eval(x,points,lambda_c,N_x,N_points,d,result);
    }
    #ifdef FLOP_COUNTER
    flops_vec.push_back(flops()/repeat);
    #else
    cycles_vec.push_back(stop_tsc(start)/repeat);
    #endif              
}

void compare_all(int N_points, int N_x, int d, double repeat, int warmup, unsigned int random_seed,
                 vector<myInt64>& cycles_vec, vector<myInt64>& flops_vec){
    srand(random_seed);
    double* x = (double*)malloc(N_x * d * sizeof(double));
    double* points = (double*)malloc(N_points * d * sizeof(double));
    double* lambda_c = (double*)aligned_alloc(32,(N_points + d + 1)*sizeof(double));
    double* result = (double*)malloc(N_x * sizeof(double));

    generate_random(x, N_x * d);
    generate_random(points, N_points * d);
    generate_random(lambda_c, N_points + d + 1);

    cycles_vec.clear();
    flops_vec.clear();
    
    compare_one(evaluate_surrogate_gt,N_points,N_x,d,repeat,warmup,x,points,lambda_c,result,cycles_vec,flops_vec);
    compare_one(evaluate_surrogate_rename,N_points,N_x,d,repeat,warmup,x,points,lambda_c,result,cycles_vec,flops_vec);
    compare_one(evaluate_surrogate_reorder,N_points,N_x,d,repeat,warmup,x,points,lambda_c,result,cycles_vec,flops_vec);
    compare_one(evaluate_surrogate_reorder_rename,N_points,N_x,d,repeat,warmup,x,points,lambda_c,result,cycles_vec,flops_vec);
    compare_one(evaluate_surrogate_unroll_8,N_points,N_x,d,repeat,warmup,x,points,lambda_c,result,cycles_vec,flops_vec);
    compare_one(evaluate_surrogate_unroll_8_sqrt,N_points,N_x,d,repeat,warmup,x,points,lambda_c,result,cycles_vec,flops_vec);
    compare_one(evaluate_surrogate_unroll_8_sqrt_vec,N_points,N_x,d,repeat,warmup,x,points,lambda_c,result,cycles_vec,flops_vec);
    compare_one(evaluate_surrogate_unroll_8_vec_sqrt_vec,N_points,N_x,d,repeat,warmup,x,points,lambda_c,result,cycles_vec,flops_vec);
    compare_one(evaluate_surrogate_unroll_8_sqrt_sample_vec,N_points,N_x,d,repeat,warmup,x,points,lambda_c,result,cycles_vec,flops_vec);
    compare_one(evaluate_surrogate_unroll_8_sqrt_sample_vec_optimize_load,N_points,N_x,d,repeat,warmup,x,points,lambda_c,result,cycles_vec,flops_vec);

    free(x);
    free(points);
    free(lambda_c);
    free(result);
}


int main(){
    // test_gt();
    // double feq = 2.4*1000000000;

    // vector<int> N;
    // vector<double> time;

    // cout << "n: " << endl;
    // for(int n = 100; n < 200; n+=200){
    //     double time_s = test_function(evaluate_surrogate_unroll_8_sqrt_sample_vec_optimize_load,154,n,43,1000,100)/feq;
    //     N.push_back(n);
    //     time.push_back(time_s);
    //     cout << n << ", ";
    // }

    // cout << "\n time: \n";
    // for(double t: time){
    //     cout << t <<", ";
    // }
    // cout << endl;
    int N_points_start = 1000;
    int N_points_end   = 500000;

    int N_points_gap = 10000;
    int N_x = 20;
    int d = 4;

    int N_points_num = 0;
    for(int i = N_points_start; i < N_points_end; i+=N_points_gap){
        N_points_num ++;
    }

    double repeat = 100;
    int warmup = 10;
    unsigned int random_seed = 2;
    vector<vector<myInt64>> cycles_vec(N_points_num,vector<myInt64>()); 
    vector<vector<myInt64>> flops_vec(N_points_num,vector<myInt64>());
    vector<string> name_vec;
    name_vec.push_back("evaluate_surrogate_gt");
    name_vec.push_back("evaluate_surrogate_rename");
    name_vec.push_back("evaluate_surrogate_reorder");
    name_vec.push_back("evaluate_surrogate_reorder_rename");
    name_vec.push_back("evaluate_surrogate_unroll_8");
    name_vec.push_back("evaluate_surrogate_unroll_8_sqrt");
    name_vec.push_back("evaluate_surrogate_unroll_8_sqrt_vec");
    name_vec.push_back("evaluate_surrogate_unroll_8_vec_sqrt_vec");
    name_vec.push_back("evaluate_surrogate_unroll_8_sqrt_sample_vec");
    name_vec.push_back("evaluate_surrogate_unroll_8_sqrt_sample_vec_optimize_load");
    


    cout << "N_points_start: " << N_points_start << endl;
    cout << "N_points_end: " << N_points_end << endl;
    cout << "N_points_gap: " << N_points_gap << endl;
    cout << "N_x: " << N_x << endl;
    cout << "d: " << d << endl;
    for(int N_points = N_points_start, i=0; N_points < N_points_end; N_points+=N_points_gap, i++){
        compare_all(N_points,N_x,d,repeat,warmup,random_seed,cycles_vec[i],flops_vec[i]);
    }
    cout << "evaluate_func_name:[\"";
    for(int i = 0; i < name_vec.size()-1; i++){
        cout << name_vec[i] << "\", ";
    }
    cout << name_vec[name_vec.size()-1] << "\"]\n";
    #ifdef FLOP_COUNTER
        cout << "evaluate_func_flops:[";

        for(int j = 0; j < N_points_num; j++){
            cout << "[";
            for(int i = 0; i < flops_vec[j].size()-1; i++){
                cout << flops_vec[j][i] << ", ";
            }
            cout << flops_vec[j][flops_vec[j].size()-1] << "]";
            if(j<N_points_num-1) cout << ",";
        }

        cout << "]\n";
    #else
        cout << "evaluate_func_cycles:[";
        for(int j= 0; j < N_points_num; j++){
            cout << "[";
            for(int i = 0; i < cycles_vec[j].size()-1; i++){
                cout << cycles_vec[j][i] << ", ";
            }
            cout << cycles_vec[j][cycles_vec[j].size()-1] << "]";
            if(j<N_points_num-1) cout << ",";
        }
        cout << "]\n";
    #endif
}