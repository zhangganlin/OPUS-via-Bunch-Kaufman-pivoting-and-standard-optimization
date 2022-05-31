#include "surrogate.hpp"
#include "tsc_x86.h"

using namespace std;

void build_surrogate(double* points, double* f, int N, int d, double* lambda_c){

    double* A = (double*)malloc((N + d + 1)*(N + d + 1)*sizeof(double));
    double* b = (double*)malloc((N + d + 1)*sizeof(double));

    double phi, error;
    memset(A, 0, (N + d + 1) * (N + d + 1) * sizeof(double));
    memset(b, 0, (N + d + 1) * sizeof(double));
    for(int i = 0; i < N; i++){
        for(int j = 0; j < d; j++){
            A[i * (N + d + 1) + N + j] = points[i * d + j];
            A[(N + j) * (N + d + 1) + i] = points[i * d + j];
        }
        A[i * (N + d + 1) + N + d] = 1;
        A[(N + d) * (N + d + 1) + i] = 1;
    }
    double sq_phi;
    // flops: N * N * (3d + 3)
    // for(int pa = 0; pa < N; pa++){
    //     for(int pb = 0; pb < N; pb++){
    //         phi = 0;
    //         // flops: 3d
    //         for(int j = 0; j < d; j++){
    //             error = points[pa * d + j] - points[pb * d + j];
    //             phi += error * error;
    //         }
    //         sq_phi = sqrt(phi); //1
    //         phi = sq_phi * phi; //2
    //         A[pa * (N + d + 1) + pb] = phi;
    //     }
    // }
    
    
    // optimized
    for(int pa = 0; pa < N; pa++){
        // A already set to zero
        A[pa * (N + d + 1) + pa] = 0;
        for(int pb = pa + 1; pb < N; pb++){
            phi = 0;
            for(int j = 0; j < d; j++){
                error = points[pa * d + j] - points[pb * d + j];
                phi += error * error;
            }
            phi = sqrt(phi);
            phi = phi * phi * phi;
            A[pa * (N + d + 1) + pb] = phi;
            A[pb * (N + d + 1) + pa] = phi;
        }
    }

    
    // for(int i = 0; i < N; i++) b[i] = f[i];
    memcpy((void *)b, (void *)f, sizeof(double) * N);
    
    solve_BunchKaufman(A,lambda_c,b,N+d+1);

    free(A);
    free(b);
}

void evaluate_surrogate_batch( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    for(int x_idx = 0; x_idx < N_x; x_idx++){
        double phi, error, res = 0, sq_phi;
        int id;
        // flops: 3Nd + 5N
        int xidxd = x_idx * d;
        for(int i = 0; i < N_points; i++){
            phi = 0;
            // flops: 3d

            id = i * d;
            for(int j = 0; j < d; j++){
                error = x[xidxd + j] - points[id + j];
                phi += error * error;
            }
            double sq_phi = sqrt(phi);            // flops: 1
            phi = phi * sq_phi;      // flops: 2
            res += phi * lambda_c[i];   // flops: 2
        }
        // flops: 2d
        for(int i = 0; i < d; i++){
            res += x[xidxd + i] * lambda_c[N_points + i];
        }
        // flops: 1
        res += lambda_c[N_points + d];

        output[x_idx] = res;
    }
}

double evaluate_surrogate( double* x, double* points,  double* lambda_c, int N, int d){
    // total flops: 3Nd + 5N + 2d + 1
    double phi, error, res = 0, sq_phi;
    int id;
    // flops: 3Nd + 5N
    for(int i = 0; i < N; i++){
        phi = 0;
        // flops: 3d

        int id = i * d;

        for(int j = 0; j < d; j++){
            error = x[j] - points[id + j];
            phi += error * error;
        }
        double sq_phi = sqrt(phi);            // flops: 1
        phi = phi * sq_phi;      // flops: 2
        res += phi * lambda_c[i];   // flops: 2
    }
    // flops: 2d
    for(int i = 0; i < d; i++){
        res += x[i] * lambda_c[N + i];
    }
    // flops: 1
    res += lambda_c[N + d];
    return res;
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
            }
            phi_0_vec = _mm256_add_pd(phi_00_vec, phi_04_vec);
            phi_1_vec = _mm256_add_pd(phi_10_vec, phi_14_vec);
            phi_2_vec = _mm256_add_pd(phi_20_vec, phi_24_vec);
            phi_3_vec = _mm256_add_pd(phi_30_vec, phi_34_vec); 

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
            }
            phi_01_vec = _mm256_permute4x64_pd(_mm256_hadd_pd(phi_0_vec, phi_2_vec), 0b11011000);
            phi_23_vec = _mm256_permute4x64_pd(_mm256_hadd_pd(phi_1_vec, phi_3_vec), 0b11011000);

            phi_vec = _mm256_hadd_pd(phi_01_vec, phi_23_vec);

            for(; j < d; j++){
                x_vec_00 = _mm256_i64gather_pd((double*)(x + pa_d + j), jump_idx, sizeof(double));
                points_vec_0 = _mm256_broadcast_sd(points + pb_d + j);
                error_00_vec = _mm256_sub_pd(x_vec_00, points_vec_0);
                phi_vec = _mm256_fmadd_pd(error_00_vec, error_00_vec, phi_vec); 
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
        }
        res_vec_01 = _mm256_permute4x64_pd(_mm256_hadd_pd(res_vec_0, res_vec_2), 0b11011000);
        res_vec_23 = _mm256_permute4x64_pd(_mm256_hadd_pd(res_vec_1, res_vec_3), 0b11011000);

        res_vec = _mm256_add_pd(_mm256_hadd_pd(res_vec_01, res_vec_23), res_vec);
        for(; i < d; i++){
            x_vec_00 = _mm256_i64gather_pd((double*)(x + pa_d + i), jump_idx, sizeof(double));
            lambda_c_vec = _mm256_broadcast_sd((double*)(lambda_c + N_points + i));
            res_vec =  _mm256_fmadd_pd(x_vec_00, lambda_c_vec, res_vec);
        }

        // flops: 1
        lambda_c_vec = _mm256_broadcast_sd((double*)(lambda_c + N_points + d));
        res_vec = _mm256_add_pd(res_vec, lambda_c_vec);
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
            }
            phi = phi_0 + phi_1 + phi_2 + phi_3 + phi_4 + phi_5 + phi_6 + phi_7;
            phi_res = 0;
            for(; j < d; j++){
                error = x[pa_d + j] - points[pb_d + j];
                phi_res += error * error;
            }
            phi += phi_res;
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
        output[pa] = res;
    }
    free(history_phi);
    free(history_phi_vec);
}