#include "surrogate.hpp"
#include <iostream>
#include "tsc_x86.h"
#include "randomlhs.hpp"
#include <time.h>

using namespace std;

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
        }
        // flops: 2d
        for(int pb = 0; pb < d; pb++){
            res += x[pa * d + pb] * lambda_c[N_points + pb];
        }
        // flops: 1
        res += lambda_c[N_points + d];

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
            phi = sq_phi * phi;      // flops: 2
            res += phi * lambda_c[pb];   // flops: 2
        }
        // flops: 2d
        for(int pb = 0; pb < d; pb++){
            res += x[pa_d + pb] * lambda_c[N_points + pb];
        }
        // flops: 1
        res += lambda_c[N_points + d];

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

        output[pa] = res;
    }
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

        output[pa] = res;
    }
}

void evaluate_surrogate_unroll_3( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    double res, sq_phi;
    double phi, phi_0, phi_1, phi_2, phi_res; 
    double error_0, error_1, error_2, error_3, error;
    int id, j, pa_d = 0, pb_d, pa_d_j, pb_d_j;
    for(int pa = 0; pa < N_x; pa++, pa_d += d){
        res = 0;
        // flops: 3Nd + 5N
        pb_d = 0;
        for(int pb = 0; pb < N_points; pb++, pb_d += d){
            phi_0 = phi_1 = phi_2  = 0; 
            j = 0;
            for(; j + 2 < d; j += 3){
                pa_d_j = pa_d + j, pb_d_j = pb_d + j;
                error_0 = x[pa_d_j] - points[pb_d + j];
                error_1 = x[pa_d_j + 1] - points[pb_d_j + 1];
                error_2 = x[pa_d_j + 2] - points[pb_d_j + 2];
                phi_0 += error_0 * error_0; 
                phi_1 += error_1 * error_1; 
                phi_2 += error_2 * error_2; 
            }
            phi = phi_0 + phi_1 + phi_2;
            phi_res = 0;
            for(; j < d; j++){
                error = x[pa_d + j] - points[pb_d + j];
                phi_res += error * error;
            }
            phi += phi_res;

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
}

void evaluate_surrogate_unroll_4( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    double res, sq_phi;
    double phi, phi_0, phi_1, phi_2, phi_3, phi_res; 
    double error_0, error_1, error_2, error_3, error;
    int id, j, pa_d = 0, pb_d, pa_d_j, pb_d_j;
    for(int pa = 0; pa < N_x; pa++, pa_d += d){
        res = 0;
        // flops: 3Nd + 5N
        pb_d = 0;
        for(int pb = 0; pb < N_points; pb++, pb_d += d){
            phi_0 = phi_1 = phi_2 = phi_3  = 0; 
            j = 0;
            for(; j + 3 < d; j += 4){
                pa_d_j = pa_d + j, pb_d_j = pb_d + j;
                error_0 = x[pa_d_j] - points[pb_d + j];
                error_1 = x[pa_d_j + 1] - points[pb_d_j + 1];
                error_2 = x[pa_d_j + 2] - points[pb_d_j + 2];
                error_3 = x[pa_d_j + 3] - points[pb_d_j + 3];
                phi_0 += error_0 * error_0; 
                phi_1 += error_1 * error_1; 
                phi_2 += error_2 * error_2; 
                phi_3 += error_3 * error_3; 

            }
            phi = phi_0 + phi_1 + phi_2 + phi_3;
            phi_res = 0;
            for(; j < d; j++){
                error = x[pa_d + j] - points[pb_d + j];
                phi_res += error * error;
            }
            phi += phi_res;

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
}

void evaluate_surrogate_unroll_5( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    double res, sq_phi;
    double phi, phi_0, phi_1, phi_2, phi_3, phi_4, phi_res; 
    double error_0, error_1, error_2, error_3, error_4, error;
    int id, j, pa_d = 0, pb_d, pa_d_j, pb_d_j;
    for(int pa = 0; pa < N_x; pa++, pa_d += d){
        res = 0;
        // flops: 3Nd + 5N
        pb_d = 0;
        for(int pb = 0; pb < N_points; pb++, pb_d += d){
            phi_0 = phi_1 = phi_2 = phi_3 = phi_4 = 0; 
            j = 0;
            for(; j + 4 < d; j += 5){
                pa_d_j = pa_d + j, pb_d_j = pb_d + j;
                error_0 = x[pa_d_j] - points[pb_d + j];
                error_1 = x[pa_d_j + 1] - points[pb_d_j + 1];
                error_2 = x[pa_d_j + 2] - points[pb_d_j + 2];
                error_3 = x[pa_d_j + 3] - points[pb_d_j + 3];
                error_4 = x[pa_d_j + 4] - points[pb_d_j + 4];
                phi_0 += error_0 * error_0; 
                phi_1 += error_1 * error_1; 
                phi_2 += error_2 * error_2; 
                phi_3 += error_3 * error_3; 
                phi_4 += error_4 * error_4; 
            }
            phi = phi_0 + phi_1 + phi_2 + phi_3 + phi_4;
            phi_res = 0;
            for(; j < d; j++){
                error = x[pa_d + j] - points[pb_d + j];
                phi_res += error * error;
            }
            phi += phi_res;

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
}

void evaluate_surrogate_unroll_6( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    double res, sq_phi;
    double phi, phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_res; 
    double error_0, error_1, error_2, error_3, error_4, error_5, error;
    int id, j, pa_d = 0, pb_d, pa_d_j, pb_d_j;
    for(int pa = 0; pa < N_x; pa++, pa_d += d){
        res = 0;
        // flops: 3Nd + 5N
        pb_d = 0;
        for(int pb = 0; pb < N_points; pb++, pb_d += d){
            phi_0 = phi_1 = phi_2 = phi_3 = phi_4 = phi_5 = 0; 
            j = 0;
            for(; j + 5< d; j += 6){
                pa_d_j = pa_d + j, pb_d_j = pb_d + j;
                error_0 = x[pa_d_j] - points[pb_d + j];
                error_1 = x[pa_d_j + 1] - points[pb_d_j + 1];
                error_2 = x[pa_d_j + 2] - points[pb_d_j + 2];
                error_3 = x[pa_d_j + 3] - points[pb_d_j + 3];
                error_4 = x[pa_d_j + 4] - points[pb_d_j + 4];
                error_5 = x[pa_d_j + 5] - points[pb_d_j + 5];
                phi_0 += error_0 * error_0; 
                phi_1 += error_1 * error_1; 
                phi_2 += error_2 * error_2; 
                phi_3 += error_3 * error_3; 
                phi_4 += error_4 * error_4; 
                phi_5 += error_5 * error_5; 
            }
            phi = phi_0 + phi_1 + phi_2 + phi_3 + phi_4 + phi_5;
            phi_res = 0;
            for(; j < d; j++){
                error = x[pa_d + j] - points[pb_d + j];
                phi_res += error * error;
            }
            phi += phi_res;

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
}

void evaluate_surrogate_unroll_2( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    double res, sq_phi;
    double phi, phi_0, phi_1, phi_res; 
    double error_0, error_1, error_2, error;
    int id, j, pa_d = 0, pb_d, pa_d_j, pb_d_j;
    for(int pa = 0; pa < N_x; pa++, pa_d += d){
        res = 0;
        // flops: 3Nd + 5N
        pb_d = 0;
        for(int pb = 0; pb < N_points; pb++, pb_d += d){
            phi_0 = phi_1 = 0; 
            j = 0;
            for(; j + 1 < d; j += 2){
                pa_d_j = pa_d + j, pb_d_j = pb_d + j;
                error_0 = x[pa_d_j] - points[pb_d + j];
                error_1 = x[pa_d_j + 1] - points[pb_d_j + 1];
                phi_0 += error_0 * error_0; 
                phi_1 += error_1 * error_1; 

            }
            phi = phi_0 + phi_1;
            phi_res = 0;
            for(; j < d; j++){
                error = x[pa_d + j] - points[pb_d + j];
                phi_res += error * error;
            }
            phi += phi_res;

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
}

void evaluate_surrogate_unroll_7( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    double res, sq_phi;
    double phi, phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_res; 
    double error_0, error_1, error_2, error_3, error_4, error_5, error_6, error;
    int id, j, pa_d = 0, pb_d, pa_d_j, pb_d_j;
    for(int pa = 0; pa < N_x; pa++, pa_d += d){
        res = 0;
        // flops: 3Nd + 5N
        pb_d = 0;
        for(int pb = 0; pb < N_points; pb++, pb_d += d){
            phi_0 = phi_1 = phi_2 = phi_3 = phi_4 = phi_5 = phi_6 = 0; 
            j = 0;
            for(; j + 6 < d; j += 7){
                pa_d_j = pa_d + j, pb_d_j = pb_d + j;
                error_0 = x[pa_d_j] - points[pb_d + j];
                error_1 = x[pa_d_j + 1] - points[pb_d_j + 1];
                error_2 = x[pa_d_j + 2] - points[pb_d_j + 2];
                error_3 = x[pa_d_j + 3] - points[pb_d_j + 3];
                error_4 = x[pa_d_j + 4] - points[pb_d_j + 4];
                error_5 = x[pa_d_j + 5] - points[pb_d_j + 5];
                error_6 = x[pa_d_j + 6] - points[pb_d_j + 6];
                phi_0 += error_0 * error_0; 
                phi_1 += error_1 * error_1; 
                phi_2 += error_2 * error_2; 
                phi_3 += error_3 * error_3; 
                phi_4 += error_4 * error_4; 
                phi_5 += error_5 * error_5; 
                phi_6 += error_6 * error_6; 
            }
            phi = phi_0 + phi_1 + phi_2 + phi_3 + phi_4 + phi_5 + phi_6;
            phi_res = 0;
            for(; j < d; j++){
                error = x[pa_d + j] - points[pb_d + j];
                phi_res += error * error;
            }
            phi += phi_res;

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
            }
            phi = phi_0 + phi_1 + phi_2 + phi_3 + phi_4 + phi_5 + phi_6 + phi_7;
            phi_res = 0;
            for(; j < d; j++){
                error = x[pa_d + j] - points[pb_d + j];
                phi_res += error * error;
            }
            phi += phi_res;

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
}

void evaluate_surrogate_unroll_9( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    double res, sq_phi;
    double phi, phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_8, phi_res; 
    double error_0, error_1, error_2, error_3, error_4, error_5, error_6, error_7, error_8, error;
    int id, j, pa_d = 0, pb_d, pa_d_j, pb_d_j;
    for(int pa = 0; pa < N_x; pa++, pa_d += d){
        res = 0;
        // flops: 3Nd + 5N
        pb_d = 0;
        for(int pb = 0; pb < N_points; pb++, pb_d += d){
            phi_0 = phi_1 = phi_2 = phi_3 = phi_4 = phi_5 = phi_6 = phi_7 = phi_8 = 0; 
            j = 0;
            for(; j + 8 < d; j += 9){
                pa_d_j = pa_d + j, pb_d_j = pb_d + j;
                error_0 = x[pa_d_j] - points[pb_d + j];
                error_1 = x[pa_d_j + 1] - points[pb_d_j + 1];
                error_2 = x[pa_d_j + 2] - points[pb_d_j + 2];
                error_3 = x[pa_d_j + 3] - points[pb_d_j + 3];
                error_4 = x[pa_d_j + 4] - points[pb_d_j + 4];
                error_5 = x[pa_d_j + 5] - points[pb_d_j + 5];
                error_6 = x[pa_d_j + 6] - points[pb_d_j + 6];
                error_7 = x[pa_d_j + 7] - points[pb_d_j + 7];
                error_8 = x[pa_d_j + 8] - points[pb_d_j + 8];
                phi_0 += error_0 * error_0; 
                phi_1 += error_1 * error_1; 
                phi_2 += error_2 * error_2; 
                phi_3 += error_3 * error_3; 
                phi_4 += error_4 * error_4; 
                phi_5 += error_5 * error_5; 
                phi_6 += error_6 * error_6; 
                phi_7 += error_7 * error_7; 
                phi_8 += error_8 * error_8; 
            }
            phi = phi_0 + phi_1 + phi_2 + phi_3 + phi_4 + phi_5 + phi_6 + phi_7 + phi_8;
            phi_res = 0;
            for(; j < d; j++){
                error = x[pa_d + j] - points[pb_d + j];
                phi_res += error * error;
            }
            phi += phi_res;

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
}

void evaluate_surrogate_unroll_10( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    double res, sq_phi;
    double phi, phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_8, phi_9, phi_res; 
    double error_0, error_1, error_2, error_3, error_4, error_5, error_6, error_7, error_8, error_9, error;
    int id, j, pa_d = 0, pb_d, pa_d_j, pb_d_j;
    for(int pa = 0; pa < N_x; pa++, pa_d += d){
        res = 0;
        // flops: 3Nd + 5N
        pb_d = 0;
        for(int pb = 0; pb < N_points; pb++, pb_d += d){
            phi_0 = phi_1 = phi_2 = phi_3 = phi_4 = phi_5 = phi_6 = phi_7 = phi_8 = phi_9 = 0; 
            j = 0;
            for(; j + 9 < d; j += 10){
                pa_d_j = pa_d + j, pb_d_j = pb_d + j;
                error_0 = x[pa_d_j] - points[pb_d + j];
                error_1 = x[pa_d_j + 1] - points[pb_d_j + 1];
                error_2 = x[pa_d_j + 2] - points[pb_d_j + 2];
                error_3 = x[pa_d_j + 3] - points[pb_d_j + 3];
                error_4 = x[pa_d_j + 4] - points[pb_d_j + 4];
                error_5 = x[pa_d_j + 5] - points[pb_d_j + 5];
                error_6 = x[pa_d_j + 6] - points[pb_d_j + 6];
                error_7 = x[pa_d_j + 7] - points[pb_d_j + 7];
                error_8 = x[pa_d_j + 8] - points[pb_d_j + 8];
                error_9 = x[pa_d_j + 9] - points[pb_d_j + 9];
                phi_0 += error_0 * error_0; 
                phi_1 += error_1 * error_1; 
                phi_2 += error_2 * error_2; 
                phi_3 += error_3 * error_3; 
                phi_4 += error_4 * error_4; 
                phi_5 += error_5 * error_5; 
                phi_6 += error_6 * error_6; 
                phi_7 += error_7 * error_7; 
                phi_8 += error_8 * error_8; 
                phi_9 += error_9 * error_9; 
            }
            phi = phi_0 + phi_1 + phi_2 + phi_3 + phi_4 + phi_5 + phi_6 + phi_7 + phi_8 + phi_9;
            phi_res = 0;
            for(; j < d; j++){
                error = x[pa_d + j] - points[pb_d + j];
                phi_res += error * error;
            }
            phi += phi_res;

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
}

void generate_random(double* arr, int n){
    for(int i = 0; i < n; i++){
        arr[i] = rand();
    }
}

void test_eval1(){
    srand(time(NULL));
    myInt64 gt_start,cur_start,gt_time, cur_time,
            cur_time_reorder,
            cur_time8,
            cur_time_reorder_rename;
    int N_points = 154, N_x = 154;
    int d = 17;
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
    double* result_reorder = (double*)malloc(N_x * sizeof(double));
    double* result_reorder_rename= (double*)malloc(N_x * sizeof(double));
    double* groundtruth = (double*)malloc(N_x * sizeof(double));

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
    cout << "Check correctness: --------"<<endl;
    cout << "groundtruth: "<<groundtruth[0]<<endl; 

    //---------------------------
    // test for unrolling reorder
    //---------------------------

    for(int i = 0; i < warmup_iter; i++){
        evaluate_surrogate_reorder( x, points,  lambda_c, N_x, N_points, d, result_reorder);
    }
    cur_start = start_tsc();
    for(int i = 0; i < repeat; i++){
        evaluate_surrogate_reorder( x, points,  lambda_c, N_x, N_points, d, result_reorder);
    }
    cur_time_reorder = stop_tsc(cur_start);
    cout << "result of unrolling-reorder: "<< result_reorder[0] << endl;

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

    //---------------------------
    // test for rename
    //---------------------------

    for(int i = 0; i < warmup_iter; i++){
        evaluate_surrogate_reorder_rename( x, points,  lambda_c, N_x, N_points, d, result_reorder_rename);
    }
    cur_start = start_tsc();
    for(int i = 0; i < repeat; i++){
        evaluate_surrogate_reorder_rename( x, points,  lambda_c, N_x, N_points, d, result_reorder_rename);
    }
    cur_time_reorder_rename = stop_tsc(cur_start);
    cout << "result of unrolling-reorder-rename: "<< result_reorder_rename[0] << endl;

    // print out performance
    cout << "Compare running time: -----------" << endl;
    cout << "groundtruth cycles: "<< gt_time/(double)repeat << endl;
    cout << "current cycles of unrolling-reorder-rename: "<< cur_time_reorder_rename/(double)repeat << " and performance improve is: " <<   ((gt_time/(double)repeat)-(cur_time_reorder_rename/(double)repeat)) / (gt_time/(double)repeat) << endl;
    cout << "current cycles of unrolling-8: "<< cur_time8/(double)repeat << " and performance improve is: " <<   ((gt_time/(double)repeat)-(cur_time8/(double)repeat)) / (gt_time/(double)repeat) << endl;
    cout << "current cycles of unrolling-reorder: "<< cur_time_reorder/(double)repeat << " and performance improve is: " <<   ((gt_time/(double)repeat)-(cur_time_reorder/(double)repeat)) / (gt_time/(double)repeat) << endl;

}


int main(){
    test_eval1();
}