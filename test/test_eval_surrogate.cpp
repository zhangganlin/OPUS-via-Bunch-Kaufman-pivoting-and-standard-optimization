#include "surrogate.hpp"
#include <iostream>
#include "tsc_x86.h"
#include "randomlhs.hpp"
#include <time.h>

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
            phi = sq_phi * phi;      // flops: 1
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
}

void evaluate_surrogate_unroll_8_sqrt_unroll_4( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    double res, sq_phi;
    double phi, phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7, phi_res; 
    double error_0, error_1, error_2, error_3, error_4, error_5, error_6, error_7, error;
    double res_0, res_1, res_2, res_3;
    double* history_phi = (double*)malloc(sizeof(double) * N_points);
    int id, j, pa_d = 0, pb_d, pa_d_j, pb_d_j;
    for(int pa = 0; pa < N_x; pa++, pa_d += d){
        res_0 = res_1 = res_2 = res_3 = 0;
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

        for(int pb = 0; pb + 3 < N_points; pb+=4){
            phi_0 = history_phi[pb];
            phi_1 = history_phi[pb + 1];
            phi_2 = history_phi[pb + 2];
            phi_3 = history_phi[pb + 3];
            phi_0 = phi_0 * sqrt(phi_0);              
            phi_1 = phi_1 * sqrt(phi_1);              
            phi_2 = phi_2 * sqrt(phi_2);              
            phi_3 = phi_3 * sqrt(phi_3);              

            res_0 += phi_0 * lambda_c[pb];            
            res_1 += phi_1 * lambda_c[pb + 1];            
            res_2 += phi_2 * lambda_c[pb + 2];            
            res_3 += phi_3 * lambda_c[pb + 3];            
        }
        res = res_0 + res_1 + res_2 + res_3;
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
            cur_time8,
            cur_time8_sqrt,
            cur_time8_sqrt4;
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
    double* result8_sqrt = (double*)malloc(N_x * sizeof(double));
    double* result8_sqrt4 = (double*)malloc(N_x * sizeof(double));
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
        evaluate_surrogate_unroll_8_sqrt_unroll_4( x, points,  lambda_c, N_x, N_points, d, result8_sqrt4);
    }
    cur_start = start_tsc();
    for(int i = 0; i < repeat; i++){
        evaluate_surrogate_unroll_8_sqrt_unroll_4( x, points,  lambda_c, N_x, N_points, d, result8_sqrt4);
    }
    cur_time8_sqrt4 = stop_tsc(cur_start);
    cout << "result of unrolling-8-sqrt4: "<< result8_sqrt4[0] << endl;


    // print out performance
    cout << "Compare running time: -----------" << endl;
    cout << "groundtruth cycles: "<< gt_time/(double)repeat << endl;
    cout << "current cycles of unrolling-8: "<< cur_time8/(double)repeat << " and performance improve is: " <<   ((gt_time/(double)repeat)-(cur_time8/(double)repeat)) / (gt_time/(double)repeat) << endl;
    cout << "current cycles of unrolling-8-sqrt: "<< cur_time8_sqrt/(double)repeat << " and performance improve is: " <<   ((gt_time/(double)repeat)-(cur_time8_sqrt/(double)repeat)) / (gt_time/(double)repeat) << endl;
    cout << "current cycles of unrolling-8-sqrt4: "<< cur_time8_sqrt4/(double)repeat << " and performance improve is: " <<   ((gt_time/(double)repeat)-(cur_time8_sqrt4/(double)repeat)) / (gt_time/(double)repeat) << endl;

}


int main(){
    test_eval1();
}