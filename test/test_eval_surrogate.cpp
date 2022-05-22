#include "surrogate.hpp"
#include <iostream>
#include "tsc_x86.h"
#include "randomlhs.hpp"

using namespace std;

void evaluate_surrogate_gt( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    for(int x_idx = 0; x_idx < N_x; x_idx++){
        double phi, error, res = 0, sq_phi;
        // flops: 3Nd + 5N
        for(int i = 0; i < N_points; i++){
            phi = 0;
            // flops: 3d
            for(int j = 0; j < d; j++){
                error = x[x_idx * d + j] - points[i * d + j];
                phi += error * error;
            }
            phi = sqrt(phi);            // flops: 1
            phi = phi * phi * phi;      // flops: 2
            res += phi * lambda_c[i];   // flops: 2
        }
        // flops: 2d
        for(int i = 0; i < d; i++){
            res += x[x_idx * d + i] * lambda_c[N_points + i];
        }
        // flops: 1
        res += lambda_c[N_points + d];

        output[x_idx] = res;
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



void generate_random(double* arr, int n){
    for(int i = 0; i < n; i++){
        arr[i] = rand();
    }
}

void test_eval1(){
    myInt64 gt_start,cur_start,gt_time, cur_time, cur_time2, cur_time4, cur_time8;
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
    double* result2 = (double*)malloc(N_x * sizeof(double));
    double* result4 = (double*)malloc(N_x * sizeof(double));
    double* result8 = (double*)malloc(N_x * sizeof(double));
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
    // test for unrolling 4
    //----------------------

    for(int i = 0; i < warmup_iter; i++){
        evaluate_surrogate_unroll_4( x, points,  lambda_c, N_x, N_points, d, result4);
    }
    cur_start = start_tsc();
    for(int i = 0; i < repeat; i++){
        evaluate_surrogate_unroll_4( x, points,  lambda_c, N_x, N_points, d, result4);
    }
    cur_time = stop_tsc(cur_start);
    cout << "result of unrolling-4: "<< result4[0] << endl;
    // cout << "current cycles 4: "<< cur_time/(double)repeat << endl;
    // cout << "performance improve 4:" <<  (gt_time/(double)repeat) / (cur_time/(double)repeat) << endl;

    //----------------------
    // test for unrolling 2
    //----------------------
    for(int i = 0; i < warmup_iter; i++){
        evaluate_surrogate_unroll_2( x, points,  lambda_c, N_x, N_points, d, result2);
    }
    cur_start = start_tsc();
    for(int i = 0; i < repeat; i++){
        evaluate_surrogate_unroll_2( x, points,  lambda_c, N_x, N_points, d, result2);
    }
    cur_time2 = stop_tsc(cur_start);
    cout << "current result 2: "<< result2[0] << endl;
    // cout << "current cycles 2: "<< cur_time2/(double)repeat << endl;
    // cout << "performance improve 2:" <<  (gt_time/(double)repeat) / (cur_time2/(double)repeat) << endl;


    // print out performance
    cout << "Compare running time: -----------" << endl;
    cout << "groundtruth cycles: "<< gt_time/(double)repeat << endl;
    cout << "current cycles of unrolling-8: "<< cur_time8/(double)repeat << "and performance improve is: " <<   ((gt_time/(double)repeat)-(cur_time8/(double)repeat)) / (gt_time/(double)repeat) << endl;
    cout << "current cycles of unrolling-4: "<< cur_time4/(double)repeat << "and performance improve is: " <<   ((gt_time/(double)repeat)-(cur_time4/(double)repeat)) / (gt_time/(double)repeat) << endl;
    cout << "current cycles of unrolling-2: "<< cur_time2/(double)repeat << "and performance improve is: " <<   ((gt_time/(double)repeat)-(cur_time2/(double)repeat)) / (gt_time/(double)repeat) << endl;

}


int main(){
    test_eval1();
}