#include "surrogate.hpp"

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
    for(int pa = 0; pa < N; pa++){
        for(int pb = 0; pb < N; pb++){
            phi = 0;
            // flops: 3d
            for(int j = 0; j < d; j++){
                error = points[pa * d + j] - points[pb * d + j];
                phi += error * error;
            }
            sq_phi = sqrt(phi); //1
            phi = sq_phi * phi; //2
            A[pa * (N + d + 1) + pb] = phi;
        }
    }
    
    // optimized
    // for(int pa = 0; pa < N; pa++){
    //     A already set to zero
    //     A[pa * (N + d + 1) + pa] = 0;
    //     for(int pb = pa + 1; pb < N; pb++){
    //         phi = 0;
    //         for(int j = 0; j < d; j++){
    //             error = points[pa * d + j] - points[pb * d + j];
    //             phi += error * error;
    //         }
    //         phi = sqrt(phi);
    //         phi = phi * phi * phi;
    //         A[pa * (N + d + 1) + pb] = phi;
    //         A[pb * (N + d + 1) + pa] = phi;
    //     }
    // }

    
    for(int i = 0; i < N; i++) b[i] = f[i];
    memcpy((void *)b, (void *)f, sizeof(double) * N);
    
    if(N==154){
        cout << "size: " << N+d+1<<endl;
        for(int i =0; i < N+d+1; i ++){
            cout << b[i] << " ";
        }
        cout << endl;
        cout << endl;
    }

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

void evaluate_surrogate_batch_unroll_8( double* x, double* points,  double* lambda_c, int N_x, int N_points, int d, double* output){
    // total flops: 3Nd + 5N + 2d + 1
    for(int pa = 0; pa < N_x; pa++){
        double phi, error, res = 0, sq_phi;
        int id;
        // flops: 3Nd + 5N
        int pa_d = pa * d;
        for(int pb = 0; pb < N_points; pb++){
            double phi, phi_0, phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7; 
            phi_0 = phi_1 = phi_2 = phi_3 = phi_4 = phi_5 = phi_6 = phi_7 = 0; 
            double error_0, error_1, error_2, error_3, error_4, error_5, error_6, error_7, error;
            // flops: 3d

            // id = i * d;
            // for(int j = 0; j < d; j++){
            //     error = x[xidxd + j] - points[id + j];
            //     phi += error * error;
            // }
            int pb_d = pb * d;
            int j = 0;
            for(; j + 7 < d; j += 8){
                int pa_d_j = pa_d + j, pb_d_j = pb_d + j;
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
            for(; j < d; j++){
                error = x[pa_d + j] - points[pb_d + j];
                phi += error * error;
            }

            double sq_phi = sqrt(phi);              // flops: 1
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