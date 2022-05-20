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
    memcpy((void *)f, (void *)b, sizeof(double) * N);

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