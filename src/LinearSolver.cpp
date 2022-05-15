// #pragma once
//#include "surrogate.hpp"

//#include "LinearSolver.hpp"
// #include <stdlib.h> // for rand() stuff
#include "LinearSolver.h"
// #include <time.h> // for time()
// #include <math.h> // for cos(), pow(), sqrt() etc.
// #include <float.h> // for DBL_MAX
// #include <string.h> // for mem*

// With references to Numerical Recipes
void LUdecomp(double* A, double* b, double* sol, int N, int func_dim){
    // N is the number of particles, function_dim is d in the paper
    int i, imax_idx, j, k;
    double imax, tmp;
    double* vv = (double*)malloc((N + func_dim + 1) * sizeof(double));
    int* permute = (int*)malloc((N + func_dim + 1) * sizeof(int));
    double d = 1.0;

    int n = N + d + 1;
    //for (i = 0; i < size; i++) {
    //    A_tmp[i] = (double *)malloc((N + d + 1) * sizeof(double));
    //}
    //double* A_tmp = (double*)malloc((N + d + 1)*(N + d + 1)*sizeof(double));
    for (i = 0; i < n; i++){
        imax = 0.0;
        for (j = 0; j < n; j++){
            tmp = abs(A[i* n + j]);
            if (tmp > imax){
                imax = tmp;
            }
        }
        vv[i] = 1.0 / imax;
    }
    
    // for (i = 0; i < 4; i++)
    //     for (j = 0; j < 4; j++)
    //         printf("%f ", A[i*n + j]);

    for (k = 0; k < n; k++){
        imax = 0;
        imax_idx = k;
        for (i = k; i < n; i++){
            tmp = vv[i] * abs(A[i * n + k]);
            if (tmp > k) {
                imax = tmp;
                imax_idx = i;
            }
        }
        if (k != imax_idx) {
            for (j = 0; j < n; j++){
                tmp = A[imax_idx * n + j];
                A[imax_idx * n + j] = A[k * n + j];
                A[k * n + j] = tmp;
            }
            d = -d;
            vv[imax_idx] = vv[k];
        }
        permute[k] = imax_idx;
        for (i = k + 1; i < n; i++){
            tmp = A[i * n + k] /A[k * n + k];
            A[i * n + k] /= A[k * n + k];
            for (j = k + 1; j < n; j ++){
                A[i * n + j] -= tmp * A[k * n + j];
            }
        }
    }


    //////////////////Solve///////////////
    double sum;
    int ip;
    int ii = 0;
    for (i = 0; i < n; i++)
        sol[i] = b[i];

    for(i = 0; i < n; i++){
        ip = permute[i];
        sum = sol[ip];
        sol[ip] = sol[i];
        if (ii != 0){
            for (j = ii-1; j < i; j++)
                sum -= A[i * n + j] * sol[j];
        } else if (sum != 0.0)
            ii = i + 1;
        sol[i] = sum;
    }

    for (i = n - 1; i >= 0; i--){
        sum = sol[i];
        for (j=i+1; j<n; j++) 
            sum -= A[i * n + j]*sol[j];
        sol[i]=sum/A[i * n + i];
    }

    free(vv);
    free(permute);

}

// This implementation refers to
void BunchKaufman(double* A, double* L, int* P, int* pivot, int M){
    //const int M = N + func_dim + 1; //r = row_size
    const double alpha = (1+sqrt(17))/8;
    int r, i, j, tmp_i;
    int k = 0;
    double w1, wr;
    double tmp_d, A_kk;
    double detE, invE_11, invE_22, invE_12, invE_21;
    //Initialize Matrices
    for (i = 0; i < M; i++) {
        L[i * M + i] = 1.0; 
        P[i] = i;
        pivot[i] = 0.0;
    }
    while (k < M - 2){
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
            for (i = k + 1; i < M; j++){
                tmp_d = A[i * M + k];
                for (j = k + 1; j < M; i++)
                    A[i * M + j] -=  L[j * M + k] * tmp_d; //##TODO: need to be checked
            }
            for (i = 0; i < M; i++) A[i * M + k] = 0.0;
            pivot[k] = 1;
            k = k + 1;
        }
        else{
            wr = 0.0;
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
                for (i = k + 1; i < M; j++){
                    tmp_d = A[i * M + k];
                    for (j = k + 1; j < M; i++)
                        A[i * M + j] -=  L[j * M + k] * tmp_d; //##TODO: need to be checked
                }
                for (i = 0; i < M; i++) A[i * M + k] = 0.0;
                pivot[k] = 1;
                k = k + 1;
                //printf("qqqqqqqqq");
            } else if (abs(A[r * M + r]) >= alpha * wr){
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
                        L[k * M + k] = L[r * M + j];
                        L[r * M + j] = tmp_d;
                    }
                }
                //printf("qqqqqqqqq");
                A_kk = A[k * M + k];
                for (i = k + 1; i < M; i++)  L[i * M + k] = A[i * M + k] / A_kk;
                for (i = k + 1; i < M; j++){
                    tmp_d = A[i * M + k];
                    for (j = k + 1; j < M; i++)
                        A[i * M + j] -=  L[j * M + k] * tmp_d; //##TODO: need to be checked
                }
                for (i = 0; i < M; i++) A[i * M + k] = 0.0;
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
                    A[i * M + k+1] = A[r * M + k+1];
                    A[r * M + i] = tmp_d;
                }
                if (k > 0){
                    //printf("qqqq");
                    for (i = 0; i < k; i++){
                        tmp_d = L[(k+1) * M + i];
                        L[(k+1) * M + i] = L[r * M + i];
                        L[r * M + i] = tmp_d;
                    }
                }
                
                //printf("qqqqqqqqq");
                detE = A[k * M + k] * A[(k+1) * M + k+1] - A[(k+1) * M + k] * A[(k+1) * M + k];
                invE_11 = A[(k+1) * M + k+1] / detE;
                invE_22 = A[k * M + k] / detE;
                invE_12 = - A[(k+1)*M + k] / detE;
                invE_21 = invE_12;
                for (i = k + 2; i < M; i++){
                    L[i * M + k] = A[i * M + k] * invE_11 + A[i * M + k+1] * invE_21;
                    L[i * M + k+1] = A[i * M + k] * invE_12 + A[i * M + k+1] * invE_22;
                }
                for (i = k+2; i < M; i++){
                    for(j = k+2; j < M ; j++){
                        A[j * M + i] -= L[i * M + k] * A[i * M + k] + L[i * M + k+1] * A[i * M + k+1];
                    }
                }
                for (i = k+2; i < M; i++){
                    A[i * M + k] = 0.0;
                    A[i * M + k+1] = 0.0;
                }
                pivot[k] = 2;
                k = k + 2;
            }
        }
    }
    if (pivot[M-1] == 0)
        if(pivot[M-2] == 1)
            pivot[M-1] = 1;
}


void solve_lower(double* L, double* x, double* b, int n){
  int i,j;
  double s;

  for(i = 0; i < n; i++){
        s = 0;
        for(j = 0; j < i; j++) {
            s = s + L[i * n + j] * x[j];
        }
        if(L[i * n + i] == 0.0){
            printf("LU divide by zero");
            exit(0);
        }
        x[i] = (b[i] - s) /  L[i * n + i];
   }
}

