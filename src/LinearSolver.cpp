
// #pragma once
//#include "surrogate.hpp"

//#include "LinearSolver.hpp"
// #include <stdlib.h> // for rand() stuff
#include "LinearSolver.h"
#include <stdio.h>

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


    ///////////////Solve///////////////
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
        if(A[i * n + i] == 0){
            printf("divide by zero in Linear Solver\n");
        }
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
        // for(int i = 0; i < M; i++) {
        //                 for(int j = 0; j < M; j++) {
        //                     printf("%ft ", A[i * M + j]);
        //                 }
        //                 printf("\n");
        //             }
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
            
            
            // A_kk = A[k * M + k];
            //     for (i = k + 1; i < M; i++)  L[i * M + k] = A[i * M + k] / A_kk;
            //     for (i = k + 1; i < M; i++){
            //         tmp_d = A[i * M + k];
            //         for (j = i; j < M; j++){
            //             A[j * M + i] -=  L[j * M + k] * tmp_d; //##TODO: need to be checked
            //         }
            //     }
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
                // for(int i = 0; i < M; i++) {
                //         for(int j = 0; j < M; j++) {
                //             printf("%fa ", A[i * M + j]);
                //         }
                //         printf("\n");
                //     }
                pivot[k] = 1;
                k = k + 1;
                //printf("qqqqqqqqq");
            } else if (abs(A[r * M + r]) >= alpha * wr){
                printf("r = %d\n", r);
                printf("k = %d\n", k);
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
                // for(int i = 0; i < M; i++) {
                //         for(int j = 0; j < M; j++) {
                //             printf("%fa ", A[i * M + j]);
                //         }
                //         printf("\n");
                //     }
                for (i = k + 1; i < r; i++){
                    tmp_d = A[i * M + k];
                    A[i * M + k] = A[r * M + i];
                    A[r * M + i] = tmp_d;
                }
                if (k > 0){
                    for (j = 0; j < k; j++){
                        tmp_d = L[k * M + j];
                        //printf("tmp_d = %f\n ", tmp_d);
                        L[k * M + j] = L[r * M + j];
                        //printf("L_r = %f\n ", L[r * M + j]);
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
                // for(int i = 0; i < M; i++) {
                //         for(int j = 0; j < M; j++) {
                //             printf("%ff ", A[i * M + j]);
                //         }
                //         printf("\n");
                //     }
                pivot[k] = 1;
                k = k + 1;
                // for(int i = 0; i < M; i++) {
                //         for(int j = 0; j < M; j++) {
                //             printf("%fa ", A[i * M + j]);
                //         }
                //         printf("\n");
                //     }
            } else {
                //printf("r = %d\n", r);
                //printf("k = %d\n", k);
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
                    //printf("qqqq");
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
                    for(i = k+2; i < M ; i++){
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
                printf("\n k = %d \n", k);
                
            }
        }
    }
    if (pivot[M-1] == 0)
        if(pivot[M-2] == 1)
            pivot[M-1] = 1;
}

void testBunchKaufman1(){
    const int M = 3;
    double* A = (double*)malloc((M * M) * sizeof(double));
    double* L = (double*)malloc((M * M) * sizeof(double));
    int* P = (int*)malloc(M * sizeof(int));
    int* pivot = (int*)malloc(M * sizeof(int));

    A[0] = 36;
    A[1] = 35;
    A[2] = 45;
    A[3] = 18;
    A[4] = 10;
    A[5] = 24;
    A[6] = 7;
    A[7] = 23;
    A[8] = 42;

    BunchKaufman(A, L, P, pivot, M);
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", L[i * M + j]);
        }
        printf("\n");
    } 
    printf("11111111 \n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", A[i * M + j]);
        }
        printf("\n");
    }
    printf("PPPPPPPP \n");
    for(int j = 0; j < M; j++) {
            printf("%d ", P[j]);
        }
    printf("piviot\n");
    for(int j = 0; j < M; j++) {
            printf("%d ", pivot[j]);
        }

    free(A);
    free(L);
    free(P);
    free(pivot);
}
void testBunchKaufman2(){
    const int M = 5;
    double* A = (double*)malloc((M * M) * sizeof(double));
    double* L = (double*)malloc((M * M) * sizeof(double));
    int* P = (int*)malloc(M * sizeof(int));
    int* pivot = (int*)malloc(M * sizeof(int));
    A[0] = 211;  
    A[1] = 63;
    A[2] = 252;
    A[3] = 569;
    A[4] = 569;
    A[5] = 63;
    A[6] = 27;
    A[7] = 72;
    A[8] = 81;
    A[9] = 81;
    A[10] = 252;
    A[11] = 72;
    A[12] = 287;
    A[13] = 608;
    A[14] = 608;
    A[15] = 569;
    A[16] = 81;
    A[17] = 608;
    A[18] = 4429;
    A[19] = 1902;
    A[20] = 569;
    A[21] = 81;  
    A[22] = 608;
    A[23] = 1902;
    A[24] = 1902;
    BunchKaufman(A, L, P, pivot, M);
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", L[i * M + j]);
        }
        printf("\n");
    } 
    printf("11111111 \n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", A[i * M + j]);
        }
        printf("\n");
    }
    printf("PPPPPPPP \n");
    for(int j = 0; j < M; j++) {
            printf("%d ", P[j]);
        }
    printf("piviot\n");
    for(int j = 0; j < M; j++) {
            printf("%d ", pivot[j]);
        }

    free(A);
    free(L);
    free(P);
    free(pivot);
}

void testBunchKaufman3(){
        int N = 2;
    int func_dim = 1;
    int d = 1;
    const int M = N + d + 1;
    double* A = (double*)malloc((M * M) * sizeof(double));
    double* L = (double*)malloc((M * M) * sizeof(double));
    int* P = (int*)malloc(M * sizeof(int));
    int* pivot = (int*)malloc(M * sizeof(int));
    int i, j;
    A[0] = 6;
    A[1] = 12;
    A[2] = 3;
    A[3] = -6;
    A[4] = 12;
    A[5] = -8;
    A[6] = -13;
    A[7] = 4;
    A[8] = 3;
    A[9] = -13;
    A[10] = -7;
    A[11] = 1;
    A[12] = -6;
    A[13] = 4;
    A[14] = 1;
    A[15] = 6;
    //(double* A, double* L, int* P, int* pivot, int M)
    // for(int i = 0; i < M; i++) {
    //     for(int j = 0; j < M; j++) {
    //         printf("%f ", A[i * M + j]);
    //     }
    //     printf("\n");
    // }
    BunchKaufman(A, L, P, pivot, M);
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", L[i * M + j]);
        }
        printf("\n");
    } 
    printf("11111111 \n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", A[i * M + j]);
        }
        printf("\n");
    }
    printf("PPPPPPPP \n");
    for(int j = 0; j < M; j++) {
            printf("%d ", P[j]);
        }
    printf("piviot\n");
    for(int j = 0; j < M; j++) {
            printf("%d ", pivot[j]);
        }

    free(A);
    free(L);
    free(P);
    free(pivot);
    printf("\n--------test1--------\n");

}
// int main(){
//     testBunchKaufman1();
//     //testBunchKaufman3();
//     //testBunchKaufman2();
// }


// int main(){
//     double **A = (double **)malloc(4 * sizeof(double *));
//     for (int i=0; i<4; i++) {
//         A[i] = (double *)malloc(4 * sizeof(double));
//     }
//     double *b = (double *)malloc(4 * sizeof(double));
//     double *sol = (double *)malloc(4 * sizeof(double));
//     int N = 2;
//     int d = 1;

//     int i, j;

//     double tmp = 1.0;
//     for (i = 0; i < 4; i++){
//         for (j = 0; j < 4; j++){
//             if (i == j)
//                 A[i][j] = tmp;
//                 tmp += 1.0;
//         }
//     }
//     for (i = 0; i < 4; i++)
//         b[i] = tmp;

//     LUdecomp(A, b, sol, N, d);

//     for (i = 0; i < 4; i++)
//         printf("%f ", sol[i]);

//     free(A);
//     free(b);
//     free(sol);
//     return 0;
// }
