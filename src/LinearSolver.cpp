#include "LinearSolver.h"
#include "blockbk.h"
#include <stdio.h>
#include "test_utils.h"

void solve_lower(double* L, double* x, double* b, int n){
    int i,j;
    double s;
    for(i = 0; i < n; i++){
        s = 0;
        for(j = 0; j < i; j++) {
            s = s + L[i * n + j] * x[j];
        }
        
        #ifdef FLOP_COUNTER
            flops() += i*2;
        #endif

        if(L[i * n + i] == 0.0){
            printf("LU divide by zero");
            exit(0);
        }
        x[i] = (b[i] - s) /  L[i * n + i];

        #ifdef FLOP_COUNTER
            flops() += 2;
        #endif
    }
}

//input is a Lower matrix, this function first transpose it and solve
void solve_upper(double* L, double* x, double* b, int n){
    int i,j;
    double s;
    for(i = n-1; i >= 0; i--){
        s = 0;
        for(j = i + 1; j < n; j++) {
            s = s + L[j * n + i] * x[j];
        }
        #ifdef FLOP_COUNTER
            flops() += (n-i-1)*2;
        #endif

        if(L[i * n + i] == 0.0){
            printf("LU divide by zero");
            exit(0);
        }
        x[i] = (b[i] - s) /  L[i * n + i];
        #ifdef FLOP_COUNTER
            flops() += 2;
        #endif
    }
}

void solve_diag(double* D, int* pivot, double* x, double* b, int n){
    int i,j;
    double s;
    for(i = 0; i < n; i++){
        if(pivot[i]==1){
            if(D[i*n+i]==0){
                printf("D is singular!\n");
            }
            x[i] = b[i]/D[i*n+i];
            #ifdef FLOP_COUNTER
                flops() += 1;
            #endif
        }
        else if(pivot[i]==2){
            double a = D[i*n+i];
            double tb = D[i*n+i+1];
            double c = D[(i+1)*n+i+1];
            double d1 = b[i];
            double d2 = b[i+1];
            double denorminator = c*a-tb*tb;
            if(denorminator==0){
                printf("D is singular!\n");
            }
            x[i] = (c*d1-tb*d2)/denorminator;
            x[i+1] = (d1*tb-a*d2)/(-denorminator);
            #ifdef FLOP_COUNTER
                flops() += 11;
            #endif
            
        }else if(pivot[i]==0){
            continue;
        }
    }
}

void solve_BunchKaufman(double* A, double* x, double* b, int n){
    double* L = (double*)calloc(n*n,sizeof(double));
    double* Pb = (double*)malloc(n*sizeof(double));
    double* DLTPx = (double*)malloc(n*sizeof(double));
    double* LTPx = (double*)malloc(n*sizeof(double));
    double* Px = (double*)malloc(n*sizeof(double));
    int* P = (int*)malloc(n*sizeof(int));

    int* pivot = (int*)calloc(n,sizeof(int));

    // BunchKaufman_block(A,L,P,pivot,n,BLOCK_SIZE);

    int* pivot_idx = (int*)calloc(n+1,sizeof(int));
    BunchKaufman_subblock(A,L,P,pivot,pivot_idx,n,0,n);
    free(pivot_idx);

    // BunchKaufman_noblock(A,L,P,pivot,n);
    // BunchKaufman(A,L,P,pivot,n);
    for(int i = 0; i < n; i++){
        Pb[i] = b[P[i]];
    }
    solve_lower(L,DLTPx,Pb,n);
    solve_diag(A,pivot,LTPx,DLTPx,n); //Assume D is saved in A
    solve_upper(L,Px,LTPx,n);

    for(int i = 0; i < n; i++){
        x[P[i]] = Px[i];
    }

    free(L);
    free(Pb);
    free(DLTPx);
    free(LTPx);
    free(Px);
    free(P);
    free(pivot);
}
