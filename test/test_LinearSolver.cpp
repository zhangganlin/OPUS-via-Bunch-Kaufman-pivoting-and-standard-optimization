#include "LinearSolver.h"
#include <iostream>
#include <cstring>
using namespace std;



void test_solve_diag(){
    /*
    A =
        3     0     0     0     0
        0     4     3     0     0
        0     3     1     0     0
        0     0     0     5     0
        0     0     0     0     6

    b = 
        [3     4     5     6     7].T
        
    x should be: [1.0000    2.2000   -1.6000    1.2000    1.1667].T
    */
    int n = 5;
    double* D = (double*)malloc(25 * sizeof(double));
    double* b = (double*)malloc(5 * sizeof(double));
    double* x = (double*)malloc(5 * sizeof(double));
    int* pivot = (int*)malloc(5*sizeof(int));
    for(int i = 0; i < 25; i++){
        D[i] = 0;        
    }
    D[0] = 3;
    D[1 * 5 + 1] = 4; D[1 * 5 + 2] = 3;
    D[2 * 5 + 1] = 3; D[2 * 5 + 2] = 1;
    D[3 * 5 + 3] = 5; D[4 * 5 + 4] = 6;
    for(int i = 3; i < 8; i++){
        b[i-3] = i;
    }
    pivot[0] = 1;pivot[1] = 2;pivot[2] = 0;pivot[3] = 1;pivot[4] = 1;
    solve_diag(D,pivot, x, b, n);
    cout << "x should be:\n";
    cout << "1 2.2 -1.6 1.2 1.16667\n";
    cout << "solved x:\n";
    for(int i = 0; i < n; i++){
        cout << x[i] << " ";
    }
    cout << endl;
    free(D);
    free(b);
    free(x);
    free(pivot);
}   

void test_solve_lower(){
    /*
    A =
     3     0     0     0     0
     7     4     0     0     0
     8     3     1     0     0
     9     8     7     5     0
     9     8     7     5    19
    b = [3,4,5,6,7].T
    x should be: [1.0000, -0.7500, -0.7500, 1.6500, 0.0526].T
    */
    int n = 5;
    double* D = (double*)malloc(25 * sizeof(double));
    double* b = (double*)malloc(5 * sizeof(double));
    double* x = (double*)malloc(5 * sizeof(double));
    for(int i = 0; i < 25; i++){
        D[i] = 0;        
    }
    D[0] = 3;
    D[1 * 5 + 0] = 7; D[1 * 5 + 1] = 4;
    D[2 * 5 + 0] = 8; D[2 * 5 + 1] = 3; D[2 * 5 + 2] = 1; 
    D[3 * 5 + 0] = 9; D[3 * 5 + 1] = 8; D[3 * 5 + 2] = 7; D[3 * 5 + 3] = 5;
    D[4 * 5 + 0] = 9; D[4 * 5 + 1] = 8; D[4 * 5 + 2] = 7; D[4 * 5 + 3] = 5; D[4 * 5 + 4] = 19;
    for(int i = 0; i < 5; i++){
        b[i] = i+3;
    }
    solve_lower(D,x,b,n);
    cout << "x should be:\n";
    cout << "1.0000, -0.7500, -0.7500, 1.6500, 0.0526\n";
    cout << "solved x:\n";
    for(int i = 0; i < n; i++){
        cout << x[i] << " ";
    }
    cout << endl;
    free(D);
    free(x);
    free(b);
}

void test_solve_upper(){
    /*
    A =
     3     0     0     0     0
     7     4     0     0     0
     8     3     1     0     0
     9     8     7     5     0
     9     8     7     5    19
    b = [3,4,5,6,7].T
    x should be: [3.7833, 1.1500, -3.4000, 0.8316, 0.3684].T
    */
    int n = 5;
    double* D = (double*)malloc(25 * sizeof(double));
    double* b = (double*)malloc(5 * sizeof(double));
    double* x = (double*)malloc(5 * sizeof(double));
    for(int i = 0; i < 25; i++){
        D[i] = 0;        
    }
    D[0] = 3;
    D[1 * 5 + 0] = 7; D[1 * 5 + 1] = 4;
    D[2 * 5 + 0] = 8; D[2 * 5 + 1] = 3; D[2 * 5 + 2] = 1; 
    D[3 * 5 + 0] = 9; D[3 * 5 + 1] = 8; D[3 * 5 + 2] = 7; D[3 * 5 + 3] = 5;
    D[4 * 5 + 0] = 9; D[4 * 5 + 1] = 8; D[4 * 5 + 2] = 7; D[4 * 5 + 3] = 5; D[4 * 5 + 4] = 19;
    for(int i = 0; i < 5; i++){
        b[i] = i+3;
    }
    solve_upper(D,x,b,n);
    cout << "x should be:\n";
    cout << "3.7833, 1.1500, -3.4000, 0.8316, 0.3684\n";
    cout << "solved x:\n";
    for(int i = 0; i < n; i++){
        cout << x[i] << " ";
    }
    cout << endl;
    free(D);
    free(x);
    free(b);
}

void test_solve_BunchKaufman(){

    /*
    D = 
     3     0     0     0     0
     0     4     9     0     0
     0     9     5     0     0
     0     0     0     6     0
     0     0     0     0     7

    L =
     3     0     0     0     0
     7     4     0     0     0
     8     3     1     0     0
     9     8     7     5     0
     9     8     7     5    19

    P = 
     0     1     0     0     0
     1     0     0     0     0
     0     0     1     0     0
     0     0     0     0     1
     0     0     0     1     0

    b = [3,4,5,6,7].T
    x should be [0.4339, 1.0693, -0.8307, -0.0004, 0.0943].T

    */

    int* P = (int*)malloc(5*sizeof(int));
    int* pivot = (int*)malloc(5*sizeof(int));
    double* L = (double*)malloc(25 * sizeof(double));
    double* b = (double*)malloc(5 * sizeof(double));
    double* x = (double*)malloc(5 * sizeof(double));
    double* D = (double*)malloc(25 * sizeof(double));
    for(int i = 0; i < 25; i++){
        L[i] = 0;        
    }
    L[0] = 3;
    L[1 * 5 + 0] = 7; L[1 * 5 + 1] = 4;
    L[2 * 5 + 0] = 8; L[2 * 5 + 1] = 3; L[2 * 5 + 2] = 1; 
    L[3 * 5 + 0] = 9; L[3 * 5 + 1] = 8; L[3 * 5 + 2] = 7; L[3 * 5 + 3] = 5;
    L[4 * 5 + 0] = 9; L[4 * 5 + 1] = 8; L[4 * 5 + 2] = 7; L[4 * 5 + 3] = 5; L[4 * 5 + 4] = 19;
    for(int i = 0; i < 5; i++){
        b[i] = i+3;
        D[i*5+i] = i + 3;
    }
    D[5*2 + 1] = D[5*1 + 2] = 9;

    P[0] = 2; P[1]=1; P[2]=3; P[3]=5; P[4] = 4;
    pivot[0] = 1;pivot[1] = 2;pivot[2] = 0;pivot[3] = 1;pivot[4] = 1;

    int n = 5;
    double* Pb = (double*)malloc(n*sizeof(double));
    double* DLTPx = (double*)malloc(n*sizeof(double));
    double* LTPx = (double*)malloc(n*sizeof(double));
    double* Px = (double*)malloc(n*sizeof(double));
    for(int i = 0; i < n; i++){
        Pb[i] = b[P[i]-1];
    }

    solve_lower(L,DLTPx,Pb,n);
    solve_diag(D,pivot,LTPx,DLTPx,n); //Assume D is saved in A
    solve_upper(L,Px,LTPx,n);
    for(int i = 0; i < n; i++){
        x[P[i]-1] = Px[i];
    }
    cout << "x should be:\n0.433896 1.06931 -0.830719 -0.000395726 0.0942846\n";
    cout << "solved x:\n";
    for(int i = 0; i < n; i++){
        cout << x[i] << " ";
    }
    cout << endl;


    free(D);
    free(P);
    free(L);
    free(x);
    free(b);
    free(pivot);
    free(Pb);free(DLTPx);free(Px);free(LTPx);
    
}

void test_LU_solver(){
    /*
    A =
         211          63         252         569         569
          63          27          72          81          81
         252          72         287         608         608
         569          81         608        4429        1902
         569          81         608        1902        1902
    b = [3,4,5,6,7].T
    x should be: [0.4339, 1.0693, -0.8307, -0.0004, 0.0943].T
    */
    double* A = (double*)malloc(25 * sizeof(double));  
    double* b = (double*)malloc(5 * sizeof(double)); 
    double* x = (double*)malloc(5 * sizeof(double));
    A[0 * 5 + 0] = 211; A[0 * 5 + 1] = 63; A[0 * 5 + 2] = 252; A[0 * 5 + 3] = 569 ; A[0 * 5 + 4] = 569;
    A[1 * 5 + 0] = 63 ; A[1 * 5 + 1] = 27; A[1 * 5 + 2] = 72 ; A[1 * 5 + 3] = 81  ; A[1 * 5 + 4] = 81;
    A[2 * 5 + 0] = 252; A[2 * 5 + 1] = 72; A[2 * 5 + 2] = 287; A[2 * 5 + 3] = 608 ; A[2 * 5 + 4] = 608;
    A[3 * 5 + 0] = 569; A[3 * 5 + 1] = 81; A[3 * 5 + 2] = 608; A[3 * 5 + 3] = 4429; A[4 * 5 + 4] = 1902;
    A[4 * 5 + 0] = 569; A[4 * 5 + 1] = 81; A[4 * 5 + 2] = 608; A[4 * 5 + 3] = 1902; A[5 * 5 + 4] = 1902;
    
    for(int i = 0; i < 5; i++){
        b[i] = i+3;
    }

    LUdecomp(A,b,x,3,2);
    cout << "solved x:\n";
    for(int i = 0; i < 5; i++){
        cout << x[i] << " ";
    }
    cout << endl;
    free(A);
    free(x);
    free(b);
}


void testBunchKaufman1(){
    const int M = 3;
    double* A = (double*)malloc((M * M) * sizeof(double));
    double* L = (double*)malloc((M * M) * sizeof(double));
    int* P = (int*)malloc(M * sizeof(int));
    int* pivot = (int*)malloc(M * sizeof(int));

    A[0] = 27;    A[1] = 98;    A[2] = 49;
    A[3] = 98;    A[4] = 83;    A[5] = 54;
    A[6] = 49;    A[7] = 54;    A[8] = 37;

    BunchKaufman(A, L, P, pivot, M);
    printf("L \n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", L[i * M + j]);
        }
        printf("\n");
    } 
    printf("D \n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", A[i * M + j]);
        }
        printf("\n");
    }
    printf("P \n");
    for(int j = 0; j < M; j++) {
        printf("%d ", P[j]);
    }
    printf("\npiviot\n");
    for(int j = 0; j < M; j++) {
        printf("%d ", pivot[j]);
    }
    printf("\n");

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
    A[0]  = 211; A[1]  = 63; A[2]  = 252; A[3]  =  569; A[4]  =  569;
    A[5]  =  63; A[6]  = 27; A[7]  =  72; A[8]  =   81; A[9]  =   81;
    A[10] = 252; A[11] = 72; A[12] = 287; A[13] =  608; A[14] =  608;
    A[15] = 569; A[16] = 81; A[17] = 608; A[18] = 4429; A[19] = 1902;
    A[20] = 569; A[21] = 81; A[22] = 608; A[23] = 1902; A[24] = 1902;
    BunchKaufman(A, L, P, pivot, M);
    printf("L \n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", L[i * M + j]);
        }
        printf("\n");
    } 
    printf("D \n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", A[i * M + j]);
        }
        printf("\n");
    }
    printf("P \n");
    for(int j = 0; j < M; j++) {
        printf("%d ", P[j]);
    }
    printf("\npiviot\n");
    for(int j = 0; j < M; j++) {
        printf("%d ", pivot[j]);
    }
    printf("\n");
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
    A[0]  =  6;  A[1]  =  12;  A[2]  =   3;  A[3]  = -6;
    A[4]  = 12;  A[5]  =  -8;  A[6]  = -13;  A[7]  =  4;
    A[8]  =  3;  A[9]  = -13;  A[10] =  -7;  A[11] =  1;
    A[12] = -6;  A[13] =   4;  A[14] =   1;  A[15] =  6;
    BunchKaufman(A, L, P, pivot, M);
    printf("L \n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", L[i * M + j]);
        }
        printf("\n");
    } 
    printf("D \n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", A[i * M + j]);
        }
        printf("\n");
    }
    printf("P \n");
    for(int j = 0; j < M; j++) {
        printf("%d ", P[j]);
    }
    printf("\npiviot\n");
    for(int j = 0; j < M; j++) {
        printf("%d ", pivot[j]);
    }
    printf("\n");

    free(A);
    free(L);
    free(P);
    free(pivot);
}

void testBunchKaufman4(){
    const int M = 5;
    double* A = (double*)malloc((M * M) * sizeof(double));
    double* L = (double*)malloc((M * M) * sizeof(double));
    int* P = (int*)malloc(M * sizeof(int));
    int* pivot = (int*)malloc(M * sizeof(int));
    A[0]  = 211; A[1]  = 63; A[2]  = 252; A[3]  =  569; A[4]  =  569;
    A[5]  =  63; A[6]  = 27; A[7]  =   0; A[8]  =   81; A[9]  =   81;
    A[10] = 252; A[11] =  0; A[12] = 287; A[13] =  608; A[14] =  608;
    A[15] = 569; A[16] = 81; A[17] = 608; A[18] =    0; A[19] = 1902;
    A[20] = 569; A[21] = 81; A[22] = 608; A[23] = 1902; A[24] = 1902;
    BunchKaufman(A, L, P, pivot, M);
    printf("L \n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", L[i * M + j]);
        }
        printf("\n");
    } 
    printf("D \n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < M; j++) {
            printf("%f ", A[i * M + j]);
        }
        printf("\n");
    }
    printf("P \n");
    for(int j = 0; j < M; j++) {
        printf("%d ", P[j]);
    }
    printf("\npiviot\n");
    for(int j = 0; j < M; j++) {
        printf("%d ", pivot[j]);
    }
    printf("\n");
    free(A);
    free(L);
    free(P);
    free(pivot);
}

void test_BunchKaufmanAndSolver(){
    const int M = 5;
    double* A = (double*)malloc((M * M) * sizeof(double));
    double* x = (double*)malloc(M * sizeof(double));
    double* b = (double*)malloc(M * sizeof(double));

    A[0]  = 211; A[1]  = 63; A[2]  = 252; A[3]  =  569; A[4]  =  569;
    A[5]  =  63; A[6]  = 27; A[7]  =   0; A[8]  =   81; A[9]  =   81;
    A[10] = 252; A[11] =  0; A[12] = 287; A[13] =  608; A[14] =  608;
    A[15] = 569; A[16] = 81; A[17] = 608; A[18] =    0; A[19] = 1902;
    A[20] = 569; A[21] = 81; A[22] = 608; A[23] = 1902; A[24] = 1902;
    for(int i = 0; i < 5; i++){
        b[i] = i+3;
    }

    solve_BunchKaufman(A,x,b,5);
    cout << "x should be:\n0.0778 -0.0039 -0.0301 0.0005 -0.0103\n";
    cout << "solved x:\n";
    for(int i = 0; i < 5; i++){
        cout << x[i] << " ";
    }
    cout << endl;

    free(A);
    free(x);
    free(b);
}


int main(){
    test_solve_diag();
    test_solve_lower();
    test_solve_upper();
    test_solve_BunchKaufman();
    // test_LU_solver();
    // testBunchKaufman1();
    // testBunchKaufman2();
    // testBunchKaufman3();
    // testBunchKaufman4();
    test_BunchKaufmanAndSolver();
}