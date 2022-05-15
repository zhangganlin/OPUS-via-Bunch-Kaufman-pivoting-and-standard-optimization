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
    solve_diag(D, x, b, n);
    cout << "x should be: ";
    cout << "[1.0000    2.2000   -1.6000    1.2000    1.1667].T\n";
    cout << "solved x:\n";
    for(int i = 0; i < n; i++){
        cout << x[i] << " ";
    }
    cout << endl;
    free(D);
    free(b);
    free(x);
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
    cout << "x should be: ";
    cout << "[1.0000, -0.7500, -0.7500, 1.6500, 0.0526].T\n";
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
    cout << "x should be: ";
    cout << "[3.7833, 1.1500, -3.4000, 0.8316, 0.3684].T\n";
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

    P[0] = 1; P[1]=0; P[2]=2; P[3]=4; P[4] = 3;

    int n = 5;
    double* Pb = (double*)malloc(n*sizeof(double));
    double* DLTPx = (double*)malloc(n*sizeof(double));
    double* LTPx = (double*)malloc(n*sizeof(double));
    double* Px = (double*)malloc(n*sizeof(double));
    for(int i = 0; i < n; i++){
        Pb[i] = b[P[i]];
    }

    solve_lower(L,DLTPx,Pb,n);
    solve_diag(D,LTPx,DLTPx,n); //Assume D is saved in A
    solve_upper(L,Px,LTPx,n);
    for(int i = 0; i < n; i++){
        x[P[i]] = Px[i];
    }

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


int main(){
    // test_solve_diag();
    // test_solve_lower();
    // test_solve_upper();
    // test_solve_BunchKaufman();
    test_LU_solver();
}