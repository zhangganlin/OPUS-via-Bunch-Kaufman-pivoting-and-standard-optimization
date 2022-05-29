#include "test_utils.h"
#include <iostream>
#include "blockbk.h"

using namespace std;

void test_matrix_update(){
    double* A = (double*)malloc(5*5*sizeof(double));
    double* L = (double*)malloc(5*5*sizeof(double));
    double* D = (double*)malloc(5*5*sizeof(double));
    double* A11 = (double*)malloc(2*2*sizeof(double));
    double* A21 = (double*)malloc(1*2*sizeof(double));
    double* A22 = (double*)malloc(1*1*sizeof(double));
    double* L10 = (double*)malloc(2*2*sizeof(double));
    double* L20 = (double*)malloc(1*2*sizeof(double));
    double* L10T = (double*)malloc(2*2*sizeof(double));
    double* L20T = (double*)malloc(1*2*sizeof(double));
    int* pivot = (int*)malloc(1*5*sizeof(int)); //useless
    int n = 5;
    int k = 2;
    int r = 2;

    generate_random_dense(A, 5, 5);
    matrix_get_block(A,2,2,2,2,A11,n);
    matrix_get_block(A,4,2,1,2,A21,n);
    matrix_get_block(A,4,4,1,1,A22,n);

    pivot[0] = 2;
    pivot[1] = 0;
    pivot[2] = 1;
    pivot[3] = 1;
    pivot[4] = 1;
    generate_random_d(D, 5, pivot);
    generate_random_l(L, 5);

    matrix_get_block(L,2,0,2,2,L10,n);
    matrix_get_block(L,4,0,1,2,L20,n);
    matrix_transpose(L10,L10T,2);
    matrix_transpose(L20,L20T,1,2);

    cout << "matrix A to be updated: " << endl;
    print_matrix(A, 5, 5, 4);    

    cout << "L:" << endl;
    print_matrix(L,5,5,4);


    matrix_update(A, D, L, pivot, n, k, r);

    cout << "lower matrix is: " << endl;
    print_matrix(L, 5, 5, 4);

    cout << "updated matrix A' is: " << endl;
    print_matrix(A, 5, 5, 4);    
    // print_matrix(a, 2, 2, 4);

    cout << "diag matrix D is: " << endl;
    print_matrix(D, 5, 5, 4);

    free (A); free(A11); free(A21); free(A22);
    free (L);
    free (D);
    free (pivot);
    free(L10);free(L10T);free(L20);free(L20T);
}

void test_BunchKaufman_subblock(){
    int n = 15;
    int bigger_b_size = 6;
    int b_size = 3;
    double* A_block = (double*)malloc(b_size*b_size*sizeof(double));
    double* L_block = (double*)malloc(b_size*b_size*sizeof(double));
    int* P_block = (int*)malloc(b_size*sizeof(int));
    int* pivot_block = (int*)malloc(b_size*sizeof(int));

    double* A = (double*)malloc(n*n*sizeof(double));
    double* L = (double*)malloc(n*n*sizeof(double));
    int* P = (int*)malloc(n*sizeof(int));
    int* pivot = (int*)malloc(n*sizeof(int));

    generate_random_symmetry(A_block,b_size);

    matrix_set_block(A,12,12,b_size,b_size,A_block,n);

    cout << "A_block:\n";
    print_matrix(A_block,b_size,b_size,4);
    cout << "A:\n";
    print_matrix(A,n,n,4);

    BunchKaufman_noblock(A_block,L_block,P_block,pivot_block,b_size);
    BunchKaufman_subblock(A,L,P,pivot,n,12,bigger_b_size);

    cout << "L_block:\n";
    print_matrix(L_block,b_size,b_size,4);
    cout << "L:\n";
    print_matrix(L,n,n,4);

    cout << "B_block:\n";
    print_matrix(A_block,b_size,b_size,4);
    cout << "B:\n";
    print_matrix(A,n,n,4);

    cout << "P_block:\n";
    print_vector(P_block,b_size);
    cout << "P:\n";
    print_vector(P,n);

    cout << "pivot_block:\n";
    print_vector(pivot_block,b_size);
    cout << "pivot:\n";
    print_vector(pivot,n);


    free(A_block); free(L_block); free(P_block); free(pivot_block);
    free(A); free(L); free(P); free(pivot);
}

void test_permute(){
    int n = 10;
    int pn = 5;
    double* A = (double*)malloc(n*n*sizeof(double));
    int* P = (int*)malloc(n*sizeof(int));
    for(int i = 0; i < 5; i++) P[i] = 4 - i;
    for(int i = 5; i < 10; i++) P[i] = 14 - i;
    print_vector(P, n);
    generate_random_dense(A, n, n);
    print_matrix(A, n, n, 4);
    cout << endl;
    permute(A, P, 5, 10, 0, 5, n, 0);
    print_matrix(A, n, n, 4);
    cout << endl;
    permute(A, P, 5, 10, 0, 5, n, 1);
    print_matrix(A, n, n, 4);
    cout << endl;
}

void test_BunchKaufman_block(){
    int n = 15;
    int b_size = 5;
    double* A = (double*)malloc(n*n*sizeof(double));
    double* a = (double*)malloc(b_size*b_size*sizeof(double));

    double* L = (double*)malloc(n*n*sizeof(double));
    int* P = (int*)malloc(n*sizeof(int));
    int* pivot = (int*)malloc(n*sizeof(int));


    generate_random_symmetry(A,n);
    cout << "matrix A:\n";
    print_matrix(A,n,n,4);
    cout << endl;

    BunchKaufman_block(A,L,P,pivot,n,b_size);

    cout << "matrix D:\n";
    print_matrix(A,n,n,4);
    cout << endl;

    cout << "L:\n";
    print_matrix(L,n,n,4);
    cout << endl;

    cout << "P:\n";
    print_vector(P,n);
    cout << endl;

    cout << "pivot:\n";
    print_vector(pivot,n);
    cout << endl;


    free(A);free(L);free(P),free(pivot);
    free(a);

}

int main(){
    // test_BunchKaufman_subblock();
    // test_permute();
    test_BunchKaufman_block();
    return 0;
}
