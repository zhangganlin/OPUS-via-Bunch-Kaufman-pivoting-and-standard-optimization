#include "test_utils.h"
#include <iostream>
#include "blockbk.h"
#include "tsc_x86.h"

using namespace std;

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
    int* pivot2 = (int*)malloc(b_size*sizeof(int));

    generate_random_symmetry(A_block,b_size);

    matrix_set_block(A,12,12,b_size,b_size,A_block,n);

    cout << "A_block:\n";
    print_matrix(A_block,b_size,b_size,4);
    cout << "A:\n";
    print_matrix(A,n,n,4);

    BunchKaufman_noblock(A_block,L_block,P_block,pivot_block,b_size);
    BunchKaufman_subblock(A,L,P,pivot,pivot2,n,12,bigger_b_size);

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
    int b_size = 4;
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

void compare_speed(int n, int b_size, int repeat){
    srand(time(NULL));
    myInt64 start, gt_time, block_time;
    gt_time = block_time = 0;
    double* A = (double*)malloc(n*n*sizeof(double));
    double* A1 = (double*)malloc(n*n*sizeof(double));
    double* A2 = (double*)malloc(n*n*sizeof(double));
    double* L = (double*)malloc(n*n*sizeof(double));
    int* P = (int*)malloc(n*sizeof(int));
    int* pivot = (int*)malloc(n*sizeof(int));   
    int* pivot_idx = (int*)malloc(n*sizeof(int));       

    generate_random_symmetry(A,n);

    for(int i = 0; i < repeat; i++){
        
        matrix_transpose(A,A1,n);
        matrix_transpose(A,A2,n);

        start = start_tsc();
        BunchKaufman_noblock(A1,L,P,pivot,n);
        gt_time += stop_tsc(start);

        start = start_tsc();
        BunchKaufman_block(A2,L,P,pivot,n,b_size);
        block_time += stop_tsc(start);   
    }
    
 

    cout << "block size: " << b_size << endl;
    cout << "gt_time: " << (double)gt_time/(double)repeat << endl;
    cout << "block_time: "<<(double)block_time/(double)repeat << endl;
    cout << "speedup: " << (double)gt_time/(double)block_time << endl;
    cout << endl;

    free(A);
    free(A1);free(A2);free(L);free(P),free(pivot);
}

void find_best_block_size(int n, int repeat, int from, int to, int gap){
    srand(time(NULL));
    myInt64 start, gt_time, block_time;
    double* A = (double*)malloc(n*n*sizeof(double));
    double* A1 = (double*)malloc(n*n*sizeof(double));
    double* A2 = (double*)malloc(n*n*sizeof(double));
    double* L = (double*)malloc(n*n*sizeof(double));
    int* P = (int*)malloc(n*sizeof(int));
    int* pivot = (int*)malloc(n*sizeof(int));   
    int* pivot_idx = (int*)malloc(n*sizeof(int));       

    generate_random_symmetry(A,n);

    gt_time = 0;
    for(int i = 0; i < repeat; i++){
        matrix_transpose(A,A1,n);
        start = start_tsc();
        BunchKaufman_noblock(A1,L,P,pivot,n);
        gt_time += stop_tsc(start);
    }

    for(int b_size = from; b_size <= to; b_size+=gap){
        block_time = 0;
        for(int i = 0; i < repeat; i++){
            matrix_transpose(A,A2,n);
            start = start_tsc();
            BunchKaufman_block(A2,L,P,pivot,n,b_size);
            block_time += stop_tsc(start);      
        }
        cout << b_size << "," << (double)gt_time/(double)block_time << endl;
    }
 

    free(A);
    free(A1);free(A2);free(L);free(P),free(pivot);
}

int main(){
    // test_BunchKaufman_subblock();
    // test_permute();
    // test_BunchKaufman_block();
    int n = 100;
    int repeat = 5;
    int b_size = 8;
    // for(int b_size = 20; b_size < 200; b_size += 1){
        compare_speed(n,b_size,repeat);
    // }

    // find_best_block_size(n,repeat,65,200,1);

    return 0;
}
