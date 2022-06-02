#include "test_utils.h"
#include <iostream>
#include "blockbk.h"
#include "tsc_x86.h"

using namespace std;

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

void test_BunchKaufman_block(int n, int b_size, int repeat){

    double* A = (double*)malloc(n*n*sizeof(double));

    double* L = (double*)malloc(n*n*sizeof(double));
    int* P = (int*)malloc(n*sizeof(int));
    int* pivot = (int*)malloc(n*sizeof(int));


    generate_random_symmetry(A,n);

    for(int i = 0; i < repeat; i++){
        BunchKaufman_block(A,L,P,pivot,n,b_size);
    }

    free(A);free(L);free(P),free(pivot);

}

void test_BunchKaufman_noblock(int n, int b_size, int repeat){
    double* A = (double*)malloc(n*n*sizeof(double));
    double* L = (double*)malloc(n*n*sizeof(double));
    int* P = (int*)malloc(n*sizeof(int));
    int* pivot = (int*)malloc(n*sizeof(int));
    int* pivot_idx = (int*)malloc(n*sizeof(int));  
    generate_random_symmetry(A,n);

    for(int i = 0; i < repeat; i++){
        BunchKaufman_subblock(A,L,P,pivot,pivot_idx,n,0,n);
    }

    free(A);free(L);free(P),free(pivot);free(pivot_idx);
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
        BunchKaufman_subblock(A1,L,P,pivot,pivot_idx,n,0,n);
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
    free(A1);free(A2);free(L);free(P),free(pivot); free(pivot_idx);
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
        BunchKaufman_subblock(A1,L,P,pivot,pivot_idx,n,0,n);
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

int main(int argc, char *argv[]){
    // test_permute();
    int n = 2000;
    int repeat = 1;
    int b_size = 32;


    if(argc>1){
        b_size = atoi(argv[1]);
        cout << "b_size: " << b_size << endl;
    }
    // test_BunchKaufman_block(n,b_size,repeat);

    // for(int b_size = 20; b_size < 200; b_size += 1){
        // compare_speed(n,b_size,repeat);
    // }

    // find_best_block_size(n,repeat,15,120,1);

    srand(2);
    test_BunchKaufman_block(n,b_size,repeat);
    // test_BunchKaufman_noblock(n,b_size,repeat);


    return 0;
}
