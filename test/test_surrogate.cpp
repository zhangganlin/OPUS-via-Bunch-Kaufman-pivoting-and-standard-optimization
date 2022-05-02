
#include "surrogate.hpp"
#include "opus.h"
#include "tsc_x86.h"
using namespace std;

int main(){
    int N = 3, d = 2;
    double** points = opus_matrix_new(N, d);
    points[0][0] = 1; points[0][1] = 1;
    points[1][0] = 4; points[1][1] = 5;
    points[2][0] = 3; points[2][1] = 3;
    double f[3] = {1, 2, 3};
    double* lambda_c = (double*)malloc((N + d + 1) * sizeof(double));
    
    // build_surrogate_eigen(points, f, 3, 2, lambda_c);
    int num_runs = (1 << 10);
    myInt64 start, cycles;
    start = start_tsc();
    for(int i = 0; i < num_runs; i++){
        build_surrogate(points, f, 3, 2, lambda_c);
    }
    cycles = stop_tsc(start) / num_runs;
    cout << "Cycles for build surrogate: " << cycles << endl;
    
    double x[2] = {3, 3};
    start = start_tsc();
    for(int i = 0; i < num_runs; i++){
        evaluate_surrogate(x, points, lambda_c, N, d);
    }
    cycles = stop_tsc(start) / num_runs;
    cout << "Cycles for evaluate surrogate: " << cycles << endl;

    return 0;
}