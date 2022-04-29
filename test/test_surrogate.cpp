
#include "surrogate.hpp"
#include "opus.h"
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
    build_surrogate(points, f, 3, 2, lambda_c);
    double x[2] = {3, 3};
    cout << evaluate_surrogate(x, points, lambda_c, N, d) << endl;

    return 0;
}