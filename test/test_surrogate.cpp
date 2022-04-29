
#include "surrogate.hpp"

using namespace std;

int main(){
    double points[6] = {1, 1, 4, 5, 3, 3};
    double f[3] = {1, 2, 3};
    int N = 3, d = 2;
    double* lambda_c = (double*)malloc((N + d + 1) * sizeof(double));;
    build_surrogate_eigen(points, f, 3, 2, lambda_c);
    double x[2] = {3, 3};
    cout << evaluate_surrogate(x, points, lambda_c, N, d) << endl;

    return 0;
}