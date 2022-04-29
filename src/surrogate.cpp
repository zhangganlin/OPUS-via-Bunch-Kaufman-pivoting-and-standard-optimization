#include "surrogate.hpp"
using namespace std;

void get_eigen_matrix( double* mat_d, Eigen::MatrixXd& mat_e, int m, int n){
    mat_e.setZero(m, n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            mat_e(i, j) = mat_d[i * n + j];
        }
    }
}

void get_eigen_vector( double* vec_d, Eigen::VectorXd& vec_e, int n){
    vec_e.setZero(n);
    for(int i = 0; i < n; i++){
        vec_e(i) = vec_d[i];
    }
}

void get_eigen_vector( double* vec_d, Eigen::RowVectorXd& vec_e, int n){
    vec_e.setZero(n);
    for(int i = 0; i < n; i++){
        vec_e(i) = vec_d[i];
    }
}

void get_double_vector(double* vec_d,  Eigen::VectorXd& vec_e, int n){
    for(int i = 0; i < n; i++){
        vec_d[i] = vec_e(i);
    }
}

void get_1d_mat(double** mat2D, double* mat1D, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            mat1D[i * n + j] = mat2D[i][j];
        }
    }
}

void build_surrogate( Eigen::MatrixXd& points,  Eigen::VectorXd& f, Eigen::VectorXd& lambda_c){
    // points           : (N x d matrix) represents particles
    // f                : (N x 1 vector) represents the corresponding value
    // return lambda_c  : ((N  + d + 1) x 1 vector
    // optimization: only compute upper part of phi, sparsity (zeros) 
    int N = points.rows(), d = points.cols();
    Eigen::MatrixXd A(N + d + 1, N + d + 1);
    Eigen::MatrixXd phi(N, N);
    Eigen::MatrixXd zeros; zeros.setZero(d + 1, d + 1);
    Eigen::VectorXd zeros_vec; zeros_vec.setZero(d + 1);
    Eigen::MatrixXd poly(N, d + 1);
    Eigen::VectorXd ones = Eigen::VectorXd::Ones(N);
    poly << points, ones;
    for (int s = 0; s < N; s++){
        phi.col(s) = (points.rowwise() - points.row(s)).matrix().rowwise().norm();
        phi.col(s) = phi.col(s).cwiseProduct(phi.col(s).cwiseProduct(phi.col(s)));
    } 
    A <<    phi             , poly,
            poly.adjoint()  , zeros;
    Eigen::VectorXd b(N + d + 1);
    b << f, zeros_vec;
    lambda_c = A.colPivHouseholderQr().solve(b);
}

void build_surrogate_eigen( double* points,  double* f, int N, int d, double* lambda_c){
    Eigen::MatrixXd points_e;
    Eigen::VectorXd f_e, lambda_c_e;
    get_eigen_matrix(points, points_e, N, d);
    get_eigen_vector(f, f_e, N);
    build_surrogate(points_e, f_e, lambda_c_e);
    get_double_vector(lambda_c, lambda_c_e, N + d + 1);
}

void build_surrogate_eigen(double** points,  double* f, int N, int d, double* lambda_c){
    double* points_1d = (double*)malloc((N*d) * sizeof(double));
    get_1d_mat(points, points_1d, N, d);
    build_surrogate_eigen(points_1d, f, N, d, lambda_c);
    free(points_1d);
}

double evaluate_surrogate( double* x, double* points,  double* lambda_c, int N, int d){
    double phi, error, res = 0;
    for(int i = 0; i < N; i++){
        phi = 0;
        for(int j = 0; j < d; j++){
            error = x[j] - points[i * d + j];
            phi += error * error;
        }
        phi = sqrt(phi);
        phi = phi * phi * phi;
        // optimize: phi = sqrt(phi) * phi;
        res += phi * lambda_c[i];
    }
    for(int i = 0; i < d; i++){
        res += x[i] * lambda_c[N + i];
    }
    res += lambda_c[N + d];
    return res;
}

double evaluate_surrogate( double* x, double** points,  double* lambda_c, int N, int d){
    double* points_1d = (double*)malloc((N*d) * sizeof(double));
    get_1d_mat(points, points_1d, N, d);
    double res = evaluate_surrogate(x, points_1d, lambda_c, N, d);
    free(points_1d);
    return res;
}