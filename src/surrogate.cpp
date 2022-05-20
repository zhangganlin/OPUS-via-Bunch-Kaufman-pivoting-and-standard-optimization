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

void build_surrogate(double* points, double* f, int N, int d, double* lambda_c){
    double* A = (double*)malloc((N + d + 1)*(N + d + 1)*sizeof(double));
    double* b = (double*)malloc((N + d + 1)*sizeof(double));

    double phi, error;
    memset(A, 0, (N + d + 1) * (N + d + 1) * sizeof(double));
    memset(b, 0, (N + d + 1) * sizeof(double));
    for(int i = 0; i < N; i++){
        for(int j = 0; j < d; j++){
            A[i * (N + d + 1) + N + j] = points[i * d + j];
            A[(N + j) * (N + d + 1) + i] = points[i * d + j];
        }
        A[i * (N + d + 1) + N + d] = 1;
        A[(N + d) * (N + d + 1) + i] = 1;
    }
    
    // flops: N * N * (3d + 3)
    for(int pa = 0; pa < N; pa++){
        for(int pb = 0; pb < N; pb++){
            phi = 0;
            // flops: 3d
            for(int j = 0; j < d; j++){
                error = points[pa * d + j] - points[pb * d + j];
                phi += error * error;
            }
            phi = sqrt(phi); //1
            phi = phi * phi * phi; //2
            A[pa * (N + d + 1) + pb] = phi;
        }
    }
    
    // optimized
    // for(int pa = 0; pa < N; pa++){
    //     A already set to zero
    //     A[pa * (N + d + 1) + pa] = 0;
    //     for(int pb = pa + 1; pb < N; pb++){
    //         phi = 0;
    //         for(int j = 0; j < d; j++){
    //             error = points[pa * d + j] - points[pb * d + j];
    //             phi += error * error;
    //         }
    //         phi = sqrt(phi);
    //         phi = phi * phi * phi;
    //         A[pa * (N + d + 1) + pb] = phi;
    //         A[pb * (N + d + 1) + pa] = phi;
    //     }
    // }

    for(int i = 0; i < N; i++) b[i] = f[i];
    
    // Eigen::MatrixXd A_e;
    // Eigen::VectorXd b_e, lambda_c_e;
    // get_eigen_matrix(A, A_e, N + d + 1, N + d + 1);
    // get_eigen_vector(b, b_e, N + d + 1);
    // lambda_c_e = A_e.lu().solve(b_e);
    // get_double_vector(lambda_c, lambda_c_e, N + d + 1);
    

    solve_BunchKaufman(A,lambda_c,b,N+d+1);
    /*for debug:-----------------------------------------------
    cout << "n:"<<N+d+1<<endl;
    for(int i = 0; i < N+d+1; i++){
        for(int j = 0; j < N+d+1; j++){
            printf("A[%d] = %lf; ",i*(N+d+1)+j,A[i*(N+d+1)+j]);
        }
        cout << endl;
    }

    for(int i =0; i < N+d+1;i++){
        printf("b[%d]=%lf; ",i,b[i]);
    }
    cout << endl;

    cout << "x should be:\n";
    for(int i = 0; i < N+d+1;i++){
        cout << lambda_c[i] << " ";
    }
    cout << endl;
    /----------------------------------------------------------*/

    free(A);
    free(b);
}

void build_surrogate(double** points, double* f, int N, int d, double* lambda_c){
    double* points_1d = (double*)malloc((N*d) * sizeof(double));
    get_1d_mat(points, points_1d, N, d);
    build_surrogate(points_1d, f, N, d, lambda_c);
    free(points_1d);
}

double evaluate_surrogate( double* x, double* points,  double* lambda_c, int N, int d){
    // total flops: 3Nd + 5N + 2d + 1
    double phi, error, res = 0, sq_phi;
    int id;
    // flops: 3Nd + 5N
    for(int i = 0; i < N; i++){
        phi = 0;
        // flops: 3d

        id = i * d;

        for(int j = 0; j < d; j++){
            error = x[j] - points[id + j];
            phi += error * error;
        }
        sq_phi = sqrt(phi);            // flops: 1
        phi = phi * sq_phi;      // flops: 2
        res += phi * lambda_c[i];   // flops: 2
    }
    // flops: 2d
    for(int i = 0; i < d; i++){
        res += x[i] * lambda_c[N + i];
    }
    // flops: 1
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