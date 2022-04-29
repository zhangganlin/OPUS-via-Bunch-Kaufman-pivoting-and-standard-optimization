#ifndef SURROGATE_H_
#define SURROGATE_H_

#include <iostream>
#include <Eigen/Dense>

void get_eigen_matrix( double* mat_d, Eigen::MatrixXd& mat_e, int m, int n);

void get_eigen_vector( double* vec_d, Eigen::VectorXd& vec_e, int n);

void get_eigen_vector( double* vec_d, Eigen::RowVectorXd& vec_e, int n);

void get_double_vector(double* vec_d,  Eigen::VectorXd& vec_e, int n);

void get_1d_mat(double** mat2D, double* mat1D, int m, int n);

void build_surrogate( Eigen::MatrixXd& points,  Eigen::VectorXd& f, Eigen::VectorXd& lambda_c);

void build_surrogate_eigen( double* points,  double* f, int N, int d, double* lambda_c);

void build_surrogate_eigen(double** points,  double* f, int N, int d, double* lambda_c);

void build_surrogate(double* points, double* f, int N, int d, double* lambda_c);

void build_surrogate(double** points, double* f, int N, int d, double* lambda_c);

double evaluate_surrogate( double* x,  double* points,  double* lambda_c, int N, int d);

double evaluate_surrogate( double* x, double** points,  double* lambda_c, int N, int d);

#endif