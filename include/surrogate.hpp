#ifndef SURROGATE_H_
#define SURROGATE_H_

#include <iostream>
#include <Eigen/Dense>

void get_eigen_matrix(const double* mat_d, Eigen::MatrixXd& mat_e, int m, int n);

void get_eigen_vector(const double* vec_d, Eigen::VectorXd& vec_e, int n);

void get_eigen_vector(const double* vec_d, Eigen::RowVectorXd& vec_e, int n);

void get_double_vector(double* vec_d, const Eigen::VectorXd& vec_e, int n);

void build_surrogate(const Eigen::MatrixXd& points, const Eigen::VectorXd& f, Eigen::VectorXd& lambda_c);

void build_surrogate_eigen(const double* points, const double* f, int N, int d, double* lambda_c);

double evaluate_surrogate(const double* x, const double* points, const double* lambda_c, int N, int d);

#endif