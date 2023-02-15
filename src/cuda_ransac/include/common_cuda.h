#pragma once

#include "utils_cuda.h"

__device__ int find_first_nonzero(double a[165]);

__device__ int find_column495(double a[495], double b[3]);

__device__ int find_column252(double a[252], double b[3]);

__device__ void quot_var3_order8_by_x2y2z2_1(double c0[165], double c_quot[84]);

__device__ void construct_order6_poly(Eigen::Matrix<double,1,10> M[6][4], double C[1260], int *Sub_index, int N);

__device__ void construct_order6_extra_poly(Eigen::Matrix<double,1,10> M[6][4], double C[168], int *Sub_index_extra, int N);

__device__ void construct_M(
    Eigen::Matrix<double,1,10> M[6][4],
    Eigen::Matrix3d*R_camera, Eigen::Vector3d*T_camera, 
    Eigen::Matrix3d*Ac, Eigen::Vector3d*Image1, Eigen::Vector3d*Image2,
    AC_TYPE actype, bool is_known_angle);

__device__ void create_coeffs2(double* input, Eigen::Matrix<double,1,10> M[6][4], int *Sub_index, int Sub_index_N,
                               int *Sub_index_extra, int Sub_index_extra_N, AC_TYPE actype, bool is_known_angle);

__device__ void create_coeffs(double* input, Eigen::Matrix<double,1,10> M[6][4],
    double* input_Image_1, double* input_Image_2, double* input_affine_tran,
    double* extrinsic_R_camera, double* extrinsic_T_camera, AC_TYPE actype, 
    bool is_known_angle);
    
__device__ void calculate_M_6_by_4(Eigen::Matrix<double, 6, 4>& M_double, Eigen::Matrix<double,1,10> M[6][4], double x, double y, double z);

__device__ void calculate_translation(Eigen::MatrixXd sols, Eigen::Matrix<double,1,10> M[6][4], 
    Eigen::Matrix<double,3,1>* q_arr, Eigen::Matrix<double,3,1>* t_arr, bool is_known_angle);