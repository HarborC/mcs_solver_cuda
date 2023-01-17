#pragma once

#include "utils_cuda.h"

__device__ int find_first_nonzero(Eigen::Matrix<double,1,165>& a);

__device__ int find_column(Eigen::Matrix<int,3,165>& a, Eigen::Matrix<int,3,1>& b);

__device__ int find_column(Eigen::Matrix<int,3,84>& a, Eigen::Matrix<int,3,1>& b);

__device__ void quot_var3_order8_by_x2y2z2_1(Eigen::Matrix<double,1,165>& c, Eigen::Matrix<double,1,84>& c_quot);

__device__ void construct_M(
    Eigen::Matrix<double,1,10>** M, 
    Eigen::Matrix3d*R_camera, Eigen::Vector3d*T_camera, 
    Eigen::Matrix3d*Ac, Eigen::Vector3d*Image1, Eigen::Vector3d*Image2,
    AC_TYPE actype, bool is_known_angle);

__device__ void create_coeffs(double* coeffs, double* input, 
    Eigen::Matrix<double,1,10>** M,
    double* input_Image_1, double* input_Image_2, double* input_affine_tran,
    double* extrinsic_R_camera, double* extrinsic_T_camera, AC_TYPE actype, 
    bool is_known_angle);
    
__device__ void calculate_M_6_by_4(Eigen::Matrix<double, 6, 4>& M_double, Eigen::Matrix<double,1,10>** M, double x, double y, double z);

__device__ void calculate_translation(Eigen::MatrixXd sols, Eigen::Matrix<double,1,10>** M, 
    Eigen::Matrix<double,3,1>* q_arr, Eigen::Matrix<double,3,1>* t_arr, bool is_known_angle);