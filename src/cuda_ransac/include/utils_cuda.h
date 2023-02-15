#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;
using namespace Eigen;

#define NEAR_ZERO_THRESHOLD 1e-14

__device__ void print_matrix(int m, int n, double** A);

__device__ void print_vector(int m, double* a);

__device__ void malloc_matrix(int m, int n, double*** A);

__device__ void malloc_vector(int m, double** a);

__device__ void memcopy_matrix(int m, int n, double** A, double* A_vector);

__device__ void memcopy_vector(int m, double** a, double* a_vector);

__device__ void mult_matrix(int m, int n, int nx, double** A, double** B, double** C);

__device__ void free_matrix(int m, int n, double** A);

__device__ void free_vector(int m, double* a);

enum AC_TYPE {
    GENERIC_CONSTRAINT_FULL,
    CASE5_CONSTRAINT_FULL,
    INTER_CAM_CONSTRAINT_FULL,
    INTRA_CAM_CONSTRAINT_FULL,
    GENERIC_CONSTRAINT_PARTIAL,
    CASE5_CONSTRAINT_PARTIAL,
    INTER_CAM_CONSTRAINT_PARTIAL,
    INTRA_CAM_CONSTRAINT_PARTIAL
};

struct UnitSample {
    // input
    bool is_known_angle;
    AC_TYPE actype;

    double *extrinsic_R_camera;
    double *extrinsic_T_camera;

    double *affine_tran;
    double *Image_1;
    double *Image_2;

    // output
    int num_sols;
    double *q_real_sols;
    double *t_real_sols;
    double *rotm_real_sols;
    // double* zr;
    // double* zi;
    // double *coeffs, *input;

    double *input;
    double *C0;
	double *C1;
    double *C12;
    double *RR;
    double *AM;
};

__device__ void format_convert_generic(
    double* input_Image_1, double* input_Image_2, double* input_affine_tran, 
    double* extrinsic_R_camera, double* extrinsic_T_camera, 
    Eigen::Vector3d Image1[2], Eigen::Vector3d Image2[2], Eigen::Matrix3d Ac[2],
    Eigen::Matrix3d R_camera[4], Eigen::Vector3d T_camera[4]);

__device__ void format_convert(
    double* input_Image_1, double* input_Image_2, double* input_affine_tran, 
    double* extrinsic_R_camera, double* extrinsic_T_camera, 
    Eigen::Vector3d Image1[2], Eigen::Vector3d Image2[2], Eigen::Matrix3d Ac[2],
    Eigen::Matrix3d R_camera[2], Eigen::Vector3d T_camera[2]);

__device__ void var3_order2_two_multiplication(double a_arr[10], double b_arr[10], double c_arr[35]);

__device__ void var3_order4_multiplication(double a_arr[35], double b_arr[35], double c_arr[165]);

__device__ void var3_order4_var3_order2_multiplication(double a_arr[35], double b_arr[10], double c_arr[84]);

__device__ void var3_order2_three_multiplication(double a_arr[10], double b_arr[10], double c_arr[10], double d_arr[10]);

__device__ void var3_order2_four_multiplication(double a_arr[10], double b_arr[10], double c_arr[10], double d_arr[10], double e_arr[165]);

__device__ void cayley2rotm(Eigen::Matrix<double,3,3>* rotm, Eigen::Matrix<double,3,1> q);