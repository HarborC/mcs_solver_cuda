// #include "mex.h"
#include <math.h>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "common.h"
#include "solver_trans_inter_2ac_core_48.h"
#include "solver_trans_inter_2ac_core_56.h"
#include "tic_toc.h"

using namespace std;
using namespace Eigen;

Eigen::MatrixXcd solve_equation_inter_cam_48(double *input, double* zr, double* zi)
{
    const VectorXd data = Map<const VectorXd>(input, 1330); 
//    std::cout << data << std::endl;
    Eigen::MatrixXcd sols = solver_trans_inter_2ac_core_48(data);
    for (Index i = 0; i < sols.size(); i++) {
        zr[i] = sols(i).real();
        zi[i] = sols(i).imag();
    }
    return sols;
}

Eigen::MatrixXcd solve_equation_inter_cam_56(double *input, double* zr, double* zi)
{
    const VectorXd data = Map<const VectorXd>(input, 1260); 
//    std::cout << data << std::endl;
    Eigen::MatrixXcd sols = solver_trans_inter_2ac_core_56(data);
    for (Index i = 0; i < sols.size(); i++) {
        zr[i] = sols(i).real();
        zi[i] = sols(i).imag();
    }
    return sols;
}

int main() 
{
    int N = 10000;
    bool is_known_angle = false;
    AC_TYPE actype = AC_TYPE::INTER_CAM_CONSTRAINT_FULL;
    double method = 0; //1
    if (method < -NEAR_ZERO_THRESHOLD || method > NEAR_ZERO_THRESHOLD)
    {
        actype = AC_TYPE::INTER_CAM_CONSTRAINT_PARTIAL;
    }

    double Image_1[6] = {0.122545792640241, -0.164658870899577, 1.0, 0.316273893339507, 0.474400031140436, 1.0};
    double Image_2[6] = {0.526136740639196, -0.373014056371327, 1.0, 0.402936480999619, 0.190852191243523, 1.0};
    double affine_tran[8] = {-38.495504397623016, 47.230200849740783, -19.236729681369706, 23.767566497845667, 
                             0.821017838844720, 0.085866053444795, 0.317754170634612, 0.512866528000997};
    double extrinsic_R_camera[18] = {0.996194698091746, 0, -0.087155742747658, 
                                     0, 1.0, 0, 
                                     0.087155742747658, 0, 0.996194698091746, 
                                     0.996194698091746, 0, 0.087155742747658, 
                                     0, 1.0, 0, 
                                     -0.087155742747658, 0, 0.996194698091746};
    double extrinsic_T_camera[6] = {-0.3, 0, 0, 0.5, 0, 0};

    TicToc t_time;

    for (int id = 0; id < N; id++) {

    double *coeffs, *input, *zr, *zi;
    if (actype == AC_TYPE::INTER_CAM_CONSTRAINT_PARTIAL)
    {
        input = new double[84*15];
        zr = new double[3*56];
        zi = new double[3*56];
        coeffs = new double[15*84];
    }
    else
    {
        input = new double[1330];
        zr = new double[3*48];
        zi = new double[3*48];
        coeffs = new double[17*84];
    }

    std::vector<std::vector<Eigen::Matrix<double,1,10>>> M;
    create_coeffs(coeffs, input, M, Image_1, Image_2, affine_tran, extrinsic_R_camera, extrinsic_T_camera, actype, is_known_angle);
    Eigen::MatrixXcd sols;
    if (actype == AC_TYPE::INTER_CAM_CONSTRAINT_PARTIAL)
    {
        sols = solve_equation_inter_cam_56(input, zr, zi);
    }
    else
    {
        sols = solve_equation_inter_cam_48(input, zr, zi);
    }

    std::vector<Eigen::Matrix<double,3,1>> q_arr, t_arr;
    std::vector<Eigen::Matrix<double,3,3>> rotm;
    calculate_translation(sols, M, q_arr, t_arr, is_known_angle);
    cayley2rotm(rotm, q_arr);

    int n_real_sol = q_arr.size();
    double *q_real_sols = new double[3*n_real_sol];
    double *t_real_sols = new double[3*n_real_sol];
    double *rotm_real_sols = new double[9*n_real_sol];
    if (n_real_sol > 0)
    {
        for (int i = 0; i < n_real_sol; i++)
        {
            Eigen::Matrix<double,3,1> q = q_arr[i];
            Eigen::Matrix<double,3,1> t = t_arr[i];
            Eigen::Matrix<double,3,3> r = rotm[i];

            // printf("q%d: %lf, %lf, %lf\n", i, q(0), q(1), q(2));
            // printf("t%d: %lf, %lf, %lf\n", i, t(0), t(1), t(2));

            q_real_sols[i*3] = q(0);
            q_real_sols[i*3+1] = q(1);
            q_real_sols[i*3+2] = q(2);
            t_real_sols[i*3] = t(0);
            t_real_sols[i*3+1] = t(1);
            t_real_sols[i*3+2] = t(2);

            rotm_real_sols[i*9] = r(0,0);
            rotm_real_sols[i*9+1] = r(1,0);
            rotm_real_sols[i*9+2] = r(2,0);
            rotm_real_sols[i*9+3] = r(0,1);
            rotm_real_sols[i*9+4] = r(1,1);
            rotm_real_sols[i*9+5] = r(2,1);
            rotm_real_sols[i*9+6] = r(0,2);
            rotm_real_sols[i*9+7] = r(1,2);
            rotm_real_sols[i*9+8] = r(2,2);
        }
    }

    delete [] input;
    delete [] q_real_sols;
    delete [] t_real_sols;
    delete [] rotm_real_sols;
    delete [] coeffs;
    delete [] zr;
    delete [] zi;

    }

    std::cout << "time : " << t_time.toc() << std::endl;

    return 0;
}
