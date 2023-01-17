// #include "mex.h"
#include <math.h>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "common.h"
#include "solver_trans_intra_2ac_core_48.h"
#include <stdio.h> 
#include "tic_toc.h"

using namespace std;
using namespace Eigen;

Eigen::MatrixXcd solve_equation_intra_cam_48(double *input, double* zr, double* zi)
{
    const VectorXd data = Map<const VectorXd>(input, 1315); // 1315 = 83*15+35*2
//    std::cout << data << std::endl;
    Eigen::MatrixXcd sols = solver_trans_intra_2ac_core_48(data);
    for (Index i = 0; i < sols.size(); i++) {
        zr[i] = sols(i).real();
        zi[i] = sols(i).imag();
    }
    return sols;
}

int main() { 
    int N = 10000;
    bool is_known_angle = false;
    AC_TYPE actype = AC_TYPE::INTRA_CAM_CONSTRAINT_FULL;

    double Image_1[6] = {0.019586866763680, 0.022571478986646, 1.0, 0.004320239058633, -0.417972219879661, 1.0};
    double Image_2[6] = {0.162872852683724, 0.107379974058292, 1.0, 0.216245541693002, -0.301145093383146, 1.0};
    double affine_tran[8] = {0.635587755262690, 0.319996888578032, -0.016240019308144, 0.996961229951916, 
                             0.971100136353982, -0.067216470192788, 0.085984638224679, 0.758593343727314};
    double extrinsic_R_camera[18] = {0.996016426352243, -0.017449748351250, -0.087445895952788, 
                                     0.018906840540077, 0.999695413509548, 0.015862269344775, 
                                     0.087142468505889, -0.017452406437284, 0.996042972814049, 
                                     0.989749591770535, 0.034894181340114, 0.138485167793029, 
                                     -0.032132431300164, 0.999238614955483, -0.022129103926483, 
                                     -0.139151904222689, 0.017452406437284, 0.990117246118230};
    double extrinsic_T_camera[6] = {0.537667139546100, 1.833885014595086, -2.258846861003648, 
                                    0.862173320368121, 0.318765239858981, -1.307688296305273};

    TicToc t_time;

    for (int id = 0; id < N; id++) {

    double *coeffs, *input;
    input = new double[1315];
    double* zr = new double[3*48];
    double* zi = new double[3*48];
    coeffs = new double[17*84];
    std::vector<std::vector<Eigen::Matrix<double,1,10>>> M;
    create_coeffs(coeffs, input, M, Image_1, Image_2, affine_tran, extrinsic_R_camera, extrinsic_T_camera, actype, is_known_angle);

    Eigen::MatrixXcd sols;
    sols = solve_equation_intra_cam_48(input, zr, zi);

    std::vector<Eigen::Matrix<double,3,1>> q_arr, t_arr;
    std::vector<Eigen::Matrix<double,3,3>> rotm;
    calculate_translation(sols, M, q_arr, t_arr, is_known_angle);
    cayley2rotm(rotm, q_arr);

    int n_real_sol = q_arr.size();
    double *q_real_sols = new double[3*n_real_sol];
    double *t_real_sols = new double[3*n_real_sol];
    double *rotm_real_sols = new double[9*n_real_sol];
    if (n_real_sol > 0) {
        for (int i = 0; i < n_real_sol; i++) {
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
