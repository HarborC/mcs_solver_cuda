#include "common_cuda.h"
#include <stdio.h> 
#include "qr_cuda.h"

__device__ int find_first_nonzero(Eigen::Matrix<double,1,165>& a)
{
    int idx = -1;
    for (int i = 0; i < 165; i++)
    {
        if (abs(a(i))>NEAR_ZERO_THRESHOLD)
        {
            idx = i;
            break;
        }
    }
    return idx;
}

__device__ int find_column(Eigen::Matrix<int,3,165>& a, Eigen::Matrix<int,3,1>& b)
{
    int idx = -1;
    for (int i = 0; i < 165; i++)
    {
        if (a(0,i)==b(0) && a(1,i)==b(1) && a(2,i)==b(2))
        {
            idx = i;
            break;
        }
    }
    return idx;
}

__device__ int find_column(Eigen::Matrix<int,3,84>& a, Eigen::Matrix<int,3,1>& b)
{
    int idx = -1;
    for (int i = 0; i < 84; i++)
    {
        if (a(0,i)==b(0) && a(1,i)==b(1) && a(2,i)==b(2))
        {
            idx = i;
            break;
        }
    }
    return idx;
}

__device__ void quot_var3_order8_by_x2y2z2_1(Eigen::Matrix<double,1,165>& c, Eigen::Matrix<double,1,84>& c_quot)
{
    Eigen::Matrix<double,1,165> c1 = c;
    c_quot.setZero();
    Eigen::Matrix<int,3,165> M1;
    Eigen::Matrix<int,3,4> M2;
    Eigen::Matrix<int,3,84> M_quot;
    M1 << 
        8, 7, 7, 7, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 1, 0, 0, 2, 1, 1, 0, 0, 0, 3, 2, 2, 1, 1, 1, 0, 0, 0, 0, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 5, 4, 4, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 6, 5, 5, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 7, 6, 6, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 8, 7, 7, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        0, 0, 1, 0, 0, 1, 0, 2, 1, 0, 0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0, 0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 0, 2, 1, 0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0, 8, 7, 6, 5, 4, 3, 2, 1, 0;
    M2 << 
        2, 0, 0, 0,
        0, 2, 0, 0,
        0, 0, 2, 0;
    M_quot <<
        6, 5, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 3, 2, 1, 0, 2, 1, 0, 1, 0, 0, 5, 4, 3, 2, 1, 0, 4, 3, 2, 1, 0, 3, 2, 1, 0, 2, 1, 0, 1, 0, 0, 4, 3, 2, 1, 0, 3, 2, 1, 0, 2, 1, 0, 1, 0, 0, 3, 2, 1, 0, 2, 1, 0, 1, 0, 0, 2, 1, 0, 1, 0, 0, 1, 0, 0, 0, 
        0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 0, 1, 0, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 0, 1, 0, 0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 0, 1, 0, 0, 1, 2, 3, 0, 1, 2, 0, 1, 0, 0, 1, 2, 0, 1, 0, 0, 1, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 0, 0, 0, 1, 1, 2, 0, 0, 1, 0;
    Eigen::Matrix<double,1,4> c2;
    c2 << 1, 1, 1, 1;
    
    Eigen::Matrix<int,3,1> LMid, m;
    Eigen::Matrix<int,3,1> LMd = M2.col(0);
    while(true)
    {
        int idx = find_first_nonzero(c1);
        if (idx==-1)
            break;

        LMid = M1.col(idx);
        bool divisible = false;

        m = LMid - LMd;
        if (m(0)>=0 && m(1)>=0 && m(2)>=0)
        {
            double c = c1(idx)/c2(0);
            int idx2 = find_column(M_quot, m);
            if (idx2==-1)
            {
                continue;
            }
            c_quot(idx2) = c_quot(idx2) + c;
            for (int k = 0; k < 4; k++)
            {
                Eigen::Matrix<int,3,1> m_tmp = m + M2.col(k);
                double c_tmp = c*c2(k);
                int idx3 = find_column(M1, m_tmp);
                if (idx3==-1)
                {
                    continue;
                }
                c1(idx3) =  c1(idx3) - c_tmp;
            }
            divisible = true;
        }
        if (!divisible)
        {
            int idx = find_first_nonzero(c1);
            if (idx==-1)
                break;
            c1(idx) = 0;
        }
    }
    return;
}

__device__ void construct_order6_poly(Eigen::Matrix<double,1,10>** M, 
    Eigen::Matrix<double,15,84>& C, Eigen::Matrix<double,15,165>& C8, Eigen::MatrixXi& Sub_index) {
    C.setZero();
    C8.setZero();

    Eigen::Matrix<int,15,4> Idx_all;
    Idx_all << 
     0, 1, 2, 3, 
     0, 1, 2, 4, 
     0, 1, 2, 5,
     0, 1, 3, 4,
     0, 1, 3, 5,
     0, 1, 4, 5,
     0, 2, 3, 4,
     0, 2, 3, 5,
     0, 2, 4, 5,
     0, 3, 4, 5,
     1, 2, 3, 4,
     1, 2, 3, 5,
     1, 2, 4, 5,
     1, 3, 4, 5,
     2, 3, 4, 5;

    int N = Sub_index.cols()*Sub_index.rows();
    
    for (int k = 0; k < N; k++)
    {
        int i = Sub_index(k);
        Eigen::Matrix<int,1,4> idx = Idx_all.row(i);

        Eigen::Matrix<double,1,10> 
            m11, m12, m13, m14, 
            m21, m22, m23, m24,
            m31, m32, m33, m34,
            m41, m42, m43, m44;

        m11 = M[idx(0)][0]; m12 = M[idx(0)][1]; m13 = M[idx(0)][2], m14 = M[idx(0)][3];
        m21 = M[idx(1)][0]; m22 = M[idx(1)][1]; m23 = M[idx(1)][2], m24 = M[idx(1)][3];
        m31 = M[idx(2)][0]; m32 = M[idx(2)][1]; m33 = M[idx(2)][2], m34 = M[idx(2)][3];
        m41 = M[idx(3)][0]; m42 = M[idx(3)][1]; m43 = M[idx(3)][2], m44 = M[idx(3)][3];

        Eigen::Matrix<double,1,165> d, d0;
        d.setZero();
        d0.setZero();
        var3_order2_four_multiplication(m11, m22, m33, m44, d0); d = d + d0;
        var3_order2_four_multiplication(m11, m22, m34, m43, d0); d = d - d0;
        var3_order2_four_multiplication(m11, m23, m32, m44, d0); d = d - d0;
        var3_order2_four_multiplication(m11, m23, m34, m42, d0); d = d + d0;
        var3_order2_four_multiplication(m11, m24, m32, m43, d0); d = d + d0;
        var3_order2_four_multiplication(m11, m24, m33, m42, d0); d = d - d0;
        var3_order2_four_multiplication(m12, m21, m33, m44, d0); d = d - d0;
        var3_order2_four_multiplication(m12, m21, m34, m43, d0); d = d + d0;
        var3_order2_four_multiplication(m12, m23, m31, m44, d0); d = d + d0;
        var3_order2_four_multiplication(m12, m23, m34, m41, d0); d = d - d0;
        var3_order2_four_multiplication(m12, m24, m31, m43, d0); d = d - d0;
        var3_order2_four_multiplication(m12, m24, m33, m41, d0); d = d + d0;
        var3_order2_four_multiplication(m13, m21, m32, m44, d0); d = d + d0;
        var3_order2_four_multiplication(m13, m21, m34, m42, d0); d = d - d0;
        var3_order2_four_multiplication(m13, m22, m31, m44, d0); d = d - d0;
        var3_order2_four_multiplication(m13, m22, m34, m41, d0); d = d + d0;
        var3_order2_four_multiplication(m13, m24, m31, m42, d0); d = d + d0;
        var3_order2_four_multiplication(m13, m24, m32, m41, d0); d = d - d0;
        var3_order2_four_multiplication(m14, m21, m32, m43, d0); d = d - d0;
        var3_order2_four_multiplication(m14, m21, m33, m42, d0); d = d + d0;
        var3_order2_four_multiplication(m14, m22, m31, m43, d0); d = d + d0;
        var3_order2_four_multiplication(m14, m22, m33, m41, d0); d = d - d0;
        var3_order2_four_multiplication(m14, m23, m31, m42, d0); d = d - d0;
        var3_order2_four_multiplication(m14, m23, m32, m41, d0); d = d + d0;

        Eigen::Matrix<double,1,84> quot;
        quot_var3_order8_by_x2y2z2_1(d, quot);
        C.block(i, 0, 1, 84) = quot;
        C8.block(i, 0, 1, 165) = d;
    }
    return;
}

__device__ void construct_order6_extra_poly(Eigen::Matrix<double,1,10>** M, 
    Eigen::Matrix<double,2,84>& C, Eigen::Matrix<double,2,165>& C8, Eigen::MatrixXi Sub_index_extra) {
    C.setZero();
    C8.setZero();

    Eigen::Matrix<int,2,3> Idx_all;
    Idx_all << 
         0, 1, 2,
         3, 4, 5;

    Eigen::Matrix<double,1,10> m_one;
    m_one.setZero(); m_one(9) = 1;

    int N = Sub_index_extra.cols()*Sub_index_extra.rows();
    for (int k = 0; k < N; k++)
    {
        int i = Sub_index_extra(k);
        Eigen::Matrix<int,1,3> idx = Idx_all.row(i);

        Eigen::Matrix<double,1,10> m11, m12, m13, 
                                   m21, m22, m23,
                                   m31, m32, m33,
                                   m44;

        m11 = M[idx(0)][0]; m12 = M[idx(0)][1]; m13 = M[idx(0)][2];
        m21 = M[idx(1)][0]; m22 = M[idx(1)][1]; m23 = M[idx(1)][2];
        m31 = M[idx(2)][0]; m32 = M[idx(2)][1]; m33 = M[idx(2)][2];
        m44 = m_one;

        Eigen::Matrix<double,1,165> d, d0;
        d.setZero();
        d0.setZero();
        var3_order2_four_multiplication(m11, m22, m33, m44, d0); d = d + d0;
        var3_order2_four_multiplication(m11, m23, m32, m44, d0); d = d - d0;
        var3_order2_four_multiplication(m12, m21, m33, m44, d0); d = d - d0;
        var3_order2_four_multiplication(m12, m23, m31, m44, d0); d = d + d0;
        var3_order2_four_multiplication(m13, m21, m32, m44, d0); d = d + d0;
        var3_order2_four_multiplication(m13, m22, m31, m44, d0); d = d - d0;

        Eigen::Matrix<double,1,84> quot;
        quot_var3_order8_by_x2y2z2_1(d, quot);
        C.block(i, 0, 1, 84) = quot;
        C8.block(i, 0, 1, 165) = d;
    }
    return;
}

__device__ void construct_M(
    Eigen::Matrix<double,1,10>** M, 
    Eigen::Matrix3d* R_camera, Eigen::Vector3d* T_camera, 
    Eigen::Matrix3d* Ac, Eigen::Vector3d* Image1, Eigen::Vector3d* Image2,
    AC_TYPE actype, bool is_known_angle) {

    int point_num = 2;
    Eigen::Matrix<double,2,10> 
        f1_New_C11, f1_New_C12, f1_New_C13, f1_New_C14,
        f2_New_C21, f2_New_C22, f2_New_C23, f2_New_C24,
        f3_New_C31, f3_New_C32, f3_New_C33, f3_New_C34;
    for (int i = 0; i < point_num; i++)
    {
        Eigen::Vector3d P1 = Image1[i];
        Eigen::Vector3d P2 = Image2[i];
        Eigen::Vector3d U1 = P1;
        U1.normalize();
        Eigen::Vector3d U2 = P2;
        U2.normalize();

        int idx1 = 0;
        int idx2 = 0;
		if(actype == GENERIC_CONSTRAINT_FULL || actype == GENERIC_CONSTRAINT_PARTIAL)
        {
            if (i==0)
            {
                idx1 = 0;
                idx2 = 1;
            }
            else
            {
                idx1 = 2;
                idx2 = 3;
            }
        }
        else if (actype == CASE5_CONSTRAINT_FULL || actype == CASE5_CONSTRAINT_PARTIAL)
        {
            if (i==0)
            {
                idx1 = 0;
                idx2 = 1;
            }
            else
            {
                idx1 = 1;
                idx2 = 1;
            }
        }
        else if (actype == INTER_CAM_CONSTRAINT_FULL || actype == INTER_CAM_CONSTRAINT_PARTIAL)
        {
            if (i==0)
            {
                idx1 = 0;
                idx2 = 1;
            }
            else
            {
                idx1 = 1;
                idx2 = 0;
            }
        }
        else if (actype == INTRA_CAM_CONSTRAINT_FULL || actype == INTRA_CAM_CONSTRAINT_PARTIAL)
        {
            if (i==0)
            {
                idx1 = 0;
                idx2 = 0;
            }
            else
            {
                idx1 = 1;
                idx2 = 1;
            }
        }
        else
        {
            return;
        }
        
        Eigen::Matrix3d R1 = R_camera[idx1];
        Eigen::Vector3d T1 = T_camera[idx1];
        Eigen::Matrix3d R2 = R_camera[idx2];
        Eigen::Vector3d T2 = T_camera[idx2];
        Eigen::Matrix3d Atemp = Ac[i].transpose()*R2.transpose();
        
        Eigen::Matrix<double,6,1> Line_i, Line_j;
        Eigen::Vector3d V = R1*U1;
        Line_i.block(0, 0, 3, 1) = V;
        Line_i.block(3, 0, 3, 1) = T1.cross(V);
        V = R2*U2;
        Line_j.block(0, 0, 3, 1) = V;
        Line_j.block(3, 0, 3, 1) = T2.cross(V);

        double L11 = Line_i(0);
        double L12 = Line_i(1);
        double L13 = Line_i(2);
        double L14 = Line_i(3);
        double L15 = Line_i(4);
        double L16 = Line_i(5);

        double L21 = Line_j(0);
        double L22 = Line_j(1);
        double L23 = Line_j(2);
        double L24 = Line_j(3);
        double L25 = Line_j(4);
        double L26 = Line_j(5);

        double tx1 = T1(0);
        double ty1 = T1(1);
        double tz1 = T1(2);

        double tx2 = T2(0);
        double ty2 = T2(1);
        double tz2 = T2(2);

        double r1 = R1(0,0);
        double r2 = R1(0,1);
        double r3 = R1(0,2);
        double r4 = R1(1,0);
        double r5 = R1(1,1);
        double r6 = R1(1,2);
        double r7 = R1(2,0);
        double r8 = R1(2,1);
        double r9 = R1(2,2);

        double a1 = Atemp(0,0);
        double a2 = Atemp(0,1);
        double a3 = Atemp(0,2);
        double a4 = Atemp(1,0);
        double a5 = Atemp(1,1);
        double a6 = Atemp(1,2);

        Eigen::Vector3d P1_rotated = R1*P1;
        double p11 = P1_rotated(0);
        double p12 = P1_rotated(1);
        double p13 = P1_rotated(2);

        Eigen::Vector3d P2_rotated = R2*P2;
        double p21 = P2_rotated(0);
        double p22 = P2_rotated(1);
        double p23 = P2_rotated(2);

        f1_New_C11.block(i, 0, 1, 10) << L13*L22 - L12*L23, 2*L11*L23, -2*L11*L22, - 2*L12*L22 - 2*L13*L23, L12*L23 + L13*L22, 2*L13*L23 - 2*L12*L22, 2*L11*L22, - L12*L23 - L13*L22, 2*L11*L23, L12*L23 - L13*L22;
        f1_New_C12.block(i, 0, 1, 10) << - L11*L23 - L13*L21, -2*L12*L23, 2*L11*L21 - 2*L13*L23, 2*L12*L21, L11*L23 - L13*L21, 2*L12*L21, - 2*L11*L21 - 2*L13*L23, L11*L23 + L13*L21, 2*L12*L23, L13*L21 - L11*L23;
        f1_New_C13.block(i, 0, 1, 10) << L11*L22 + L12*L21, 2*L12*L22 - 2*L11*L21, 2*L13*L22, 2*L13*L21, - L11*L22 - L12*L21, -2*L13*L21, 2*L13*L22, L12*L21 - L11*L22, - 2*L11*L21 - 2*L12*L22, L11*L22 - L12*L21;
        f1_New_C14.block(i, 0, 1, 10) << L11*L24 + L14*L21 - L12*L25 - L15*L22 - L13*L26 - L16*L23, 2*L11*L25 + 2*L12*L24 + 2*L14*L22 + 2*L15*L21, 2*L11*L26 + 2*L13*L24 + 2*L14*L23 + 2*L16*L21, 2*L12*L26 - 2*L13*L25 + 2*L15*L23 - 2*L16*L22, L12*L25 - L14*L21 - L11*L24 + L15*L22 - L13*L26 - L16*L23, 2*L12*L26 + 2*L13*L25 + 2*L15*L23 + 2*L16*L22, 2*L13*L24 - 2*L11*L26 - 2*L14*L23 + 2*L16*L21, L13*L26 - L14*L21 - L12*L25 - L15*L22 - L11*L24 + L16*L23, 2*L11*L25 - 2*L12*L24 + 2*L14*L22 - 2*L15*L21, L11*L24 + L14*L21 + L12*L25 + L15*L22 + L13*L26 + L16*L23;
        f2_New_C21.block(i, 0, 1, 10) << a3*p12 - a2*p13 + p23*r4 - p22*r7, - 2*a3*p11 - 2*p23*r1, 2*a2*p11 + 2*p22*r1, 2*a2*p12 + 2*a3*p13 + 2*p22*r4 + 2*p23*r7, - a2*p13 - a3*p12 - p23*r4 - p22*r7, 2*a2*p12 - 2*a3*p13 + 2*p22*r4 - 2*p23*r7, - 2*a2*p11 - 2*p22*r1, a2*p13 + a3*p12 + p23*r4 + p22*r7, - 2*a3*p11 - 2*p23*r1, a2*p13 - a3*p12 - p23*r4 + p22*r7;
        f2_New_C22.block(i, 0, 1, 10) << a1*p13 + a3*p11 + p23*r1 + p21*r7, 2*a3*p12 + 2*p23*r4, 2*a3*p13 - 2*a1*p11 - 2*p21*r1 + 2*p23*r7, - 2*a1*p12 - 2*p21*r4, a1*p13 - a3*p11 - p23*r1 + p21*r7, - 2*a1*p12 - 2*p21*r4, 2*a1*p11 + 2*a3*p13 + 2*p21*r1 + 2*p23*r7, - a1*p13 - a3*p11 - p23*r1 - p21*r7, - 2*a3*p12 - 2*p23*r4, a3*p11 - a1*p13 + p23*r1 - p21*r7;
        f2_New_C23.block(i, 0, 1, 10) << - a1*p12 - a2*p11 - p22*r1 - p21*r4, 2*a1*p11 - 2*a2*p12 + 2*p21*r1 - 2*p22*r4, - 2*a2*p13 - 2*p22*r7, - 2*a1*p13 - 2*p21*r7, a1*p12 + a2*p11 + p22*r1 + p21*r4, 2*a1*p13 + 2*p21*r7, - 2*a2*p13 - 2*p22*r7, a2*p11 - a1*p12 + p22*r1 - p21*r4, 2*a1*p11 + 2*a2*p12 + 2*p21*r1 + 2*p22*r4, a1*p12 - a2*p11 - p22*r1 + p21*r4;
        f2_New_C24.block(i, 0, 1, 10) << a3*p12*tx1 - a2*p13*tx1 + a2*p13*tx2 - a3*p12*tx2 - a1*p13*ty1 - a3*p11*ty1 - a1*p13*ty2 - a3*p11*ty2 + a1*p12*tz1 + a2*p11*tz1 + a1*p12*tz2 + a2*p11*tz2 + p23*r4*tx1 - p23*r4*tx2 - p22*r7*tx1 + p22*r7*tx2 - p23*r1*ty1 - p23*r1*ty2 - p21*r7*ty1 - p21*r7*ty2 + p22*r1*tz1 + p22*r1*tz2 + p21*r4*tz1 + p21*r4*tz2, 2*a1*p13*tx1 + 2*a3*p11*tx2 - 2*a2*p13*ty1 - 2*a3*p12*ty2 - 2*a1*p11*tz1 - 2*a1*p11*tz2 + 2*a2*p12*tz1 + 2*a2*p12*tz2 + 2*p23*r1*tx2 + 2*p21*r7*tx1 - 2*p23*r4*ty2 - 2*p22*r7*ty1 - 2*p21*r1*tz1 - 2*p21*r1*tz2 + 2*p22*r4*tz1 + 2*p22*r4*tz2, 2*a1*p11*ty1 - 2*a2*p11*tx2 - 2*a1*p12*tx1 + 2*a1*p11*ty2 - 2*a3*p13*ty1 - 2*a3*p13*ty2 + 2*a3*p12*tz1 + 2*a2*p13*tz2 - 2*p22*r1*tx2 - 2*p21*r4*tx1 + 2*p21*r1*ty1 + 2*p21*r1*ty2 - 2*p23*r7*ty1 - 2*p23*r7*ty2 + 2*p23*r4*tz1 + 2*p22*r7*tz2, 2*a2*p12*tx1 - 2*a2*p12*tx2 + 2*a3*p13*tx1 - 2*a3*p13*tx2 - 2*a2*p11*ty1 + 2*a1*p12*ty2 - 2*a3*p11*tz1 + 2*a1*p13*tz2 + 2*p22*r4*tx1 - 2*p22*r4*tx2 + 2*p23*r7*tx1 - 2*p23*r7*tx2 - 2*p22*r1*ty1 + 2*p21*r4*ty2 - 2*p23*r1*tz1 + 2*p21*r7*tz2, a2*p13*tx1 + a3*p12*tx1 + a2*p13*tx2 + a3*p12*tx2 + a1*p13*ty1 - a3*p11*ty1 - a1*p13*ty2 + a3*p11*ty2 - a1*p12*tz1 - a2*p11*tz1 - a1*p12*tz2 - a2*p11*tz2 + p23*r4*tx1 + p23*r4*tx2 + p22*r7*tx1 + p22*r7*tx2 - p23*r1*ty1 + p23*r1*ty2 + p21*r7*ty1 - p21*r7*ty2 - p22*r1*tz1 - p22*r1*tz2 - p21*r4*tz1 - p21*r4*tz2, 2*a3*p13*tx1 - 2*a2*p12*tx2 - 2*a2*p12*tx1 + 2*a3*p13*tx2 + 2*a2*p11*ty1 + 2*a1*p12*ty2 - 2*a3*p11*tz1 - 2*a1*p13*tz2 - 2*p22*r4*tx1 - 2*p22*r4*tx2 + 2*p23*r7*tx1 + 2*p23*r7*tx2 + 2*p22*r1*ty1 + 2*p21*r4*ty2 - 2*p23*r1*tz1 - 2*p21*r7*tz2, 2*a2*p11*tx2 - 2*a1*p12*tx1 + 2*a1*p11*ty1 - 2*a1*p11*ty2 + 2*a3*p13*ty1 - 2*a3*p13*ty2 - 2*a3*p12*tz1 + 2*a2*p13*tz2 + 2*p22*r1*tx2 - 2*p21*r4*tx1 + 2*p21*r1*ty1 - 2*p21*r1*ty2 + 2*p23*r7*ty1 - 2*p23*r7*ty2 - 2*p23*r4*tz1 + 2*p22*r7*tz2, a1*p13*ty1 - a3*p12*tx1 - a2*p13*tx2 - a3*p12*tx2 - a2*p13*tx1 + a3*p11*ty1 + a1*p13*ty2 + a3*p11*ty2 - a1*p12*tz1 + a2*p11*tz1 + a1*p12*tz2 - a2*p11*tz2 - p23*r4*tx1 - p23*r4*tx2 - p22*r7*tx1 - p22*r7*tx2 + p23*r1*ty1 + p23*r1*ty2 + p21*r7*ty1 + p21*r7*ty2 + p22*r1*tz1 - p22*r1*tz2 - p21*r4*tz1 + p21*r4*tz2, 2*a3*p11*tx2 - 2*a1*p13*tx1 - 2*a2*p13*ty1 + 2*a3*p12*ty2 + 2*a1*p11*tz1 - 2*a1*p11*tz2 + 2*a2*p12*tz1 - 2*a2*p12*tz2 + 2*p23*r1*tx2 - 2*p21*r7*tx1 + 2*p23*r4*ty2 - 2*p22*r7*ty1 + 2*p21*r1*tz1 - 2*p21*r1*tz2 + 2*p22*r4*tz1 - 2*p22*r4*tz2, a2*p13*tx1 - a3*p12*tx1 - a2*p13*tx2 + a3*p12*tx2 - a1*p13*ty1 + a3*p11*ty1 + a1*p13*ty2 - a3*p11*ty2 + a1*p12*tz1 - a2*p11*tz1 - a1*p12*tz2 + a2*p11*tz2 - p23*r4*tx1 + p23*r4*tx2 + p22*r7*tx1 - p22*r7*tx2 + p23*r1*ty1 - p23*r1*ty2 - p21*r7*ty1 + p21*r7*ty2 - p22*r1*tz1 + p22*r1*tz2 + p21*r4*tz1 - p21*r4*tz2;
        f3_New_C31.block(i, 0, 1, 10) << a6*p12 - a5*p13 + p23*r5 - p22*r8, - 2*a6*p11 - 2*p23*r2, 2*a5*p11 + 2*p22*r2, 2*a5*p12 + 2*a6*p13 + 2*p22*r5 + 2*p23*r8, - a5*p13 - a6*p12 - p23*r5 - p22*r8, 2*a5*p12 - 2*a6*p13 + 2*p22*r5 - 2*p23*r8, - 2*a5*p11 - 2*p22*r2, a5*p13 + a6*p12 + p23*r5 + p22*r8, - 2*a6*p11 - 2*p23*r2, a5*p13 - a6*p12 - p23*r5 + p22*r8;
        f3_New_C32.block(i, 0, 1, 10) << a4*p13 + a6*p11 + p23*r2 + p21*r8, 2*a6*p12 + 2*p23*r5, 2*a6*p13 - 2*a4*p11 - 2*p21*r2 + 2*p23*r8, - 2*a4*p12 - 2*p21*r5, a4*p13 - a6*p11 - p23*r2 + p21*r8, - 2*a4*p12 - 2*p21*r5, 2*a4*p11 + 2*a6*p13 + 2*p21*r2 + 2*p23*r8, - a4*p13 - a6*p11 - p23*r2 - p21*r8, - 2*a6*p12 - 2*p23*r5, a6*p11 - a4*p13 + p23*r2 - p21*r8;
        f3_New_C33.block(i, 0, 1, 10) << - a4*p12 - a5*p11 - p22*r2 - p21*r5, 2*a4*p11 - 2*a5*p12 + 2*p21*r2 - 2*p22*r5, - 2*a5*p13 - 2*p22*r8, - 2*a4*p13 - 2*p21*r8, a4*p12 + a5*p11 + p22*r2 + p21*r5, 2*a4*p13 + 2*p21*r8, - 2*a5*p13 - 2*p22*r8, a5*p11 - a4*p12 + p22*r2 - p21*r5, 2*a4*p11 + 2*a5*p12 + 2*p21*r2 + 2*p22*r5, a4*p12 - a5*p11 - p22*r2 + p21*r5;
        f3_New_C34.block(i, 0, 1, 10) << a6*p12*tx1 - a5*p13*tx1 + a5*p13*tx2 - a6*p12*tx2 - a4*p13*ty1 - a6*p11*ty1 - a4*p13*ty2 - a6*p11*ty2 + a4*p12*tz1 + a5*p11*tz1 + a4*p12*tz2 + a5*p11*tz2 + p23*r5*tx1 - p23*r5*tx2 - p22*r8*tx1 + p22*r8*tx2 - p23*r2*ty1 - p23*r2*ty2 - p21*r8*ty1 - p21*r8*ty2 + p22*r2*tz1 + p22*r2*tz2 + p21*r5*tz1 + p21*r5*tz2, 2*a4*p13*tx1 + 2*a6*p11*tx2 - 2*a5*p13*ty1 - 2*a6*p12*ty2 - 2*a4*p11*tz1 - 2*a4*p11*tz2 + 2*a5*p12*tz1 + 2*a5*p12*tz2 + 2*p23*r2*tx2 + 2*p21*r8*tx1 - 2*p23*r5*ty2 - 2*p22*r8*ty1 - 2*p21*r2*tz1 - 2*p21*r2*tz2 + 2*p22*r5*tz1 + 2*p22*r5*tz2, 2*a4*p11*ty1 - 2*a5*p11*tx2 - 2*a4*p12*tx1 + 2*a4*p11*ty2 - 2*a6*p13*ty1 - 2*a6*p13*ty2 + 2*a6*p12*tz1 + 2*a5*p13*tz2 - 2*p22*r2*tx2 - 2*p21*r5*tx1 + 2*p21*r2*ty1 + 2*p21*r2*ty2 - 2*p23*r8*ty1 - 2*p23*r8*ty2 + 2*p23*r5*tz1 + 2*p22*r8*tz2, 2*a5*p12*tx1 - 2*a5*p12*tx2 + 2*a6*p13*tx1 - 2*a6*p13*tx2 - 2*a5*p11*ty1 + 2*a4*p12*ty2 - 2*a6*p11*tz1 + 2*a4*p13*tz2 + 2*p22*r5*tx1 - 2*p22*r5*tx2 + 2*p23*r8*tx1 - 2*p23*r8*tx2 - 2*p22*r2*ty1 + 2*p21*r5*ty2 - 2*p23*r2*tz1 + 2*p21*r8*tz2, a5*p13*tx1 + a6*p12*tx1 + a5*p13*tx2 + a6*p12*tx2 + a4*p13*ty1 - a6*p11*ty1 - a4*p13*ty2 + a6*p11*ty2 - a4*p12*tz1 - a5*p11*tz1 - a4*p12*tz2 - a5*p11*tz2 + p23*r5*tx1 + p23*r5*tx2 + p22*r8*tx1 + p22*r8*tx2 - p23*r2*ty1 + p23*r2*ty2 + p21*r8*ty1 - p21*r8*ty2 - p22*r2*tz1 - p22*r2*tz2 - p21*r5*tz1 - p21*r5*tz2, 2*a6*p13*tx1 - 2*a5*p12*tx2 - 2*a5*p12*tx1 + 2*a6*p13*tx2 + 2*a5*p11*ty1 + 2*a4*p12*ty2 - 2*a6*p11*tz1 - 2*a4*p13*tz2 - 2*p22*r5*tx1 - 2*p22*r5*tx2 + 2*p23*r8*tx1 + 2*p23*r8*tx2 + 2*p22*r2*ty1 + 2*p21*r5*ty2 - 2*p23*r2*tz1 - 2*p21*r8*tz2, 2*a5*p11*tx2 - 2*a4*p12*tx1 + 2*a4*p11*ty1 - 2*a4*p11*ty2 + 2*a6*p13*ty1 - 2*a6*p13*ty2 - 2*a6*p12*tz1 + 2*a5*p13*tz2 + 2*p22*r2*tx2 - 2*p21*r5*tx1 + 2*p21*r2*ty1 - 2*p21*r2*ty2 + 2*p23*r8*ty1 - 2*p23*r8*ty2 - 2*p23*r5*tz1 + 2*p22*r8*tz2, a4*p13*ty1 - a6*p12*tx1 - a5*p13*tx2 - a6*p12*tx2 - a5*p13*tx1 + a6*p11*ty1 + a4*p13*ty2 + a6*p11*ty2 - a4*p12*tz1 + a5*p11*tz1 + a4*p12*tz2 - a5*p11*tz2 - p23*r5*tx1 - p23*r5*tx2 - p22*r8*tx1 - p22*r8*tx2 + p23*r2*ty1 + p23*r2*ty2 + p21*r8*ty1 + p21*r8*ty2 + p22*r2*tz1 - p22*r2*tz2 - p21*r5*tz1 + p21*r5*tz2, 2*a6*p11*tx2 - 2*a4*p13*tx1 - 2*a5*p13*ty1 + 2*a6*p12*ty2 + 2*a4*p11*tz1 - 2*a4*p11*tz2 + 2*a5*p12*tz1 - 2*a5*p12*tz2 + 2*p23*r2*tx2 - 2*p21*r8*tx1 + 2*p23*r5*ty2 - 2*p22*r8*ty1 + 2*p21*r2*tz1 - 2*p21*r2*tz2 + 2*p22*r5*tz1 - 2*p22*r5*tz2, a5*p13*tx1 - a6*p12*tx1 - a5*p13*tx2 + a6*p12*tx2 - a4*p13*ty1 + a6*p11*ty1 + a4*p13*ty2 - a6*p11*ty2 + a4*p12*tz1 - a5*p11*tz1 - a4*p12*tz2 + a5*p11*tz2 - p23*r5*tx1 + p23*r5*tx2 + p22*r8*tx1 - p22*r8*tx2 + p23*r2*ty1 - p23*r2*ty2 - p21*r8*ty1 + p21*r8*ty2 - p22*r2*tz1 + p22*r2*tz2 + p21*r5*tz1 - p21*r5*tz2;
    }

    // row 1
    M[0][0] = (f1_New_C11.row(0));
    M[0][1] = (f1_New_C12.row(0));
    M[0][2] = (f1_New_C13.row(0));
    M[0][3] = (f1_New_C14.row(0));
    // row 2
    M[1][0] = (f2_New_C21.row(0));
    M[1][1] = (f2_New_C22.row(0));
    M[1][2] = (f2_New_C23.row(0));
    M[1][3] = (f2_New_C24.row(0));
    // row 3
    M[2][0] = (f3_New_C31.row(0));
    M[2][1] = (f3_New_C32.row(0));
    M[2][2] = (f3_New_C33.row(0));
    M[2][3] = (f3_New_C34.row(0));
    // row 4
    M[3][0] = (f1_New_C11.row(1));
    M[3][1] = (f1_New_C12.row(1));
    M[3][2] = (f1_New_C13.row(1));
    M[3][3] = (f1_New_C14.row(1));
    // row 5
    M[4][0] = (f2_New_C21.row(1));
    M[4][1] = (f2_New_C22.row(1));
    M[4][2] = (f2_New_C23.row(1));
    M[4][3] = (f2_New_C24.row(1));
    // row 6
    M[5][0] = (f3_New_C31.row(1));
    M[5][1] = (f3_New_C32.row(1));
    M[5][2] = (f3_New_C33.row(1));
    M[5][3] = (f3_New_C34.row(1));
    return;
}

__device__ void create_coeffs(double* coeffs, double* input, Eigen::Matrix<double,1,10>** M,
    double* input_Image_1, double* input_Image_2, double* input_affine_tran,
    double* extrinsic_R_camera, double* extrinsic_T_camera, AC_TYPE actype, 
    bool is_known_angle) {

    Eigen::Matrix3d* R_camera, *Ac;
    Eigen::Vector3d* T_camera, *Image1, *Image2;
    if (actype == GENERIC_CONSTRAINT_FULL || actype == GENERIC_CONSTRAINT_PARTIAL) {
        R_camera = (Eigen::Matrix3d*)malloc(4 * sizeof(Eigen::Matrix3d));
        T_camera = (Eigen::Vector3d*)malloc(4 * sizeof(Eigen::Vector3d));
        Ac = (Eigen::Matrix3d*)malloc(2 * sizeof(Eigen::Matrix3d));
        Image1 = (Eigen::Vector3d*)malloc(2 * sizeof(Eigen::Vector3d));
        Image2 = (Eigen::Vector3d*)malloc(2 * sizeof(Eigen::Vector3d));
        format_convert_generic(input_Image_1, input_Image_2, input_affine_tran, extrinsic_R_camera, extrinsic_T_camera, 
            Image1, Image2, Ac, R_camera, T_camera);
    } else {
        R_camera = (Eigen::Matrix3d*)malloc(2 * sizeof(Eigen::Matrix3d));
        T_camera = (Eigen::Vector3d*)malloc(2 * sizeof(Eigen::Vector3d));
        Ac = (Eigen::Matrix3d*)malloc(2 * sizeof(Eigen::Matrix3d));
        Image1 = (Eigen::Vector3d*)malloc(2 * sizeof(Eigen::Vector3d));
        Image2 = (Eigen::Vector3d*)malloc(2 * sizeof(Eigen::Vector3d));
        format_convert(input_Image_1, input_Image_2, input_affine_tran, extrinsic_R_camera, extrinsic_T_camera, 
            Image1, Image2, Ac, R_camera, T_camera);
    }

    construct_M(M, R_camera, T_camera, Ac, Image1, Image2, actype, is_known_angle);

    free(R_camera);
    free(Ac);
    free(T_camera);
    free(Image1);
    free(Image2);

    Eigen::MatrixXi Sub_index, Sub_index_extra;
    if (!is_known_angle) {
        Sub_index.resize(1,15);
        Sub_index << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14;

        Sub_index_extra.resize(1,2);
        Sub_index_extra << 0, 1;
    } else {
        Sub_index.resize(1,5);
        Sub_index << 0, 1, 3, 6, 10;

        Sub_index_extra.resize(1,1);
        Sub_index_extra << 0;
    }

    Eigen::Matrix<double,15,84> C;
    Eigen::Matrix<double,15,165> C8;
    construct_order6_poly(M, C, C8, Sub_index);

    Eigen::Matrix<double,2,84> C_extra;
    Eigen::Matrix<double,2,165> C8_extra;
    if (actype == GENERIC_CONSTRAINT_FULL || actype == CASE5_CONSTRAINT_FULL 
        || actype == INTER_CAM_CONSTRAINT_FULL || actype == INTRA_CAM_CONSTRAINT_FULL) {
        construct_order6_extra_poly(M, C_extra, C8_extra, Sub_index_extra);
    }

    // prepare data for Matlab interface
    // Matlab memory is column-major order
    int cnt = 0;
    if (actype == GENERIC_CONSTRAINT_PARTIAL || actype == CASE5_CONSTRAINT_PARTIAL 
        || actype == INTER_CAM_CONSTRAINT_PARTIAL || actype == INTRA_CAM_CONSTRAINT_PARTIAL) {
        for (int j = 0; j < 84; j++) {
            for (int i = 0; i < Sub_index.cols(); i++) {
                coeffs[cnt] = C(Sub_index(i), j);
                cnt++;
            }
        }
    }

    cnt = 0;
    if (actype == GENERIC_CONSTRAINT_FULL || actype == CASE5_CONSTRAINT_FULL 
        || actype == INTER_CAM_CONSTRAINT_FULL || actype == INTRA_CAM_CONSTRAINT_FULL) {
        for (int j = 0; j < 84; j++) {
            for (int i = 0; i < Sub_index.cols(); i++) {
                coeffs[cnt] = C(Sub_index(i), j);
                cnt++;
            }

            for (int i = 0; i < Sub_index_extra.cols(); i++) {
                coeffs[cnt] = C_extra(Sub_index_extra(i), j);
                cnt++;
            }
        }
    }

    // prepare data for the solver
    cnt = 0;
    if (actype == GENERIC_CONSTRAINT_PARTIAL || actype == CASE5_CONSTRAINT_PARTIAL
        ||actype == INTER_CAM_CONSTRAINT_PARTIAL) {
        for (int i = 0; i < Sub_index.cols(); i++) {
            for (int j = 0; j < 84; j++) {
                input[cnt] = C(Sub_index(i), j);
                cnt++;
            }
        }
    }

    cnt = 0;
    if (actype == INTRA_CAM_CONSTRAINT_PARTIAL) {
        for (int i = 0; i < Sub_index.cols(); i++) {
            for (int j = 0; j < 83; j++) {
                input[cnt] = C(Sub_index(i), j);
                cnt++;
            }
        }
    }

    cnt = 0;
    if (actype == GENERIC_CONSTRAINT_FULL || actype == CASE5_CONSTRAINT_FULL
        || actype == INTER_CAM_CONSTRAINT_FULL) {
        for (int i = 0; i < Sub_index.cols(); i++) {
            for (int j = 0; j < 84; j++) {
                input[cnt] = C(Sub_index(i), j);
                cnt++;
            }
        }

        if (is_known_angle) {
            cnt+=4;
        }

        for (int i = 0; i < Sub_index_extra.cols(); i++) {
            for (int j = 49; j < 84; j++) {
                input[cnt] = C_extra(Sub_index_extra(i), j);
                cnt++;
            }
        }
    }

    cnt = 0;
    if (actype == INTRA_CAM_CONSTRAINT_FULL) {
        for (int i = 0; i < Sub_index.cols(); i++) {
            for (int j = 0; j < 83; j++) {
                input[cnt] = C(Sub_index(i), j);
                cnt++;
            }
        }

        if (is_known_angle) {
            cnt+=4;
        }

        for (int i = 0; i < Sub_index_extra.cols(); i++) {
            for (int j = 49; j < 84; j++) {
                input[cnt] = C_extra(Sub_index_extra(i), j);
                cnt++;
            }
        }
    }
    return;
}

__device__ void calculate_M_6_by_4(Eigen::Matrix<double, 6, 4>& M_double, Eigen::Matrix<double,1,10>** M, double x, double y, double z)
{    
    Eigen::Matrix<double, 10, 1> m;
    m(0) = x*x;
    m(1) = x*y;
    m(2) = x*z;
    m(3) = x;
    m(4) = y*y;
    m(5) = y*z;
    m(6) = y;
    m(7) = z*z;
    m(8) = z;
    m(9) = 1;

    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            Eigen::Matrix<double,1,1> rslt= M[i][j]*m;
            M_double(i,j) = rslt(0);
        }
    }

    return;
}

__device__ void calculate_translation(
    Eigen::MatrixXd sols, Eigen::Matrix<double,1,10>** M, 
    Eigen::Matrix<double,3,1>* q_arr, Eigen::Matrix<double,3,1>* t_arr, bool is_known_angle) {
    for (int s = 0; s < sols.cols(); s++) {
        double x, y, z;
        x = sols(0, s);
        y = sols(1, s);
        z = sols(2, s);
        
        Eigen::Vector3d q;
        q << x, y, z;

        q_arr[s] = q;

        Eigen::Matrix<double, 6, 4> M_double;
        calculate_M_6_by_4(M_double, M, x, y, z);

        Eigen::Matrix<double, 3, 1> C12;
        if (is_known_angle) {
            Eigen::Matrix<double, 5, 3> C0 = M_double.block(0, 0, 5, 3);
            Eigen::Matrix<double, 5, 1> C1 = M_double.block(0, 3, 5, 1);
            // C12 = -C0.colPivHouseholderQr().solve(C1);
            double** A;
            double** B;
            double** X;
            malloc_matrix(5, 3, &A);
            malloc_matrix(5, 1, &B);
            malloc_matrix(3, 1, &X);

            for (int i = 0; i < 5; i++) 
                for (int j = 0; j < 3; j++) 
                    A[i][j] = C0(i, j);

            for (int i = 0; i < 5; i++) 
                for (int j = 0; j < 1; j++) 
                    B[i][j] = C1(i, j);

            qr::solve(5, 3, 1, A, B, X);
            for (int i = 0; i < 3; i++) 
                for (int j = 0; j < 1; j++) 
                    C12(i, j) = -X[i][j];
            
            free_matrix(5, 3, A);
            free_matrix(5, 1, B);
            free_matrix(3, 1, X);
        } else {
            Eigen::Matrix<double, 6, 3> C0 = M_double.block(0, 0, 6, 3);
            Eigen::Matrix<double, 6, 1> C1 = M_double.block(0, 3, 6, 1);
            // C12 = -C0.colPivHouseholderQr().solve(C1);
            double** A;
            double** B;
            double** X;
            malloc_matrix(6, 3, &A);
            malloc_matrix(6, 1, &B);
            malloc_matrix(3, 1, &X);

            for (int i = 0; i < 6; i++) 
                for (int j = 0; j < 3; j++) 
                    A[i][j] = C0(i, j);

            for (int i = 0; i < 6; i++) 
                for (int j = 0; j < 1; j++) 
                    B[i][j] = C1(i, j);

            qr::solve(6, 3, 1, A, B, X);
            for (int i = 0; i < 3; i++) 
                for (int j = 0; j < 1; j++) 
                    C12(i, j) = -X[i][j];
            
            free_matrix(6, 3, A);
            free_matrix(6, 1, B);
            free_matrix(3, 1, X);
        }
        
        t_arr[s] = C12;
    }

    return;
}