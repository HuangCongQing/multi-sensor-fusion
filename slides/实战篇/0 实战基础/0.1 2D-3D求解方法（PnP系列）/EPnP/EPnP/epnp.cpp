#include <iostream>
using namespace std;
#include "precomp.hpp"
#include "epnp.h"

epnp::epnp(const cv::Mat& cameraMatrix, const cv::Mat& opoints, const cv::Mat& ipoints)
{
  if (cameraMatrix.depth() == CV_32F)
      init_camera_parameters<float>(cameraMatrix);
  else
    init_camera_parameters<double>(cameraMatrix);

  number_of_correspondences = std::max(opoints.checkVector(3, CV_32F), opoints.checkVector(3, CV_64F));    // 多少组2d-3d对应点

  pws.resize(3 * number_of_correspondences);                   // 世界坐标系中的3D点
  us.resize(2 * number_of_correspondences);                    // 图像坐标系中的2D点

  if (opoints.depth() == ipoints.depth())
  {
    if (opoints.depth() == CV_32F)                             // 初始化，把opoints和ipoints中的点存到pws和us中
      init_points<cv::Point3f,cv::Point2f>(opoints, ipoints);
    else
      init_points<cv::Point3d,cv::Point2d>(opoints, ipoints);
  }
  else if (opoints.depth() == CV_32F)
    init_points<cv::Point3f,cv::Point2d>(opoints, ipoints);
  else
    init_points<cv::Point3d,cv::Point2f>(opoints, ipoints);

  alphas.resize(4 * number_of_correspondences);
  pcs.resize(3 * number_of_correspondences);

  max_nr = 0;
  A1 = NULL;
  A2 = NULL;
}

epnp::~epnp()
{
    if (A1)
        delete[] A1;
    if (A2)
        delete[] A2;
}

void epnp::choose_control_points(void)
{
  // Take C0 as the reference points centroid:
  cws[0][0] = cws[0][1] = cws[0][2] = 0;                            
  for(int i = 0; i < number_of_correspondences; i++)
    for(int j = 0; j < 3; j++)
      cws[0][j] += pws[3 * i + j];

  for(int j = 0; j < 3; j++)                                     // cws0为全部3D点的质点（参考点）
    cws[0][j] /= number_of_correspondences;


  // Take C1, C2, and C3 from PCA on the reference points:
  CvMat * PW0 = cvCreateMat(number_of_correspondences, 3, CV_64F);

  double pw0tpw0[3 * 3], dc[3], uct[3 * 3];
  CvMat PW0tPW0 = cvMat(3, 3, CV_64F, pw0tpw0);
  CvMat DC      = cvMat(3, 1, CV_64F, dc);
  CvMat UCt     = cvMat(3, 3, CV_64F, uct);

  for(int i = 0; i < number_of_correspondences; i++)             // 去中心(PW0其实就是所有3D点到中心点cws0的距离)
    for(int j = 0; j < 3; j++)
      PW0->data.db[3 * i + j] = pws[3 * i + j] - cws[0][j];      //   3*3      3*n    n*3     
                                                                 //    ^        ^      ^
  cvMulTransposed(PW0, &PW0tPW0, 1);                             // PW0tPW0 = PW0^t * PW0. 求协方差矩阵。  如果cvMulTransposed的第三个参数为0，那么PW0PW0t = PW0 * PW0^t
  cvSVD(&PW0tPW0, &DC, &UCt, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);   // A=U*W*VT; cvSVD(A, W, U, V, flags); W(DC):3*1; U(UCt):3*3; V:0（表示不返回V的值）

  cvReleaseMat(&PW0);

  for(int i = 1; i < 4; i++) { 
    double k = sqrt(dc[i - 1] / number_of_correspondences);      // 以cws0为中心的主轴(经过PCA得到)上的单位向量 为cws[1][2][3]
    for(int j = 0; j < 3; j++)
      cws[i][j] = cws[0][j] + k * uct[3 * (i - 1) + j];
  }
}

void epnp::compute_barycentric_coordinates(void)
{
  double cc[3 * 3], cc_inv[3 * 3];
  CvMat CC     = cvMat(3, 3, CV_64F, cc);
  CvMat CC_inv = cvMat(3, 3, CV_64F, cc_inv);

  for(int i = 0; i < 3; i++)                                      //      | cws1_x-cws0_x  cws2_x-cws0_x  cws3_x-cws0_x |    cws0为世界坐标系下的中心控制点
    for(int j = 1; j < 4; j++)                                    // CC = | cws1_y-cws0_y  cws2_y-cws0_y  cws3_y-cws0_y |    cws1 cws2 cws3为世界坐标系下的其他控制点
      cc[3 * i + j - 1] = cws[j][i] - cws[0][i];                  //      | cws1_z-cws0_z  cws2_z-cws0_z  cws3_z-cws0_z |

  cvInvert(&CC, &CC_inv, CV_SVD);                                 // 以svd方式求CC的逆
  double * ci = cc_inv;
  for(int i = 0; i < number_of_correspondences; i++) {
    double * pi = &pws[0] + 3 * i;
    double * a = &alphas[0] + 4 * i;

    for(int j = 0; j < 3; j++)                   
      a[1 + j] =                                                  // |ai_1|      -1   |pi_x-cws0_x|    pi为世界坐标系下的3d点
  ci[3 * j    ] * (pi[0] - cws[0][0]) +                           // |ai_2| =  CC   * |pi_y-cws0_y|
  ci[3 * j + 1] * (pi[1] - cws[0][1]) +                           // |ai_3|           |pi_z-cws0_z|
  ci[3 * j + 2] * (pi[2] - cws[0][2]);                            
    a[0] = 1.0f - a[1] - a[2] - a[3];                             //  ai_0 = 1 - ai_1 - ai_2 - ai_3 
  }                                                               //  pi = ai_0*cws0 + ai_1*cws1 + ai_2*cws2 + ai_3*cws3;
}

void epnp::fill_M(CvMat * M,
      const int row, const double * as, const double u, const double v)    // M 的推导见论文 很简单
{ 
  double * M1 = M->data.db + row * 12;     //     | ai_0*fx  0  ai_0*(cx-ui)  ai_1*fx  0  ai_1*(cx-ui)  ai_2*fx  0  ai_2*(cx-ui)  ai_3*fx  0  ai_3*(cx-ui) | -> M1
  double * M2 = M1 + 12;                   //     | 0  ai_0*fy  ai_0*(cy-vi)  0  ai_1*fy  ai_1*(cy-vi)  0  ai_2*fy  ai_2*(cy-vi)  0  ai_3*fy  ai_3*(cy-vi) | -> M2
                                           //     |       .  .             .        .  .             .        .  .             .        .  .             . |
  for(int i = 0; i < 4; i++) {             //     |       .  .             .        .  .             .        .  .             .        .  .             . |
    M1[3 * i    ] = as[i] * fu;            //     |       .  .             .        .  .             .        .  .             .        .  .             . |
    M1[3 * i + 1] = 0.0;                   // M = |       .  .             .   一组点提供两个式子，n组点 调用n次 fill_M 函数 来填充M矩阵    .  .             . |
    M1[3 * i + 2] = as[i] * (uc - u);      //     |       .  .             .        .  .             .        .  .             .        .  .             . |
                                           //     |       .  .             .        .  .             .        .  .             .        .  .             . |
    M2[3 * i    ] = 0.0;                   //     |       .  .             .        .  .             .        .  .             .        .  .             . |
    M2[3 * i + 1] = as[i] * fv;            //     | an_0*fx  0  an_0*(cx-ui)  an_1*fx  0  an_1*(cx-ui)  an_2*fx  0  an_2*(cx-ui)  an_3*fx  0  an_3*(cx-ui) |
    M2[3 * i + 2] = as[i] * (vc - v);      //     | 0  an_0*fy  an_0*(cy-vi)  0  an_1*fy  an_1*(cy-vi)  0  an_2*fy  an_2*(cy-vi)  0  an_3*fy  an_3*(cy-vi) |
  }
}

void epnp::compute_ccs(const double * betas, const double * ut)
{
  for(int i = 0; i < 4; i++)
    ccs[i][0] = ccs[i][1] = ccs[i][2] = 0.0f;                 // 计算控制点在相机坐标系下的坐标

  for(int i = 0; i < 4; i++) {                                // x = betas1*v1 + betas2*v2 + betas3*v3 + betas4*v4
    const double * v = ut + 12 * (11 - i);
    for(int j = 0; j < 4; j++)
      for(int k = 0; k < 3; k++)
        ccs[j][k] += betas[i] * v[3 * j + k];
  }
}

void epnp::compute_pcs(void)                                  // 计算3d点在相机坐标系下的坐标
{
  for(int i = 0; i < number_of_correspondences; i++) {
    double * a = &alphas[0] + 4 * i;
    double * pc = &pcs[0] + 3 * i;
                                                              // pcsi = ai_0*cws0 + ai_1*cws1 + ai_2*cws2 + ai_3*cws3
    for(int j = 0; j < 3; j++)                                
      pc[j] = a[0] * ccs[0][j] + a[1] * ccs[1][j] + a[2] * ccs[2][j] + a[3] * ccs[3][j];
  }
}

void epnp::compute_pose(cv::Mat& R, cv::Mat& t)
{
  choose_control_points();                                                    // 选择4个控制点（质点+3个主轴方向的单位向量）
  compute_barycentric_coordinates();                                          // 根据4个控制点计算所有3d空间点的阿尔法系数
                                                                              // pi = ai_0*cws0 + ai_1*cws1 + ai_2*cws2 + ai_3*cws3
  CvMat * M = cvCreateMat(2 * number_of_correspondences, 12, CV_64F);         // Mx=0 M为2n*12的矩阵，x为4个控制点在相机坐标系下的xyz值，为12*1矩阵

  for(int i = 0; i < number_of_correspondences; i++)                          // 填充M矩阵（由阿尔法，uv和fxfycxcy组成）
    fill_M(M, 2 * i, &alphas[0] + 4 * i, us[2 * i], us[2 * i + 1]);

  double mtm[12 * 12], d[12], ut[12 * 12];
  CvMat MtM = cvMat(12, 12, CV_64F, mtm);
  CvMat D   = cvMat(12,  1, CV_64F, d);
  CvMat Ut  = cvMat(12, 12, CV_64F, ut);

  cvMulTransposed(M, &MtM, 1);                                                // 对M^tM进行SVD分解，得到的特征向量既是M的右奇异向量
  cvSVD(&MtM, &D, &Ut, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);
  cvReleaseMat(&M);

  double l_6x10[6 * 10], rho[6];                                              // L * betas = rho
  CvMat L_6x10 = cvMat(6, 10, CV_64F, l_6x10);
  CvMat Rho    = cvMat(6,  1, CV_64F, rho);

  compute_L_6x10(ut, l_6x10);                                                 // N = 4 时， L为6*10的矩阵
  compute_rho(rho);                                                           // rho 一直为 6*1 的矩阵，记录着4个控制点之间各自的距离

  double Betas[4][4], rep_errors[4];
  double Rs[4][3][3], ts[4][3];

  find_betas_approx_1(&L_6x10, &Rho, Betas[1]);                               // 算出4个betas的初值 （N=4: B1 B2 B3 B4）
  gauss_newton(&L_6x10, &Rho, Betas[1]);                                      // 根据L*betas-rho的误差，迭代精确（提纯）betas
  rep_errors[1] = compute_R_and_t(ut, Betas[1], Rs[1], ts[1]);                // 利用v和betas，先求出4个ccs，再求出pcs，然后利用pcs和pws计算R和t，最后计算重投影误差

  find_betas_approx_2(&L_6x10, &Rho, Betas[2]);                               // 算4个betas初值（N=2: B1 B2 (B3B4为0)）
  gauss_newton(&L_6x10, &Rho, Betas[2]);
  rep_errors[2] = compute_R_and_t(ut, Betas[2], Rs[2], ts[2]);

  find_betas_approx_3(&L_6x10, &Rho, Betas[3]);                               // 算4个betas初值（N=3: B1 B2 B3 (B4为0)）
  gauss_newton(&L_6x10, &Rho, Betas[3]);
  rep_errors[3] = compute_R_and_t(ut, Betas[3], Rs[3], ts[3]);

  int N = 1;
  if (rep_errors[2] < rep_errors[1]) N = 2;                                   // 选出重投影误差最小的R和t
  if (rep_errors[3] < rep_errors[N]) N = 3;

  cv::Mat(3, 1, CV_64F, ts[N]).copyTo(t);
  cv::Mat(3, 3, CV_64F, Rs[N]).copyTo(R);
}

void epnp::copy_R_and_t(const double R_src[3][3], const double t_src[3],
      double R_dst[3][3], double t_dst[3])
{
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      R_dst[i][j] = R_src[i][j];
    t_dst[i] = t_src[i];
  }
}

double epnp::dist2(const double * p1, const double * p2)          // 计算两点间距离
{
  return
    (p1[0] - p2[0]) * (p1[0] - p2[0]) +
    (p1[1] - p2[1]) * (p1[1] - p2[1]) +
    (p1[2] - p2[2]) * (p1[2] - p2[2]);
}

double epnp::dot(const double * v1, const double * v2)            // 向量点乘
{
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

void epnp::estimate_R_and_t(double R[3][3], double t[3])          // 计算R和t （Horn，...）
{
  double pc0[3], pw0[3];

  pc0[0] = pc0[1] = pc0[2] = 0.0;
  pw0[0] = pw0[1] = pw0[2] = 0.0;

  for(int i = 0; i < number_of_correspondences; i++) {
    const double * pc = &pcs[3 * i];
    const double * pw = &pws[3 * i];

    for(int j = 0; j < 3; j++) {
      pc0[j] += pc[j];
      pw0[j] += pw[j];
    }
  }
  for(int j = 0; j < 3; j++) {                                     // 求出pcs和pws的几何中心
    pc0[j] /= number_of_correspondences;
    pw0[j] /= number_of_correspondences;
  }

  double abt[3 * 3], abt_d[3], abt_u[3 * 3], abt_v[3 * 3];
  CvMat ABt   = cvMat(3, 3, CV_64F, abt);
  CvMat ABt_D = cvMat(3, 1, CV_64F, abt_d);
  CvMat ABt_U = cvMat(3, 3, CV_64F, abt_u);
  CvMat ABt_V = cvMat(3, 3, CV_64F, abt_v);

  cvSetZero(&ABt);
  for(int i = 0; i < number_of_correspondences; i++) {
    double * pc = &pcs[3 * i];
    double * pw = &pws[3 * i];

    for(int j = 0; j < 3; j++) {
      abt[3 * j    ] += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);       // 对 各个pcs和pws点减去各自的几何中心所形成的矩阵 进行svd分解
      abt[3 * j + 1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
      abt[3 * j + 2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
    }
  }

  cvSVD(&ABt, &ABt_D, &ABt_U, &ABt_V, CV_SVD_MODIFY_A);

  for(int i = 0; i < 3; i++)                                       // 计算旋转矩阵
    for(int j = 0; j < 3; j++)
      R[i][j] = dot(abt_u + 3 * i, abt_v + 3 * j);

  const double det =
    R[0][0] * R[1][1] * R[2][2] + R[0][1] * R[1][2] * R[2][0] + R[0][2] * R[1][0] * R[2][1] -
    R[0][2] * R[1][1] * R[2][0] - R[0][1] * R[1][0] * R[2][2] - R[0][0] * R[1][2] * R[2][1];

  if (det < 0) {
    R[2][0] = -R[2][0];
    R[2][1] = -R[2][1];
    R[2][2] = -R[2][2];
  }

  t[0] = pc0[0] - dot(R[0], pw0);                                  // 计算平移矩阵
  t[1] = pc0[1] - dot(R[1], pw0);
  t[2] = pc0[2] - dot(R[2], pw0);
}

void epnp::solve_for_sign(void)
{
  if (pcs[2] < 0.0) {                        // 检查第一个相机坐标系下的3d点，若发现深度为负，则调整所有ccs和所有pcs使其坐标非负
    for(int i = 0; i < 4; i++)
      for(int j = 0; j < 3; j++)
        ccs[i][j] = -ccs[i][j];

    for(int i = 0; i < number_of_correspondences; i++) {
      pcs[3 * i    ] = -pcs[3 * i];
      pcs[3 * i + 1] = -pcs[3 * i + 1];
      pcs[3 * i + 2] = -pcs[3 * i + 2];
    }
  }
}

double epnp::compute_R_and_t(const double * ut, const double * betas,
           double R[3][3], double t[3])
{
  compute_ccs(betas, ut);                   // 计算控制点在相机坐标系下的坐标 x=betas*v
  compute_pcs();                            // 计算3d点在相机坐标系下的坐标 pcsi = ai_0*cws0 + ai_1*cws1 + ai_2*cws2 + ai_3*cws3

  solve_for_sign();                         // 保证pcs和ccs坐标非负

  estimate_R_and_t(R, t);                   // 计算R和t

  return reprojection_error(R, t);          // 计算重投影误差并返回其值
}

double epnp::reprojection_error(const double R[3][3], const double t[3])
{
  double sum2 = 0.0;

  for(int i = 0; i < number_of_correspondences; i++) {
    double * pw = &pws[3 * i];
    double Xc = dot(R[0], pw) + t[0];                                // pws经外参（R和t）变为pcs
    double Yc = dot(R[1], pw) + t[1];
    double inv_Zc = 1.0 / (dot(R[2], pw) + t[2]);
    double ue = uc + fu * Xc * inv_Zc;                               // pcs经内参变为uv
    double ve = vc + fv * Yc * inv_Zc;
    double u = us[2 * i], v = us[2 * i + 1];

    sum2 += sqrt( (u - ue) * (u - ue) + (v - ve) * (v - ve) );       // 计算与真实2d点之间的误差
  }

  return sum2 / number_of_correspondences;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]                   N=4

void epnp::find_betas_approx_1(const CvMat * L_6x10, const CvMat * Rho,
             double * betas)
{
  double l_6x4[6 * 4], b4[4];
  CvMat L_6x4 = cvMat(6, 4, CV_64F, l_6x4);
  CvMat B4    = cvMat(4, 1, CV_64F, b4);

  for(int i = 0; i < 6; i++) {
    cvmSet(&L_6x4, i, 0, cvmGet(L_6x10, i, 0));
    cvmSet(&L_6x4, i, 1, cvmGet(L_6x10, i, 1));
    cvmSet(&L_6x4, i, 2, cvmGet(L_6x10, i, 3));
    cvmSet(&L_6x4, i, 3, cvmGet(L_6x10, i, 6));
  }

  cvSolve(&L_6x4, Rho, &B4, CV_SVD);                               // 解线性方程组 求出B11 B12 B13 B14

  if (b4[0] < 0) {                                                 // 然后再求出 B1 B2 B3 B4
    betas[0] = sqrt(-b4[0]);
    betas[1] = -b4[1] / betas[0];
    betas[2] = -b4[2] / betas[0];
    betas[3] = -b4[3] / betas[0];
  } else {
    betas[0] = sqrt(b4[0]);
    betas[1] = b4[1] / betas[0];
    betas[2] = b4[2] / betas[0];
    betas[3] = b4[3] / betas[0];
  }
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]              N=2

void epnp::find_betas_approx_2(const CvMat * L_6x10, const CvMat * Rho,
             double * betas)
{
  double l_6x3[6 * 3], b3[3];
  CvMat L_6x3  = cvMat(6, 3, CV_64F, l_6x3);
  CvMat B3     = cvMat(3, 1, CV_64F, b3);

  for(int i = 0; i < 6; i++) {
    cvmSet(&L_6x3, i, 0, cvmGet(L_6x10, i, 0));
    cvmSet(&L_6x3, i, 1, cvmGet(L_6x10, i, 1));
    cvmSet(&L_6x3, i, 2, cvmGet(L_6x10, i, 2));
  }

  cvSolve(&L_6x3, Rho, &B3, CV_SVD);                                       // 解线性方程组 求出B11 B12 B22

  if (b3[0] < 0) {                                                         // 然后再求出 B1 B2
    betas[0] = sqrt(-b3[0]);
    betas[1] = (b3[2] < 0) ? sqrt(-b3[2]) : 0.0;
  } else {
    betas[0] = sqrt(b3[0]);
    betas[1] = (b3[2] > 0) ? sqrt(b3[2]) : 0.0;
  }

  if (b3[1] < 0) betas[0] = -betas[0];

  betas[2] = 0.0;
  betas[3] = 0.0;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]               N=3

void epnp::find_betas_approx_3(const CvMat * L_6x10, const CvMat * Rho,
             double * betas)
{
  double l_6x5[6 * 5], b5[5];
  CvMat L_6x5 = cvMat(6, 5, CV_64F, l_6x5);
  CvMat B5    = cvMat(5, 1, CV_64F, b5);

  for(int i = 0; i < 6; i++) {
    cvmSet(&L_6x5, i, 0, cvmGet(L_6x10, i, 0));
    cvmSet(&L_6x5, i, 1, cvmGet(L_6x10, i, 1));
    cvmSet(&L_6x5, i, 2, cvmGet(L_6x10, i, 2));
    cvmSet(&L_6x5, i, 3, cvmGet(L_6x10, i, 3));
    cvmSet(&L_6x5, i, 4, cvmGet(L_6x10, i, 4));
  }

  cvSolve(&L_6x5, Rho, &B5, CV_SVD);                                        // 解线性方程组 求出B11 B12 B22 B13 B23       

  if (b5[0] < 0) {
    betas[0] = sqrt(-b5[0]);
    betas[1] = (b5[2] < 0) ? sqrt(-b5[2]) : 0.0;
  } else {
    betas[0] = sqrt(b5[0]);                                                 // 然后再求出 B1 B2 B3
    betas[1] = (b5[2] > 0) ? sqrt(b5[2]) : 0.0;
  }
  if (b5[1] < 0) betas[0] = -betas[0];
  betas[2] = b5[3] / betas[0];
  betas[3] = 0.0;
}

void epnp::compute_L_6x10(const double * ut, double * l_6x10)
{
  const double * v[4];

  v[0] = ut + 12 * 11;               // 最后一个特征向量   （因为是ut，所以这里特征向量为 行向量）
  v[1] = ut + 12 * 10;               // 倒数第二个特征向量
  v[2] = ut + 12 *  9;               // 倒数第三个特征向量
  v[3] = ut + 12 *  8;               // 倒数第四个特征向量

  double dv[4][6][3];

  for(int i = 0; i < 4; i++) {                         // betas1 对应的是 v的前三个值，以此类推
    int a = 0, b = 1;
    for(int j = 0; j < 6; j++) {
      dv[i][j][0] = v[i][3 * a    ] - v[i][3 * b];              // 4个v（i），每个v对应6个组合（j），（4个控制点，每个控制点对应三个值），一共4*6=24种形式
      dv[i][j][1] = v[i][3 * a + 1] - v[i][3 * b + 1];
      dv[i][j][2] = v[i][3 * a + 2] - v[i][3 * b + 2];      

      b++;
      if (b > 3) {                                 //       a   b 控制着6个组合（6行）。
        a++;                                       //       ^   ^
        b = a + 1;                                 //       |   |
      }                                            //       |   |
    }                                              //       |   |
  }                                                //       |   |
                                                   //       |   |
  for(int i = 0; i < 6; i++) {                     //       |   |
    double * row = l_6x10 + 10 * i;                //       |   |           L (6*10)                 *    bates (10*1)   = rho (6*1)
                                                   //       |   |
    row[0] =        dot(dv[0][i], dv[0][i]);       // | ||v1i-v1j||^2  2*|(v1i-v1j)(v2i-v2j)|    ..|   | bates1*betas1 |   | dcw0_1 |
    row[1] = 2.0f * dot(dv[0][i], dv[1][i]);       // |      ..                                    |   | betas1*betas2 |   | dcw0_2 |
    row[2] =        dot(dv[1][i], dv[1][i]);       // |      ..                                    |   | betas2*betas2 |   | dcw0_3 |
    row[3] = 2.0f * dot(dv[0][i], dv[2][i]);       // |      ..                                    | * | betas1*betas3 | = | dcw1_2 |
    row[4] = 2.0f * dot(dv[1][i], dv[2][i]);       // |      ..                                    |   | betas2*betas3 |   | dcw1_3 |
    row[5] =        dot(dv[2][i], dv[2][i]);       // |      ..                                    |   | betas3*betas3 |   | dcw2_3 |
    row[6] = 2.0f * dot(dv[0][i], dv[3][i]);       //                                                  | betas1*betas4 |
    row[7] = 2.0f * dot(dv[1][i], dv[3][i]);       //                                                  | betas2*betas4 |
    row[8] = 2.0f * dot(dv[2][i], dv[3][i]);       //                                                  | betas3*betas4 |
    row[9] =        dot(dv[3][i], dv[3][i]);       //                                                  | betas4*betas4 |
  }
}

void epnp::compute_rho(double * rho)                                 // rho 一直为 6*1 的矩阵，记录着4个控制点之间各自的距离
{
  rho[0] = dist2(cws[0], cws[1]);
  rho[1] = dist2(cws[0], cws[2]);
  rho[2] = dist2(cws[0], cws[3]);
  rho[3] = dist2(cws[1], cws[2]);
  rho[4] = dist2(cws[1], cws[3]);
  rho[5] = dist2(cws[2], cws[3]);
}

void epnp::compute_A_and_b_gauss_newton(const double * l_6x10, const double * rho,               // A * delta_x = b
          const double betas[4], CvMat * A, CvMat * b)
{
  for(int i = 0; i < 6; i++) {                                  // 注意 betas10 = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44] L与之对应
    const double * rowL = l_6x10 + i * 10;
    double * rowA = A->data.db + i * 4;

    rowA[0] = 2 * rowL[0] * betas[0] +     rowL[1] * betas[1] +     rowL[3] * betas[2] +     rowL[6] * betas[3];          // 4个含B1的
    rowA[1] =     rowL[1] * betas[0] + 2 * rowL[2] * betas[1] +     rowL[4] * betas[2] +     rowL[7] * betas[3];          // 4个含B2的
    rowA[2] =     rowL[3] * betas[0] +     rowL[4] * betas[1] + 2 * rowL[5] * betas[2] +     rowL[8] * betas[3];          // 4个含B3的
    rowA[3] =     rowL[6] * betas[0] +     rowL[7] * betas[1] +     rowL[8] * betas[2] + 2 * rowL[9] * betas[3];          // 4个含B4的

    cvmSet(b, i, 0, rho[i] -                                    // b 为 L*betas-rho （这里的betas是初值，已经通过find_betas_approx函数求出了）
     (
      rowL[0] * betas[0] * betas[0] +
      rowL[1] * betas[0] * betas[1] +
      rowL[2] * betas[1] * betas[1] +
      rowL[3] * betas[0] * betas[2] +
      rowL[4] * betas[1] * betas[2] +
      rowL[5] * betas[2] * betas[2] +
      rowL[6] * betas[0] * betas[3] +
      rowL[7] * betas[1] * betas[3] +
      rowL[8] * betas[2] * betas[3] +
      rowL[9] * betas[3] * betas[3]
      ));
  }
}

void epnp::gauss_newton(const CvMat * L_6x10, const CvMat * Rho, double betas[4])
{
  const int iterations_number = 5;

  double a[6*4], b[6], x[4];
  CvMat A = cvMat(6, 4, CV_64F, a);
  CvMat B = cvMat(6, 1, CV_64F, b);
  CvMat X = cvMat(4, 1, CV_64F, x);

  for(int k = 0; k < iterations_number; k++)                            // 高斯牛顿迭代，提纯beta1,beta2,beta3和beta4
  {
    compute_A_and_b_gauss_newton(L_6x10->data.db, Rho->data.db, betas, &A, &B);               // 最小化||L_i * betas - rho_i||，i=0,...5.
    qr_solve(&A, &B, &X);
    for(int i = 0; i < 4; i++)
      betas[i] += x[i];                                                                       // 调整 betas（根据计算出的 使得误差最小 的delta）
  }
}

void epnp::qr_solve(CvMat * A, CvMat * b, CvMat * X)
{
  const int nr = A->rows;
  const int nc = A->cols;

  if (max_nr != 0 && max_nr < nr)
  {
    delete [] A1;
    delete [] A2;
  }
  if (max_nr < nr)
  {
    max_nr = nr;
    A1 = new double[nr];
    A2 = new double[nr];
  }

  double * pA = A->data.db, * ppAkk = pA;
  for(int k = 0; k < nc; k++)
  {
    double * ppAik1 = ppAkk, eta = fabs(*ppAik1);
    for(int i = k + 1; i < nr; i++)
    {
      double elt = fabs(*ppAik1);
      if (eta < elt) eta = elt;
      ppAik1 += nc;
    }
    if (eta == 0)
    {
      A1[k] = A2[k] = 0.0;
      //cerr << "God damnit, A is singular, this shouldn't happen." << endl;
      return;
    }
    else
    {
      double * ppAik2 = ppAkk, sum2 = 0.0, inv_eta = 1. / eta;
      for(int i = k; i < nr; i++)
      {
        *ppAik2 *= inv_eta;
        sum2 += *ppAik2 * *ppAik2;
        ppAik2 += nc;
      }
      double sigma = sqrt(sum2);
      if (*ppAkk < 0)
      sigma = -sigma;
      *ppAkk += sigma;
      A1[k] = sigma * *ppAkk;
      A2[k] = -eta * sigma;
      for(int j = k + 1; j < nc; j++)
      {
        double * ppAik = ppAkk, sum = 0;
        for(int i = k; i < nr; i++)
        {
          sum += *ppAik * ppAik[j - k];
          ppAik += nc;
        }
        double tau = sum / A1[k];
        ppAik = ppAkk;
        for(int i = k; i < nr; i++)
        {
          ppAik[j - k] -= tau * *ppAik;
          ppAik += nc;
        }
      }
    }
    ppAkk += nc + 1;
  }

  // b <- Qt b
  double * ppAjj = pA, * pb = b->data.db;
  for(int j = 0; j < nc; j++)
  {
    double * ppAij = ppAjj, tau = 0;
    for(int i = j; i < nr; i++)
    {
      tau += *ppAij * pb[i];
      ppAij += nc;
    }
    tau /= A1[j];
    ppAij = ppAjj;
    for(int i = j; i < nr; i++)
    {
      pb[i] -= tau * *ppAij;
      ppAij += nc;
    }
    ppAjj += nc + 1;
  }

  // X = R-1 b
  double * pX = X->data.db;
  pX[nc - 1] = pb[nc - 1] / A2[nc - 1];
  for(int i = nc - 2; i >= 0; i--)
  {
    double * ppAij = pA + i * nc + (i + 1), sum = 0;

    for(int j = i + 1; j < nc; j++)
    {
      sum += *ppAij * pX[j];
      ppAij++;
    }
    pX[i] = (pb[i] - sum) / A2[i];
  }
}
