#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp> 
#include <opencv2/ml/ml.hpp> 
#include "kalmansmooth.hpp"
std::tuple<Mat_d, Mat_d, Mat_d, std::vector<Mat_d>, Mat_d, std::vector<Mat_d>, std::vector<Mat_d>, Mat_d,std::vector<Mat_d>>
estep_lds_compute_Z_distr(Mat_d P,Mat_d S_bar,Mat_d V,Mat_d RR,Mat_d Tr,Mat_d phi,Mat_d mu0,Mat_d sigma0,Mat_d Q,double sigma_sq)
{
	int K = V.rows / 3;
	int T = P.rows / 2;
	int J = P.cols;

	Mat_d Tr_temp;
	Tr_temp.push_back(Tr.col(0));
	Tr_temp.push_back(Tr.col(1));

	Mat_d Pc = P - Tr_temp*Mat_d::ones(1, J );
	Mat_d M_t = Mat_d::zeros(2 * J, K );
	Mat_d P_hat_t = Mat_d::zeros(2 * J, 1 );

	Mat_d E_z = Mat_d::zeros(K, T );
	Mat_d E_zz = Mat_d::zeros(T*K, K );

	Mat_d invSigmaSq_p = Mat_d::eye(2 * J, 2 * J ) / sigma_sq;

	std::vector<Mat_d> M(T+1);
	M[0] = Mat_d::zeros(0,0 );
	Mat_d y = Mat_d::zeros(2 * J, T + 1 );
	y.col(1).setTo(0); //scalar(0)

	for (int t = 0; t < T; t++)
	{
		Mat_d Pt,transpose_temp;

		Pt.push_back(P.row(t).t());
		Pt.push_back(P.row(t + T).t());
		Mat_d Rt;
		Rt.push_back(RR.row(t));
		Rt.push_back(RR.row(t + T));

		for (int kk = 0; kk < K; kk++)
		{
			Mat_d transpos_kk_temp;
			transpos_kk_temp = (Rt.row(0)*V.rowRange(kk * 3, (kk + 1) * 3)).t();
			transpos_kk_temp.copyTo(M_t.rowRange(0, J).col(kk));
			transpos_kk_temp = (Rt.row(1)*V.rowRange(kk * 3, (kk + 1) * 3)).t();
			transpos_kk_temp.copyTo(M_t.rowRange(J, M_t.rows).col(kk));
		}
		transpose_temp = (Rt.row(0)*S_bar).t();
		transpose_temp.copyTo(P_hat_t.rowRange(0, J));

		transpose_temp = (Rt.row(1)*S_bar).t();
		transpose_temp.copyTo(P_hat_t.rowRange(J, P_hat_t.rows));

		transpose_temp = Pt - P_hat_t;
		transpose_temp.copyTo(y.col(t+1));

		M[t+1] = M_t.clone();
	}

	double maxIter = 20.0f;
	Mat_d xt_n, xt_t1;
	std::vector<Mat_d> Pt_n, Ptt1_n, Pt_t1;

	Mat_d mu0matrix(4, 1 , mu0.at<double>(0,0));

	std::tie(xt_n, Pt_n, Ptt1_n, xt_t1, Pt_t1) = kalmansmooth(y, M, maxIter, phi, mu0, sigma0, Q, sigma_sq, K, 2 * J, T);
	E_z = xt_n.colRange(1, xt_n.cols);
	for (int t = 0; t < T; t++)
	{
		Mat_d tempMat = Pt_n[t+1] + E_z.col(t)*(E_z.col(t).t());
		tempMat.copyTo(E_zz.rowRange((t)*K, (t+1)*K));
	}

	return std::make_tuple(E_z, E_zz, y, M, xt_n, Pt_n, Ptt1_n, xt_t1, Pt_t1);
}