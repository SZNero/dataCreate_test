#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp> 
#include <opencv2/ml/ml.hpp> 
#include "fgMath.hpp"
std::tuple<Mat_d,Mat_d,double,Mat_d,Mat_d> 
mstep_lds_update(Mat_d y, std::vector<Mat_d> M, Mat_d xt_n, std::vector<Mat_d> Pt_n, std::vector<Mat_d> Ptt1_n)
{
	int q = y.rows;
	int n = y.cols - 1;
	int p = M[1].cols;

	Mat_d A = Mat_d::zeros(p, p );
	Mat_d B = Mat_d::zeros(p, p );
	Mat_d C = Mat_d::zeros(p, p );

	int tIdx = 0;

	for (int t = 0; t < n; t++)
	{
		tIdx = t + 1;
		A = A + Pt_n[tIdx - 1] + xt_n.col(tIdx - 1)*(xt_n.col(tIdx-1).t());
		B = B + Ptt1_n[tIdx] + xt_n.col(tIdx )*(xt_n.col(tIdx-1).t());
		C = C + Pt_n[tIdx] + xt_n.col(tIdx)*(xt_n.col(tIdx).t());
	}
	/*
	%----------
	% Eqns(12) - (14)
	*/
	Mat_d invA = A.inv();
	Mat_d phi = B*invA;

	//Mat_d B_transpose;
	//cv::transpose(B, B_transpose);
	Mat_d Q = (C - B*invA*B.t()) / n;
	Mat_d R = Mat_d::zeros(q, q );
	for (int t = 0; t < n; t++)
	{
		tIdx = t + 1;
		R = R + (y.col(tIdx) - M[tIdx] * xt_n.col(tIdx))*((y.col(tIdx) - M[tIdx] * xt_n.col(tIdx)).t())
			+ M[tIdx] * Pt_n[tIdx] * (M[tIdx].t());
		/*Mat_d xt_n_transpose, M_transpose;
		cv::transpose(xt_n.col(tIdx - 1), xt_n_transpose);
		cv::transpose(M[tIdx - 1], M_transpose);

		R = R + (y.col(tIdx - 1) - M[tIdx - 1] * xt_n.col(tIdx - 1))*(y.col(tIdx - 1) - M[tIdx - 1] * xt_n_transpose) +
			M[tIdx - 1] * Pt_n[tIdx - 1] * M_transpose;*/
	}

	R = R / n;
	R = Mat_d::eye(q, q )*cv::mean(R.diag())[0];
	double sigma_sq = R.at<double>(0, 0);
	Mat_d mu0 = xt_n.col(0);
	Mat_d sigma0 = Pt_n[0].clone();

	return std::make_tuple(phi, Q, sigma_sq, mu0, sigma0);
}