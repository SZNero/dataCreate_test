#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp> 
#include <opencv2/ml/ml.hpp> 

std::tuple<Mat_d,std::vector<Mat_d>, std::vector<Mat_d>,Mat_d, std::vector<Mat_d>>
kalmansmooth(Mat_d y,std::vector<Mat_d> M,double maxIter,Mat_d phi,Mat_d mu0,Mat_d sigma0,Mat_d Q,double sigma_sq,int p,int q,int n)
{
	Mat_d xt_t1 = Mat_d::zeros(p, n + 1 );
	std::vector<Mat_d> Pt_t1(n + 1);
	std::vector<Mat_d> K(n + 1);
	Mat_d xt_t = Mat_d::zeros(p, n + 1 );
	std::vector<Mat_d> Pt_t(n + 1);

	int t = 0;
	int tIdx = t;
	mu0.copyTo(xt_t.col(tIdx));
	Pt_t[tIdx] = sigma0;

	for (int t = 0; t < n; t++)
	{
		tIdx = t + 1;
		Mat_d tempMatrix = phi*xt_t.col(tIdx - 1);
		tempMatrix.copyTo(xt_t1.col(tIdx));
		Pt_t1[tIdx] = phi*Pt_t[tIdx - 1] * (phi.t()) + Q;
		if (1)
		{
			K[tIdx] = Pt_t1[tIdx]*(M[tIdx].t())*((M[tIdx]*Pt_t1[tIdx]* (M[tIdx].t()) +sigma_sq*Mat_d::eye(q,q )).inv());
		}

		tempMatrix = xt_t1.col(tIdx) + K[tIdx] * (y.col(tIdx) - M[tIdx] * xt_t1.col(tIdx));
		tempMatrix.copyTo(xt_t.col(tIdx));
		tempMatrix = Pt_t1[tIdx] - K[tIdx] * M[tIdx] * Pt_t1[tIdx];
		Pt_t[tIdx] = tempMatrix.clone();

	}

	std::vector<Mat_d> Jt(n + 1);
	Mat_d xt_n = Mat_d::zeros(p, n + 1 );
	std::vector<Mat_d> Pt_n(n + 1);

	t = n;
	tIdx = t;
	xt_t.col(tIdx).copyTo(xt_n.col(tIdx));// A9
	Pt_n[tIdx] = Pt_t[tIdx].clone();

	for (int it = n-1; it >= 0; it--)
	{
		tIdx = it + 1;

		Mat_d temp_transpose;
		//cv::transpose(phi, it_phi_trans/*pose);*/

		Jt[tIdx - 1] = Pt_t[tIdx - 1] *( phi.t())*(Pt_t1[tIdx].inv());
	//	xt_n.col(tIdx - 1) = 
		temp_transpose = xt_t.col(tIdx - 1) + Jt[tIdx - 1] * (xt_n.col(tIdx) - phi*xt_t.col(tIdx - 1));
		temp_transpose.copyTo(xt_n.col(tIdx - 1));
		Pt_n[tIdx - 1] = Pt_t[tIdx - 1] + Jt[tIdx - 1] * (Pt_n[tIdx] - Pt_t1[tIdx])*(Jt[tIdx-1].t());
	}

	std::vector<Mat_d> Ptt1_n(n + 1);

	t = n; tIdx = t;
	Ptt1_n[tIdx] = (Mat_d::eye(p, p ) - K[tIdx] * M[tIdx] ) * phi*Pt_t[tIdx - 1];
	for (int it = n; it > 1; it--)
	{
		tIdx = it;
		Ptt1_n[tIdx - 1] = Pt_t[tIdx - 1] * (Jt[tIdx-2].t()) + Jt[tIdx - 1] * (Ptt1_n[tIdx] - phi*Pt_t[tIdx -1] * (Jt[tIdx-2].t()));
	}

	return std::make_tuple(xt_n, Pt_n, Ptt1_n, xt_t1, Pt_t1);
}