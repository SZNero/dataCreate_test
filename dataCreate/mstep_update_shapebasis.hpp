#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp> 
#include <opencv2/ml/ml.hpp> 
#include "fgMath.hpp"
#include "computeH.hpp"

std::tuple<Mat_d, Mat_d> mstep_update_shapebasis(Mat_d P, Mat_d E_z, Mat_d E_zz, Mat_d RR, Mat_d Tr, Mat_d S_bar, Mat_d V)
{
	int K = E_z.rows;
	int T = P.rows;
	int J = P.cols;
	T = T / 2;

	Mat_d Uc = P.rowRange(0, T - 1) - Tr.col(0)*Mat_d::ones(1, J );
	Mat_d Vc = P.rowRange(T, 2 * T - 1) - Tr.col(1)*Mat_d::ones(1, J );

	Mat_d vecH_hat = computeH(Uc, Vc, E_z, E_zz, RR);
	
	Mat_d H_hat =fgReshapeForMatlab(vecH_hat,3 * J);
	Mat_d newS_bar = fgReshapeForMatlab(H_hat.col(0),3);
	Mat_d newV = Mat_d::zeros(3 * K, J );
	for (int kk = 0; kk < K; kk++)
	{
		Mat_d tempRshape = fgReshapeForMatlab(H_hat.col(kk), 3);
		tempRshape.copyTo(newV.rowRange((kk - 2) * 3 + 1, (kk - 2) * 3 + 3));
	}
	return std::make_tuple(newS_bar, newV);
}