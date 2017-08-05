#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp> 
#include <opencv2/ml/ml.hpp> 

Mat_d apprRot(Mat_d Ra)
{
	double i1 = 0.5,i2 = 0.5;
	Mat_d U = Ra.row(0);
	Mat_d V = Ra.row(1);
	double un = cv::norm(U);
	double vn = cv::norm(V);
	Mat_d Un = U / un;
	Mat_d Vn = V / vn;

	Mat_d vp = Un*Vn.t();
	Mat_d up = Vn*Un.t();

	Mat_d Vc = Vn - vp*Un; Vc = Vc / cv::norm(Vc);
	Mat_d Uc = Un - up*Vn; Uc = Uc / cv::norm(Uc);

	Mat_d Ua = i1*Un + i2*Uc; Ua = Ua / cv::norm(Ua);
	Mat_d Va = i1*Vn + i2*Vc; Va = Va / cv::norm(Va);

	Mat_d R;
	R.push_back(Ua);
	R.push_back(Va);
	R.push_back(Ua.cross(Va));

	if (cv::determinant(R) < 0)
	{
		R.row(3) = -R.row(3);
	}
	return R;
}