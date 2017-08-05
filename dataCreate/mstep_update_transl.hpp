#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp> 
#include <opencv2/ml/ml.hpp> 
#include "mstep_update_missingdata.hpp"

Mat_d mstep_update_transl(Mat_d P, Mat_d S_bar, Mat_d V, Mat_d E_z, std::vector<Mat_d> RO)
{

	int K = E_z.rows;
	int T = E_z.cols;
	int J = S_bar.cols;

	Mat_d Tr = Mat_d::zeros(T, 2 );
	for (int t = 0; t < T; t++)
	{
		Mat_d Sdef = S_bar;
		for (int kk = 0; kk < K; kk++)
		{
			Sdef = Sdef + E_z.at<double>(kk, t)*V.rowRange((kk) * 3,(kk+1) * 3);
		}

		Mat_d R_t = RO[t];
		Mat_d XY = R_t.rowRange(0, 2)*Sdef;
		Mat_d tempP = P.row(t);
		tempP.push_back(P.row(t + T));
		tempP = tempP - XY;
		Mat_d t_t_forSum;
		//Mat_d t_t = cv::sum(t_t_forSum, 2).div(J);
		t_t_forSum = fgSumRow(tempP);
		Mat_d t_t = t_t_forSum / J;

		Mat_d  trans_t_t = t_t.t();
		trans_t_t.copyTo(Tr.row(t));
	}

	return Tr;
}