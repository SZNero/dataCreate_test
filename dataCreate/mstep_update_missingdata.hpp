#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp> 
#include <opencv2/ml/ml.hpp> 
#include "fgMath.hpp"


Mat_d mstep_update_missingdata(Mat_d P, Mat_d MD, Mat_d S_bar, Mat_d V, Mat_d E_z, std::vector<Mat_d> RO, Mat_d Tr)
{
	int K = E_z.rows;
	int T = E_z.cols;

	int J = S_bar.cols;

	Mat_d MD_transpose;
	cv::transpose(MD, MD_transpose);

	Mat_d sumMatrix = fgSum(MD_transpose);

	std::vector<int> ind = fgFindNonZeros(sumMatrix);

	Mat_d P_hat = P;
	for (int kk = 0; kk < ind.size(); kk++)
	{
		int t = ind[kk];
		//Mat_d MD_temp = MD.row(t - 1);
		std::vector<int> missingpoints_t = fgFindNonZeros(MD.row(t - 1));
		for (int s = 0; s < missingpoints_t.size(); s++)
		{
			int j = missingpoints_t[s];
			Mat_d H_j = S_bar.col(j);

			Mat_d h_jreshape = fgReshapeForMatlab(V.col(j), 3);

			H_j.push_back(h_jreshape);
			Mat_d E_z_temp(1,1 ,1);
			E_z_temp.push_back(E_z.col(t));

			Mat_d S_tj = H_j*E_z_temp;

			Mat_d tr_trans;
			cv::transpose(Tr.row(t), tr_trans);
			Mat_d newf_t = RO[t].rowRange(0, 1)*S_tj + tr_trans;
			P_hat(cv::Range(t, t + T), cv::Range(j, j)) = newf_t;
		}
	}

	return P_hat;
}