#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp> 
#include <opencv2/ml/ml.hpp> 
#include <string>
#include "fgMath.hpp"
std::tuple<std::vector<Mat_d>, Mat_d>
mstep_update_rotation(Mat_d P, Mat_d S_bar, Mat_d V, Mat_d E_z, Mat_d E_zz, std::vector<Mat_d> RO, Mat_d Tr)
{
	double tw_step = 0.3;
	int K = E_z.rows;
	int T = E_z.cols;
	int J = S_bar.cols;

	Mat_d Tr_allInCol;
	for (int i = 0; i < Tr.cols; i++)
	{
		Tr_allInCol.push_back(Tr.col(i));
	}

	Mat_d Pc = P - Tr_allInCol*Mat_d::ones(1, J );

	Mat_d newRR = Mat_d::zeros(2 * T, 3 );
	std::vector<Mat_d> newRO;
	
	for(int i= 0;i < RO.size();i++)
		newRO.push_back(RO[i].clone());

	for (int iter = 0; iter < 1; iter++)
	{
		for (int t = 0; t < T; t++)
		{
			Mat_d A = Mat_d::zeros(3, 3);
			Mat_d B = Mat_d::zeros(2, 3);

			Mat_d zz_hat_t(E_z.rows+1, E_z.rows+1);
			zz_hat_t.at<double>(0, 0) = 1.0f;
			Mat_d col_transpose = E_z.col(t).t();
			col_transpose.copyTo(zz_hat_t.row(0).colRange(1, 3));
			E_z.col(t).copyTo(zz_hat_t.col(0).rowRange(1, 3));
			E_zz.rowRange((t)*K, (t + 1)*K).copyTo(zz_hat_t.rowRange(1, 3).colRange(1, 3));
			//E_z.col(t).copyTo(zz_hat_t(cv::Range(1, zz_hat_t.rows - 1), cv::Range(0, 0)));
			//E_zz(cv::Range((t - 1)*K + 1, t*K), cv::Range::all()).copyTo(zz_hat_t(cv::Range(1, zz_hat_t.rows - 1), cv::Range(1, zz_hat_t.cols - 1)));



			for (int j = 0; j < J; j++)
			{
				Mat_d v_reshape = fgReshapeForMatlab(V.col(j), 3);

				/*Mat_d for_reshape = V.col(j).clone();*/

				//Mat_d v_reshape = reshapeMat;
				Mat_d H_j(S_bar.col(j).rows, V.col(j).rows / 3 + 1);
				S_bar.col(j).copyTo(H_j.col(0));
				for (int colvr = 0; colvr < v_reshape.cols; colvr++)
					v_reshape.col(colvr).copyTo(H_j.col(colvr + 1));

				//	H_j.push_back(v_reshape);

		/*			Mat_d H_j_tm;
					cv::transpose(H_j, H_j_tm);*/
				A = A + H_j*zz_hat_t*(H_j.t());
				//Mat_d transpose_mat, tm1;
				//cv::transpose(E_z.col(t), transpose_mat);
				//cv::transpose(H_j, tm1);
				Mat_d Pc_mtemp;
				Pc_mtemp.push_back(Pc.at<double>(t, j));
				Pc_mtemp.push_back(Pc.at<double>(t + T, j));

				Mat_d E_z_tm1;
				E_z_tm1.push_back((double)1.0);
				E_z_tm1.push_back(E_z.col(t));
				//cv::transpose(E_z_tm1, E_z_tm1);

				B = B + Pc_mtemp*(E_z_tm1.t())*(H_j.t());
			}
			Mat_d oldRO_t = RO[t].clone();

			Mat_d C = oldRO_t*A;
			Mat_d D = B - oldRO_t.rowRange(0, 2)*A;
			//% now we solve the system : [1 0 0; 0 1 0] * twist*C = D
			double cc_buff[6][3] = {
				{0 ,C.at<double>(2,0), -C.at<double>(1,0)},
				{ -C.at<double>(2,0) ,0 ,C.at<double>(0,0) },
				{ 0 ,  C.at<double>(2,1) ,-C.at<double>(1,1) },
				{ -C.at<double>(2,1), 0, C.at<double>(0,1) },
				{0,   C.at<double>(2,2), -C.at<double>(1,2)},
				{ -C.at<double>(2,2), 0, C.at<double>(0,2) }
			};
			int size_cc[2] = { 6,3 };
			Mat_d CC(2, (int *)size_cc, (double*)cc_buff);

			Mat_d DD;
			for (int icol = 0; icol < D.cols; icol++)
			{
				DD.push_back(D.col(icol));
			}
			Mat_d pinv_CC;
			cv::invert(CC, pinv_CC, cv::DECOMP_SVD);
			//% twist optimization
			Mat_d twist_vect = tw_step*pinv_CC*DD;

			double twh_buff[3][3] = {
				{0, -twist_vect.at<double>(2),twist_vect.at<double>(1)},
				{ twist_vect.at<double>(2),  0, -twist_vect.at<double>(0) },
				{ -twist_vect.at<double>(1), twist_vect.at<double>(0),  0 }
			};


			Mat_d twh(3, 3, (double*)twh_buff);
			//cv::exp() expm
			
			Mat_d dR = fgExpm(twh);
			Mat_d newRO_t = dR*oldRO_t;

			newRO[t] = newRO_t.clone();
			newRO_t.row(0).copyTo(newRR.row(t));
			newRO_t.row(1).copyTo(newRR.row(t + T));
		}
		for (int i = 0; i < RO.size(); i++)
			RO[i] = newRO[i].clone();
	}
	return std::make_tuple(newRO, newRR);

}