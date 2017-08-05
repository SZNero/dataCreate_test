#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp> 
#include <opencv2/ml/ml.hpp> 

Mat_d computeH(Mat_d Xc, Mat_d Yc, Mat_d E_z, Mat_d E_zz, Mat_d RR)
{
	printf("%s","You are running the Matlab version of function ''computeH''. This program will run very slowly.... \nI recommend that you try to compile the CMEX code on your platform using command ''mex computeH.c'' (''mex computeH.c -l matlb'' under Unix)\n\n");

	int K = E_z.rows;
	int J = Xc.cols;
	int T = Xc.rows;

	Mat_d KK2 = Mat_d::zeros(3 * J*(K + 1), 2 * J*(K + 1) );
	Mat_d KK3 = Mat_d::zeros(3 * J, K + 1 );

	for (int t = 0; t < T; t++)
	{
		Mat_d P_t;
		P_t.push_back(Xc.row(t));
		P_t.push_back(Yc.row(t));

		Mat_d zz_hat_t(1+E_z.rows,1+E_z.rows ,cv::Scalar(0));
		Mat_d temp;
		temp.push_back(1);
		temp.push_back(E_z.col(t));
		Mat_d trans_tmp =  temp.t();
		trans_tmp.copyTo(zz_hat_t.row(0));
		E_z.col(t).copyTo(zz_hat_t(cv::Range(1, zz_hat_t.rows - 1), cv::Range(0, 0)));
		E_zz.rowRange((t - 1)*K + 1, t*K).copyTo(zz_hat_t(cv::Range(1,zz_hat_t.rows-1),cv::Range(1,zz_hat_t.cols-1)));
		Mat_d R_t;
		R_t.push_back(RR.row(t));
		R_t.push_back(RR.row(t + T));

		int size_sparsemat[2] = { J,J };
		cv::SparseMat sparsemat(2, size_sparsemat );
		int idx[2] = {0};
		for (int i = 0; i < J; i++)
		{
			idx[0] = i;
			for (int k = 0; k < J; k++)
			{
				idx[1] = k;
				sparsemat.ref<double>(idx) = 1.f;
			}
		}

		Mat_d R_t_kron(size_sparsemat[0]*R_t.rows, size_sparsemat[1]*R_t.cols );

		int R_T_idx_r = 0,R_T_idx_c = 0;

		int i = 0, j = 0, k = 0, m = 0, n = 0;

		for (R_T_idx_r = 0; R_T_idx_r < R_t_kron.rows; R_T_idx_r++)
		{
			for (R_T_idx_c = 0; R_T_idx_c < R_t_kron.cols; R_T_idx_c++)
			{
				int index[2] = { 0,0 };
				index[0] = i;
				index[1] = j;
				R_t_kron.at<double>(R_T_idx_r, R_T_idx_c) = sparsemat.ref<double>(index)*R_t.at<double>(k, m);

				 j++; m++;
				if (i >= size_sparsemat[0])
					i = 0;

				if (m >= R_t.cols)
					m = 0;
			}
			i++; k++;
			if (j >= size_sparsemat[1])
				j = 0;
			if (k >= R_t.rows)
				k = 0;
		}

		Mat_d KK1_trans = R_t_kron.t()*R_t_kron;

	

		Mat_d zzKK_kron(zz_hat_t.cols*KK1_trans.rows, zz_hat_t.rows*KK1_trans.cols );
		Mat_d zz_hat_t_trans = zz_hat_t.t();
		i = 0, j = 0, k = 0, m = 0, n = 0;
		int zzKK_kron_r = 0, zzKK_kron_c = 0;
		for (zzKK_kron_r = 0; zzKK_kron_r < zzKK_kron.rows; zzKK_kron_r++)
		{
			for (zzKK_kron_c = 0; zzKK_kron_c < zzKK_kron.cols; zzKK_kron_c++)
			{
				int index[2] = { 0,0 };
				index[0] = i;
				index[1] = j;
				zzKK_kron.at<double>(R_T_idx_r, R_T_idx_c) = zz_hat_t_trans.at<double>(i,j)*KK1_trans.at<double>(k, m);

				j++; m++;
				if (i >= zz_hat_t_trans.rows)
					i = 0;

				if (m >= KK1_trans.cols)
					m = 0;
			}
			i++; k++;
			if (j >= zz_hat_t_trans.col)
				j = 0;
			if (k >= KK1_trans.rows)
				k = 0;
		}
		Mat_d P_t_allinCol;
		for (int z = 0; z < P_t.cols; z++)
		{
			P_t_allinCol.push_back(P_t.col(z));
		}
		KK2 = KK2 + zzKK_kron;

		Mat_d E_z_temp;
		E_z_temp.push_back(1);
		E_z_temp.push_back(E_z.col(t));
		E_z_temp = E_z_temp.t();
		KK3 = KK3 + R_t_kron.t()*P_t_allinCol*E_z_temp;
	}
	Mat_d pinv2;
	cv::invert(KK2, pinv2, cv::DECOMP_SVD);

	Mat_d KK3_allincol;
	for (int z = 0; z < KK3.cols; z++)
	{
		KK3_allincol.push_back(KK3.col(z));
	}
	Mat_d vecH_hat = pinv2*KK3_allincol;
	return vecH_hat;
}