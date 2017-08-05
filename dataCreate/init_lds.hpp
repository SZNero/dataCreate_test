#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp> 
#include <opencv2/ml/ml.hpp> 
#include <string>



std::tuple<Mat_d,Mat_d,Mat_d,Mat_d>
init_lds(Mat_d P, Mat_d S_bar, Mat_d V, Mat_d RR, Mat_d Tr, double sigma_sq)
{
	int K = V.rows / 3;
	int T = P.rows;
	int J = P.cols;
	std::vector<Mat_d> M(T);
	T = T / 2;
	Mat_d y = Mat_d::zeros(2 * J, T );
	Mat_d P_hat_t = Mat_d::zeros(2 * J, 1 );
	Mat_d M_t = Mat_d::zeros(2 * J, K );

	for (int t = 0; t < T; t++)
	{
		Mat_d Pt,Rt, temp_transpos,Pt_temp;

		Pt.push_back(P.row(t).t());
		Pt.push_back(P.row(t+T).t());

		Rt.push_back(RR.row(t));
		Rt.push_back(RR.row(t + T));

		temp_transpos = (Rt.row(0)*S_bar).t();
		temp_transpos.copyTo(P_hat_t.rowRange(0, J).col(0));
		temp_transpos = (Rt.row(1)*S_bar).t();
		temp_transpos.copyTo(P_hat_t.rowRange(J, P_hat_t.rows).col(0));

		for (int kk = 0; kk < K; kk++)
		{
			temp_transpos = (Rt.row(0)*V.rowRange(kk * 3, (kk + 1) * 3)).t();
			temp_transpos.copyTo(M_t.rowRange(0, J).col(kk));
			temp_transpos = (Rt.row(1)*V.rowRange(kk * 3, (kk + 1) * 3)).t();
			temp_transpos.copyTo(M_t.rowRange(J, M_t.rows).col(kk));
		}
		M[t] = M_t.clone();
		temp_transpos = Pt - P_hat_t;
		temp_transpos.copyTo(y.col(t));
	}

	Mat_d A = Mat_d::eye(2 * J,2*J )/sigma_sq;

	Mat_d temp1b(T,2 );
	temp1b.zeros(T,2);
	for (int t = 0; t < T; t++)
	{
		Mat_d temp1 = A*M[t];

		Mat_d temp2 = A - temp1*((Mat_d::eye(K, K ) + (M[t].t())*temp1).inv())*(temp1.t());

		Mat_d temp1b_t = (y.col(t).t())*temp2*M[t];
		temp1b_t.copyTo(temp1b.row(t));
	}
	Mat_d mu0In(temp1b.cols,1 );

	for (int rnum = 0; rnum < temp1b.cols; rnum++)
	{
		double sum = 0.0f;
		for (int cnum = 0; cnum < temp1b.rows; cnum++)
		{
			sum += temp1b.at<double>(cnum, rnum);
		}
		mu0In.at<double>(rnum, 0) = sum/temp1b.rows;
	}


	Mat_d Q;
	Mat_d Q_mean;
	cv::calcCovarMatrix(temp1b, Q, Q_mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
	Q = Q / temp1b.rows;
	Mat_d QIn = Q.clone();
	Mat_d t1 = temp1b.rowRange(0, T -1);
	Mat_d t2 = temp1b.rowRange(1, T);
	Mat_d phiIn = ((t1.t())*t1 + Q).inv()*(t1.t())*t2;
	Mat_d sigma0In = QIn.clone();

	return std::make_tuple(phiIn, mu0In, sigma0In, QIn);
}