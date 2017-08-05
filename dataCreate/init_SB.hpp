#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp> 
#include <opencv2/ml/ml.hpp> 
#include <string>

void OutputMatrix(cv::Mat_<double> mat)
{
	for (int i = 0; i < mat.rows; i++)
	{
		for (int j = 0; j < mat.cols; j++)
		{
			//fprintfk
			printf("%17.16f\t", mat.at<double>(i, j));
		}
		printf("\n");
	}
}

std::tuple<Mat_d,Mat_d> init_SB(Mat_d P, Mat_d Tr, Mat_d RR, Mat_d S_bar, int K)
{
	int T = P.rows;
	int J = P.cols;

	T = T / 2;

	Mat_d V = Mat_d::zeros(3 * K, J );
	Mat_d Z = Mat_d::zeros(T, K );

	Mat_d Tr_temp;
	Tr_temp.push_back(Tr.col(0));
	Tr_temp.push_back(Tr.col(1));

	

	Mat_d W_tilda = P - RR*S_bar - Tr_temp*Mat_d::ones(1, J );
	//OutputMatrix(P);
	//OutputMatrix(RR);
	//OutputMatrix(S_bar);
	//OutputMatrix(Tr_temp);
	



	for (int kk = 0; kk < K; kk++)
	{
		Mat_d V_kk = Mat_d::zeros(T, 3 * J );
		for (int t = 0; t < T; t++)
		{
			Mat_d temp_RR = RR.row(t);
			temp_RR.push_back(RR.row(t + T));

			Mat_d temp_RR_pinv;
			cv::invert(temp_RR, temp_RR_pinv, cv::DECOMP_SVD);

			Mat_d temp_W_tilda = W_tilda.row(t);
			temp_W_tilda.push_back(W_tilda.row(T + t));

			Mat_d V_kk_t = temp_RR_pinv*temp_W_tilda;

			Mat_d V_kk_t_allincol;
			for (int v_kkcount = 0; v_kkcount < V_kk_t.cols; v_kkcount++)
			{
				V_kk_t_allincol.push_back(V_kk_t.col(v_kkcount));
			}

			Mat_d t_v_kkallincol = V_kk_t_allincol.t();
			t_v_kkallincol.copyTo(V_kk.row(t));
		}

		Mat_d a, b, c;
		cv::SVD::compute(V_kk, b, a, c,cv::SVD::FULL_UV);
		c = c.t();

		double sqrtb = cv::sqrt(b.at<double>(0, 0));
	
		Mat_d tempA = a.col(0)*sqrtb;
		tempA.copyTo(Z.col(kk));

		Mat_d new_V_kk = sqrtb*(c.col(0).t());
		Mat_d after_reshape = fgReshapeForMatlab(new_V_kk, 3);
		//Mat_d after_reshape = new_V_kk.reshape(0, 3).clone();
		after_reshape.copyTo(V.rowRange((kk) * 3, (kk + 1) * 3));
		//W_tilda([t t+T], :) = W_tilda([t t+T], :) - RR([t t+T], :)*(Z(t,kk)*V((kk-1)*3+1:kk*3, :));
		for (int t = 0; t < T; t++)
		{
			W_tilda.row(t) = W_tilda.row(t) - RR.row(t)
				*(Z.at<double>(t,kk)*V.rowRange((kk)*3,(kk+1)*3));
			W_tilda.row(t + T) = W_tilda.row(t + T) - RR.row(T + t)
				*(Z.at<double>(t, kk)*V.rowRange((kk) * 3,( kk+1) * 3));
		}
	}

	return std::make_tuple(V,Z);
}