#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp> 
#include <opencv2/ml/ml.hpp> 
#include <string>

double mstep_update_noisevar(Mat_d P, Mat_d S_bar, Mat_d V, Mat_d E_z, Mat_d& E_zz, std::vector<Mat_d> RO, Mat_d Tr)
{
	int K = E_z.rows;
	int T = E_z.cols;
	int J = S_bar.cols;

	Mat_d M_t = Mat_d::zeros(2 * J, K );
	double sigma_sq = 0.0;
	for (int t = 0; t < T; t++)
	{
		Mat_d R_t = RO[t].clone();

		//std::cout << "R_t\n" << R_t << std::endl;

		Mat_d Sdef = S_bar.clone();

		for (int kk = 0; kk < K; kk++)
		{
			Sdef = Sdef + E_z.at<double>(kk, t)*V.rowRange((kk) * 3, (kk) * 3 + 3);

			Mat_d t_temp;

			t_temp = (R_t.row(0)*V.rowRange((kk) * 3, (kk) * 3 + 3)).t();
			t_temp.copyTo(M_t.rowRange(0, J).col(kk));

			t_temp = (R_t.row(1)*V.rowRange((kk) * 3, (kk) * 3 + 3)).t();
			t_temp.copyTo(M_t.rowRange(J, M_t.rows).col(kk));
		}

		Mat_d f_bar_t = R_t.rowRange(0,2)*S_bar;
		Mat_d temp_transpose;
		temp_transpose.push_back(f_bar_t.row(0).t());
		temp_transpose.push_back(f_bar_t.row(1).t());
		f_bar_t = temp_transpose;

		temp_transpose = P.row(t).t();
		temp_transpose.push_back(P.row(t + T).t());

		Mat_d f_t = temp_transpose;
		Mat_d TestNouse = f_t;
//		std::cout << "f_t.type()" << f_t.type() << std::endl;
//		std::cout << "TestNouse" << TestNouse << std::endl;

		Mat_d t_vect_t;
		t_vect_t.push_back(Tr.at<double>(t, 0)*Mat_d::ones(J, 1 ));
		t_vect_t.push_back(Tr.at<double>(t, 1)*Mat_d::ones(J, 1 ));

	//	std::cout << "f_t.type()" << f_bar_t.type() << "  "<<t_vect_t.type() << std::endl;

		Mat_d tempTset = f_t - f_bar_t;
		Mat_d tempS = (f_t - f_bar_t) - t_vect_t;
//		std::cout << "f_t\n" << f_t << std::endl;
//		std::cout << "f_t\n" << f_bar_t << std::endl;
//		std::cout << "f_t\n" << t_vect_t << std::endl;
//		std::cout << "resualt\n" << tempS << std::endl;
		Mat_d tempResualt = tempS.t()*tempS;
//		std::cout << "resualt2\n" << tempResualt << std::endl;


		Mat_d s1 = tempResualt;
		Mat_d s2 = 2 * (tempS.t())*M_t*E_z.col(t);


		Mat_d tempE_zz = E_zz.rowRange(t*K, (t + 1)*K);
//		std::cout << "M_t.t()\n" << M_t.t() << std::endl;
//		std::cout << "M_t\n" << M_t << std::endl;
//		std::cout << "tempE_zz\n" << tempE_zz << std::endl;

		cv::Scalar traceS3 = cv::trace((M_t.t())*M_t*E_zz.rowRange(t*K, (t + 1)*K));

		sigma_sq = sigma_sq + (s1.at<double>(0,0) - s2.at<double>(0,0) + traceS3[0]);
	}
	sigma_sq = sigma_sq / (2 * J*T);
	return sigma_sq;
}