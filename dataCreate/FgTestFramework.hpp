#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp> 
#include <opencv2/ml/ml.hpp> 
#include <string>
#include "em_sfm.hpp"

#include <fstream>
#include <time.h>
using namespace std;
using namespace cv;

//opencv   mat   range   左开右闭，传值与传内存

namespace Fg_NRSFM
{

	class testFrameWorkInstance {
	public:
		void Load2DMat(Mat_d datas)
		{
			mP3_gt = datas.clone();
			mP3_gt_size = std::make_tuple(mP3_gt.rows / 2, mP3_gt.cols);

		}
		void Running()
		{
			Mat_d  temp_For_test;

			int T = std::get<0>(mP3_gt_size);
			int J = std::get<1>(mP3_gt_size);

			Mat_d p2_obs = mP3_gt.clone();


			/*Mat_d meanMat = Mat_d::ones(T, 1 );

			cv::reduce(mP3_gt.rowRange(2 * T, 3 * T), meanMat, 1, CV_REDUCE_AVG );
			Mat_d onesTemp = Mat_d::ones(1, J );
			Mat_d rowRangeTemp = mP3_gt.rowRange(2 * T , 3 * T);
			Mat_d Zcoords_gt = rowRangeTemp - meanMat*onesTemp;
			Mat_d zdist,zdist_max,zdist_min;

			cv::reduce(Zcoords_gt, zdist_max, 2, CV_REDUCE_MAX );
			cv::reduce(Zcoords_gt, zdist_min, 2, CV_REDUCE_MIN );

			zdist = zdist_max - zdist_min;*/
			Mat_d MD = Mat_d::zeros(T, J);

			//em_sfm
			Mat_d P3;

			clock_t testTime = clock();

			P3 = em_sfm(p2_obs, MD, m_K, m_use_lds, m_tol, m_max_em_iter);
			std::cout << "training time: " << clock() - testTime << std::endl;
			/*Mat_d average_P3;
			cv::reduce(P3.rowRange(2 * T, 3 * T), average_P3, 2, CV_REDUCE_AVG );
			Mat_d Zcoords_em = P3.rowRange(2 * T, 3 * T ) - average_P3*Mat_d::ones(1, J );

			Mat_d tempResualt = cv::abs(Zcoords_em - Zcoords_gt);

			Mat_d Zerror1, Zerror2;

			cv::reduce(tempResualt, tempResualt, 2, CV_REDUCE_AVG );
			cv::reduce(tempResualt/zdist, Zerror1, 2, CV_REDUCE_AVG );
			tempResualt = abs(-Zcoords_em - Zcoords_gt);
			cv::reduce(tempResualt, tempResualt, 2, CV_REDUCE_AVG );
			cv::reduce(tempResualt / zdist, Zerror2, 2, CV_REDUCE_AVG );

			double avg_zerror = 0.0;

			if (Zerror2.at<double>(0, 0) < Zerror1.at<double>(0, 0))
			{
				avg_zerror = 100 * Zerror2.at<double>(0, 0);
				cv::reduce(P3.rowRange(2*T,3*T-1), tempResualt, 2, CV_REDUCE_AVG );
				Mat_d mResualt = -(P3.rowRange(2 * T, 3 * T - 1) - tempResualt*Mat_d::ones(1, J ));
				mResualt.copyTo(P3.rowRange(2 * T, 3 * T - 1));
			}
			else {
				avg_zerror = 100 * Zerror1.at<double>(0, 0);
				cv::reduce(P3.rowRange(2 * T, 3 * T - 1), tempResualt, 2, CV_REDUCE_AVG );
				Mat_d mResualt =  P3.rowRange(2 * T, 3 * T - 1) - tempResualt*Mat_d::ones(1, J );
				mResualt.copyTo(P3.rowRange(2 * T, 3 * T - 1));
			}*/

			//std::cout << "\nP3: \n" << P3 << std::endl;
			OutputObjFile(P3);
		}
	private:
		Mat mP3_gt;
		std::tuple<int, int> mP3_gt_size;

		int m_use_lds = 3;
		int m_max_em_iter = 5;
		double m_tol = 0.0001;
		int m_K = 2;

	protected:
		void LogOutMat(Mat_d & mat)
		{
			std::cout << mat << std::endl;
		}
		void OutputObjFile(Mat_d mat)
		{
			if (mat.rows % 3 != 0)
			{
				std::cout << "out obj file failed!" << std::endl;
				return;
			}

			int rows = mat.rows;
			int cols = mat.cols;
			int T = mat.rows / 3;


			for (int r = 0; r < T; r++)
			{
				ofstream fd;

				char fileN[64] = { 0 };
				sprintf_s(fileN, "./OutPutFile/resualtObj_%d.OBJ", r + 1);

				fd.open(fileN);
				if (fd.is_open())
				{
					for (int i = 0; i < cols; i++)
					{
						fd << "v ";
						fd << mat.at<double>(r, i) << " ";
						fd << mat.at<double>(r + T, i) << " ";
						fd << mat.at<double>(r + 2 * T, i) << "\n";
					}
					fd << "f ";
					for (int i = 0; i < cols; i++)
					{
						fd << i + 1;
						fd << " ";
					}
				}
				fd.close();
			}
		}
	};

}

