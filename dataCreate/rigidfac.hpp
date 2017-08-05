#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/ml/ml.hpp> 
#include <string>
#include "fgMath.hpp"


Mat_d zt2(Mat_d i, Mat_d j)
{
	int D = i.rows > i.cols ? i.rows : i.cols;
	Mat_d M;

	for (int x = 0; x < D; x++)
	{
		for (int y = x; y < D; y++)
		{
			if (x == y)
				M.push_back(i.at<double>(0, x) * j.at<double>(0, y));
			else
				M.push_back(i.at<double>(0, x) *j.at<double>(0, y) + i.at<double>(0, y) * j.at<double>(0, x));
		}
	}
	M = M.t();
	return M;
}
/*
% Build matrix Q such that Q * v = [1,...,1,0,...,0] where v is a six
% element vector containg all six distinct elements of the Matrix C
*/
Mat_d findG(Mat_d Rhat)
{
	Mat_d G;

	int F = Rhat.rows;
	int D = Rhat.cols;
	F = F / 2;

	//%clear Q 
	double zero = 0.0;
	Mat_d Q(3*F, 2*D);
	Q = cv::Mat::zeros(3 * F,2* D, CV_64F);
	int g = 0, h = 0;
	for (int f = 0; f < F; f++)
	{
		g = f + F;
		h = g + F;
		zt2(Rhat.row(f), Rhat.row(f)).copyTo(Q.row(f));
		zt2(Rhat.row(g), Rhat.row(g)).copyTo(Q.row(g));
		zt2(Rhat.row(f), Rhat.row(g)).copyTo(Q.row(h));

	}

	//% Solve for v

	Mat_d rhs;
	rhs.push_back(Mat_d::ones(2 * F, 1));
	rhs.push_back(Mat_d::zeros(F, 1));


	//Mat_d tempInv = rhs.inv();
	//uchar* v_values = v.ptr(0);
	//% C is a symmetric 3x3 matrix such that C = G * transpose(G)
	int n = 0;
	Mat_d C(D, D);
	Mat_d v;
	Mat_d temp_v = Mat_d::zeros(Q.rows, Q.rows);
	Q.copyTo(temp_v(cv::Range(0, Q.rows), cv::Range(0, Q.cols)));
	cv::solve(Q, rhs, v, cv::DECOMP_SVD);

	for (int x = 0; x < D; x++)
	{
		for (int y = x; y < D; y++)
		{
			C.at<double>(x, y) = v.at<double>(n, 0);
			C.at<double>(y, x) = v.at<double>(n, 0);
			n += 1;
		}
	}

	Mat_d eValuesMat;
	Mat_d eVectorsMat;
	cv::eigen(C, eValuesMat, eVectorsMat);
	int iNonZero = cv::countNonZero(eValuesMat);
	int negetiveNonZeroCount = 0;
	for (int icount = 0; icount < eValuesMat.rows; icount++)
	{
		if (eValuesMat.at<double>(icount, 0) < 0.0)
			negetiveNonZeroCount++;
	}
	if (negetiveNonZeroCount > 0)
	{
		Mat_d u, s, v; // a,b,c
		cv::SVD::compute(C, s, u, v, cv::SVD::FULL_UV);
		v = v.t();
		Mat_d temp_S = cv::Mat::diag(s);
		//Mat_d after_sqrt;
		cv::sqrt(temp_S, temp_S);
		G = u*temp_S;

		Mat_d cc = G*G.t();
		Mat_d cc_pow;

		Mat_d temp_CC, cc_inline, C_inline;

		for (int i = 0; i < cc.cols; i++)
		{
			cc_inline.push_back(cc.col(i));
		}
		for (int i = 0; i < C.cols; i++)
		{
			C_inline.push_back(C.col(i));
		}

		temp_CC = cc_inline - C_inline;

		cv::pow(temp_CC, 2, cc_pow);
		double err = cv::sum(cc_pow)[0];
		if ( err > 0.03)
		{
			G = Mat_d::zeros(G.rows, G.cols);
		}
	}
	else {
		//cv::sqrt(C, G);
		///matlab sqrtm function;
		G = fgSqrtm(C);
	}
	return G;
}

/*
% [R, Tr, S] = rigidfac(P, MD)
%
% Computes rank 3 factorization: P = R*S + Tr
*/

std::tuple<Mat_d, Mat_d, Mat_d>
rigidfac(Mat_d P, Mat_d MD)
{
	Mat_d pnew = P;
	cv::Size pnew_size(pnew.rows / 2, pnew.cols);
	int T = pnew_size.width;
	int J = pnew_size.height;
	//if there is missing data, then it uses an iterative solution to get a rough initialization for the missing points
	double md_sum = 0.0;
	md_sum = cv::sum(MD)[0];
	int numIter = 0;
	Mat_d ind;
	if (md_sum > 0)
	{
		int count = cv::countNonZero(MD == 1);
		numIter = 10;
	}
	else {
		numIter = 1;
	}


	Mat_d Rhat;
	Mat_d Shat;

	int r = 3;

	Mat_d Tr;
	for (int iter = 0; iter < numIter; iter++)
	{
		Tr = pnew*Mat_d::ones(J, 1) / J;
		Mat_d pnew_c = pnew - Tr*Mat_d::ones(1, J);

		//奇异值分解 u   vectors, s values , v = vT   so, it must to be transpose
		Mat_d u, s, v; // a,b,c
		cv::SVD::compute(pnew_c, s, u, v, cv::SVD::FULL_UV);
		v = v.t();

		//利用对角元素 构建对角矩阵
		s = cv::Mat::diag(s);

		//Mat_d tempS =Mat_d::eye(r, r );
		//for (int i = 0; i < r; i++)
		//{
		//	tempS.at<double>(i, i) = s.at<double>(i);
		//}
		//s = tempS;
		Mat_d smallb = s(cv::Range(0, r), cv::Range(0, r));
		//std::cout << smallb << std::endl;
		Mat_d sqrtb = matSqrt(smallb);
		//std::cout << sqrtb << std::endl;
		Rhat = u.colRange(0, r)*sqrtb;
		Shat = sqrtb*v.colRange(0, r).t();

		Mat_d P_approx = Rhat*Shat + Tr*Mat_d::ones(1, J);
		//std::cout << P_approx << std::endl;
		pnew = P_approx;
		//std::cout << pnew << std::endl;
	}

	Mat_d G = findG(Rhat);
	Mat_d R = Rhat*G;
	Mat_d S = G.inv()*Shat;

	return std::make_tuple(R, Tr, S);
}