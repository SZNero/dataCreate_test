#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp> 
#include <opencv2/ml/ml.hpp> 
#include <string>
#include "rigidfac.hpp"
#include "apprRot.hpp"
#include "init_SB.hpp"
#include "mstep_update_noisevar.hpp"
#include "init_lds.hpp"
#include "estep_lds_compute_Z_distr.hpp"
#include "mstep_lds_update.hpp"
#include "mstep_update_missingdata.hpp"
#include "mstep_update_rotation.hpp"
#include "mstep_update_transl.hpp"
/*
%  INPUT:
%
%  P           - (2*T) x J tracking matrix:          P([t t+T],:) contains the 2D projections of the J points at time t
%  MD          - T x J missing data binary matrix:   MD(t, j)=1 if no valid data is available for point j at time t, 0 otherwise
%  K           - number of deformation basis
%  use_lds     - set to 1 to model deformations using a linear dynamical system; set to 0 otherwise
%  tol         - termination tolerance (proportional change in likelihood)
%  max_em_iter - maximum number of EM iterations
%
%
%  OUTPUT:
%
%  P3          - (3*T) x J 3D-motion matrix:                    ( P3([t t+T t+2*T],:) contains the 3D coordinates of the J points at time t )
%  S_bar       - shape average:            3 x J matrix
%  V           - deformation shapes:       (3*K) x J matrix     ( V((n-1)*3+[1:3],:) contains the n-th deformation basis )
%  RO          - rotation:                 cell array           ( RO{t} gives the rotation matrix at time t )
%  Tr          - translation:              T x 2 matrix
%  Z           - deformation weights:      T x K matrix
%  sigma_sq    - variance of the noise in feature position
%  phi         - LDS transition matrix
%  Q           - LDS state noise matrix
%  mu0         - initial state mean
%  sigma0      - initial state variance
*/

Mat_d
em_sfm(Mat_d P, Mat_d MD, int K, int use_lds, double tol, int max_em_iter)
{
	
	int T = MD.rows;
	int J = MD.cols;
	Mat_d Tr( T, 2 );
	Mat_d P3 = Mat_d::zeros(3 * T, J );
	
	double modNum = 1.0;
	if (std::modf((double)P.rows, &modNum) != 0)
		return P3;
	if ((P.rows / 2 != MD.rows) || (P.cols != MD.cols))
		return P3;
	if (std::modf((double)K, &modNum) != 0)
		return P3;

	int r = 3 * (K + 1); //motion rank
	Mat_d P_hat = P.clone();

	Mat_d R_init;
	Mat_d Trvect;
	Mat_d S_bar;
	/*
	% if any of the points are missing, P_hat will be updated during the M - step

	% uses rank 3 factorization to get a first initialization for rotation and S_bar
	*/
	std::tie(R_init,Trvect,S_bar) = rigidfac(P_hat.clone(), MD.clone());


	Trvect.rowRange(0, T).copyTo(Tr.col(0));
	Trvect.rowRange(T, 2*T).copyTo(Tr.col(1));

	Mat_d  R = Mat_d::zeros(2 * T, 3 );
	std::vector<Mat_d> RO;
	RO.resize(T);
	for (int t = 0; t < T; t++)
	{
		Mat_d Ru = R_init.row(t);
		Mat_d Rv = R_init.row(T + t);
		Mat_d Rz = Ru.cross(Rv);

		Mat_d temp;
		temp.push_back(Ru);
		temp.push_back(Rv);
		temp.push_back(Rz);
		double det_temp = cv::determinant(temp);
		if (cv::determinant(temp) < 0)
		{
			Rz = -Rz;
			Rz.copyTo(temp.row(2));
		}
		Mat_d RO_approx = apprRot(temp);

		RO[t] = RO_approx;
		RO_approx.row(0).copyTo(R.row(t));
		RO_approx.row(1).copyTo(R.row(t+T));
	}
	/*
	% given the initial estimates of rotation, translation and shape average, it initializes
	% deformation shapes and weights through LSQ minimization of the reprojection error
	*/
	Mat_d V, Z;

	std::tie(V, Z) = init_SB(P_hat, Tr, R, S_bar, K);

	Mat_d E_zz_init, E_zz_init_temp;
	Mat_d meanZ;

	cv::calcCovarMatrix(Z, E_zz_init_temp, meanZ , CV_COVAR_NORMAL | CV_COVAR_ROWS);
	E_zz_init_temp = E_zz_init_temp / E_zz_init_temp.rows;
	E_zz_init = cv::repeat(E_zz_init_temp, T,1);

	double sigma_sq = mstep_update_noisevar(P_hat, S_bar, V, Z.t(), E_zz_init, RO, Tr);

	Mat_d phi, sigma0,Q;
	Mat_d mu0;
	Mat_d mu0temp_mlu;
	if (use_lds)
		std::tie(phi, mu0, sigma0, Q) = init_lds(P_hat, S_bar, V, R, Tr, sigma_sq);
	else
	{
		phi = Mat_d::zeros(phi.rows, phi.cols );
		mu0 = Mat_d::zeros(mu0.rows, mu0.cols );
		sigma0 = Mat_d::zeros(sigma0.rows, sigma0.cols );
		Q = Mat_d::zeros(Q.rows, Q.cols );
	}
	double loglik = 0.0;
	double annealing_const = 60.0f;
	double max_anneal_iter = round(max_em_iter / 2);

	for (int em_iter = 0; em_iter < max_em_iter; em_iter++)
	{
		Mat_d E_z, E_zz;
		if (use_lds)
		{
			Mat_d y, xt_n, xt_t1;
			std::vector<Mat_d>  M, Pt_n, Ptt1_n, Pt_t1;
			std::tie(E_z, E_zz, y, M, xt_n, Pt_n, Ptt1_n, xt_t1, Pt_t1) = estep_lds_compute_Z_distr(P_hat.clone(), S_bar.clone(), V.clone(), R, Tr, phi, mu0, sigma0, Q, sigma_sq);
			std::tie(phi, Q, sigma_sq, mu0temp_mlu, sigma0) = mstep_lds_update(y, M, xt_n, Pt_n, Ptt1_n);
		}
		/*else {
			std::tie(S_bar, V) = estep_compute_Z_distr(P_hat, S_bar, V, R, Tr, sigma_sq);
		}*/
		
		Z = E_z.t();
		Mat_d MDforSum;
		for (int i = 0; i < MD.cols; i++)
		{
			MDforSum.push_back(MD.col(i));
		}
		if (cv::sum(MDforSum)[0] > 0)
			P_hat = mstep_update_missingdata(P_hat.clone(), MD, S_bar.clone(), V.clone(), E_z.clone(), RO, Tr);

		std::tie(RO, R) = mstep_update_rotation(P_hat.clone(), S_bar.clone(), V.clone(), E_z.clone(), E_zz, RO, Tr);
		Tr = mstep_update_transl(P_hat.clone(), S_bar.clone(), V.clone(), E_z.clone(), RO);

		// case Two
		//if (!use_lds)
		//{
		//	sigma_sq = mstep_update_noisevar(P_hat, S_bar, V, E_z, E_zz, RO, Tr);
		//	if (em_iter < max_anneal_iter)
		//		sigma_sq = sigma_sq*(1 + annealing_const*(1 - em_iter / max_anneal_iter));
		//	oldloglik = loglik;

		//}
		printf("Iteration %d/%d\n", em_iter, max_em_iter);
	}

	//Mat_d P3 = Mat_d::zeros(3 * T, J );
	for (int t = 0; t < T; t++)
	{
		Mat_d z_t = Z.row(t);
		Mat_d Rf;
		Rf.push_back(R.row(t));
		Rf.push_back(R.row(t + T));
		Mat_d S;
		S = S_bar;
		for (int kk = 0; kk < K; kk++)
		{
			S = S + z_t.at<double>(0, kk)*V.rowRange((kk) * 3, (kk+1) * 3);
		}
		S = RO[t] * S;
		cv::Scalar mean_s = cv::mean(S.row(2));
		double buffer_temp[3] = { Tr.at<double>(t,0) , Tr.at<double>(t,1), -mean_s[0] };
		Mat_d buffer_mat(1, 3 , buffer_temp);
		Mat_d right_s = (S + buffer_mat.t()*Mat_d::ones(1, J ));
		right_s.row(0).copyTo(P3.row(t));
		right_s.row(1).copyTo(P3.row(t+T));
		right_s.row(2).copyTo(P3.row(t+2*T));
	}

	return P3;
}