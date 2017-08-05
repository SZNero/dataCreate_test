#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp> 
#include <opencv2/ml/ml.hpp> 


#include <Eigen/Dense>
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;
using namespace std;


typedef cv::Mat_<double_t> Mat_d;



void fgEignByeglib(Mat_d mat, Mat_d& eigenValues, Mat_d& eigenVectors)
{
	Eigen::Matrix3d A(mat.rows,mat.cols);
	
	for (int i = 0; i < mat.rows; i++)
	{
		for (int j = 0; j < mat.cols; j++)
		{
			A(i, j) = mat.at<double>(i, j);
		}		
	}

	Eigen::EigenSolver<Matrix3d> es(A);

	Matrix3d D = es.pseudoEigenvalueMatrix();
	Matrix3d V = es.pseudoEigenvectors();

	Eigen::Index size = D.size();
	int r = static_cast<int>(D.rows());
	int c = static_cast<int>(D.cols());
	Mat_d rD(r, c,D.data());
	r = static_cast<int>(V.rows());
	c = static_cast<int>(V.cols());
	Mat_d rV(r, c,V.data());
	

	eigenValues = rD.clone();
	eigenVectors = rV.clone();

}

Mat_d fgExpm(Mat_d mat)
{
	Mat_d src = mat.clone();

	Mat_d D,V;

	fgEignByeglib(src,D,V);

	V = V.t();
	Mat_d tempD(D.rows, D.rows);

	Mat_d expD;
	
	cv::exp(D.diag(), expD);

	Mat_d tDiag = cv::Mat::diag(expD);

	Mat_d resualtMat = V * tDiag * V.inv();

	return resualtMat;
}

Mat_d fgSum(Mat_d mat)
{
	Mat_d temp(1, mat.cols );
	double sum = 0.0f;
	for (int i = 0; i < mat.cols; i++)
	{
		sum = 0.0f;
		for (int j = 0; j < mat.rows; j++)
		{
			sum += mat.at<double>(j, i);
		}
		temp.at<double>(0, i) = sum;
	}

	return temp;
}

Mat_d fgSumRow(Mat_d mat)
{
	Mat_d temp(mat.rows, 1 );
	double sum = 0.0f;
	for (int i = 0; i < mat.rows; i++)
	{
		sum = 0.0f;
		for (int j = 0; j < mat.cols; j++)
		{
			sum += mat.at<double>(i, j);
		}
		temp.at<double>(i, 0) = sum;
	}

	return temp;
}

std::vector<int> fgFindNonZeros(Mat_d mat)
{
	int R = mat.rows;
	int C = mat.cols;
	std::vector<int> vec;
	for (int i = 0; i < R; i++)
	{
		for (int j = 0; j < C; j++)
		{
			int p;
			if (mat.at<double>(i, j) > 0)
			{
				p = j*C + i;
				vec.push_back(p);
			}
		}
	}

	return vec;
}

void fgSetSubMatrix(Mat_d &drst, Mat_d &src, cv::Range row, cv::Range col)
{
	for (int i = row.start; i < row.end - row.start + 1; i++)
	{
		for (int j = col.start; j < col.end - col.start + 1; j++)
		{
			drst.at<double>(i, j) = src.at<double>(i - row.start, j - col.start);
		}
	}
}

void fgSetSubMatrix(Mat_d &drst, Mat_d &&src, cv::Range row, cv::Range col)
{
	for (int i = row.start; i < row.end - row.start + 1; i++)
	{
		for (int j = col.start; j < col.end - col.start + 1; j++)
		{
			drst.at<double>(i, j) = src.at<double>(i - row.start, j - col.start);
		}
	}
}

Mat_d fgReshapeForMatlab(Mat_d &src, int rows)
{
	int cols = src.rows*src.cols / rows;
	Mat_d temp(rows, cols );
	int k = 0;
	for (int i = 0; i < cols; i++)
	{
		for (int j = 0; j < rows; j++)
		{
			temp.at<double>(j, i) = src.at<double>(k);
			k++;
		}
	}
	return temp;
}

Mat_d fgSqrtm(Mat_d &src)
{
	Mat_d drst;
	Mat_d u, s, v; // a,b,c
	cv::SVD::compute(src, s, u, v, cv::SVD::FULL_UV);
	v = v.t();


	int size = s.rows;
	Mat_d temp_zeros = Mat_d::zeros(size, size);
	for (int i = 0; i < size; i++)
	{
		temp_zeros.at<double>(i, i) = s.at<double>(i, 0);
	}
	Mat_d temp_sqrt;
	cv::sqrt(temp_zeros, temp_sqrt);
	u = u(cv::Range(0, size), cv::Range(0, size));
	drst = u*temp_sqrt*u.inv();

	return drst;
}

void nonZerofind(const Mat_d& binary, std::vector<cv::Point> idx) {

	assert(binary.cols > 0 && binary.rows > 0 && binary.channels() == 1 && binary.depth() == CV_8U);
	const int M = binary.rows;
	const int N = binary.cols;
	for (int m = 0; m < M; ++m) {
		const char* bin_ptr = binary.ptr<char>(m);
		for (int n = 0; n < N; ++n) {
			if (bin_ptr[n] > 0) idx.push_back(cv::Point(n, m));
		}
	}
}

Mat_d matSqrt(Mat_d & mat)
{
	Mat_d tempMat = mat;

	if (tempMat.cols > 0 && tempMat.rows > 0)
		;
	else
		return tempMat;

	const int Cols = tempMat.cols;
	const int Rows = tempMat.rows;

	for (int i = 0; i < Rows; i++)
	{
		for (int j = 0; j < Cols; j++)
		{
			tempMat.at<double>(i, j) = cv::sqrt(tempMat.at<double>(i, j));
		}
	}
	return tempMat;
}



namespace MatFile {
	std::ofstream &operator<<(std::ofstream & Out, Mat_d & Obj)
	{
		Out << Obj.rows << " " << Obj.cols << std::endl;
		for (int32_t row = 0; row < Obj.rows; ++row)
		{
			for (int32_t col = 0; col < Obj.cols; ++col)
				Out << Obj(row, col) << " ";
			Out << std::endl;
		}

		return Out;
	}

	std::ifstream &operator>>(std::ifstream & In, Mat_d & Obj)
	{
		int32_t row = 0, col = 0;

		In >> row >> col;
		if (row == 0)
			return In;
		Obj = Mat_d(row, col, 0.0);

		for (int32_t row = 0; row < Obj.rows; ++row)
		{
			for (int32_t col = 0; col < Obj.cols; ++col)
			{
				In >> Obj(row, col);
			}
		}

		//std::getline()
		return In;
	}



}

