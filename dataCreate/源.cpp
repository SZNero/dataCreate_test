#include <iostream>

#include "glm\glm.hpp"
#include "glm\gtc\matrix_transform.hpp"

#include <fstream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>

#include "FgTestFramework.hpp"
#include "fgMath.hpp"
const char* image_path = "D:/Qt_vs_space/Nonrigid_structure_from_motion/dataCreate/dataCreate/1.png";
using namespace std;
using namespace cv;
void outResualtDataToCSV(const char* filename, std::vector<std::vector< glm::vec3>>& resualt_vals)
{


	std::ofstream file_fd;
	size_t size_res = resualt_vals.size();

	file_fd.open(filename);
	if (file_fd.is_open())
	{
		if (size_res > 0 && resualt_vals[0].size() > 0)
		{
			for (int j = 0; j < resualt_vals[0].size(); j++)
			{
				for (int i = 0; i < size_res; i++)
				{
					file_fd << resualt_vals[i][j].x << ",";
				}
				file_fd << "\n";
			}
			for (int j = 0; j < resualt_vals[0].size(); j++)
			{
				for (int i = 0; i < size_res; i++)
				{
					file_fd << resualt_vals[i][j].y << ",";
				}
				file_fd << "\n";
			}
			for (int j = 0; j < resualt_vals[0].size(); j++)
			{
				for (int i = 0; i < size_res; i++)
				{
					file_fd << resualt_vals[i][j].z << ",";
				}
				file_fd << "\n";
			}

		}
		file_fd.close();
	}
}

void DoPca(const Mat &_data, int dim, Mat &eigenvalues, Mat &eigenvectors)
{
	assert(dim>0);
	Mat data = cv::Mat_<double>(_data);

	int R = data.rows;
	int C = data.cols;

	if (dim>C)
		dim = C;

	//计算均值  
	Mat m = Mat::zeros(1, C, data.type());

	for (int j = 0; j<C; j++)
	{
		for (int i = 0; i<R; i++)
		{
			m.at<double>(0, j) += data.at<double>(i, j);
		}
	}

	m = m / R;
	//求取6列数据对应的均值存放在m矩阵中，均值： [1.67、2.01、1.67、2.01、1.67、2.01]  


	//计算协方差矩阵  
	Mat S = Mat::zeros(R, C, data.type());
	for (int i = 0; i<R; i++)
	{
		for (int j = 0; j<C; j++)
		{
			S.at<double>(i, j) = data.at<double>(i, j) - m.at<double>(0, j); // 数据矩阵的值减去对应列的均值  
		}
	}
	  
	Mat Average = S.t() * S / (R);
	//计算协方差矩阵的方式----(S矩阵的转置 * S矩阵)/行数  


	//使用opencv提供的eigen函数求特征值以及特征向量  
	cv::eigen(Average, eigenvalues, eigenvectors);
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

static void read_csv_file(const string & filename, Mat & mat, char separator = ',')
{
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file)
	{
		string error_message = "No valid input file";
		std::cout << error_message << std::endl;
	}
	string line, path;
	Mat_d datas;
	while (getline(file, line))
	{
		stringstream liness(line);
		mat.release();
		while(getline(liness, path, separator))
		{
			double d = atof(path.c_str());
			if (!path.empty())
			{
				mat.push_back(d);
			//	std::cout << mat << std::endl;
			}
		}
		datas.push_back(mat.t());
	}
	mat = datas.clone();
}



int main()
{
#ifdef Running_CreateCSV




	std::vector<glm::vec3> sphere_points;
	/*sphere_points.push_back(glm::vec3(25.0f, 0.0f, 0.0f));
	sphere_points.push_back(glm::vec3(-25.0f, 0.0f, 0.0f));
	sphere_points.push_back(glm::vec3(0.0f, 25.0f, 0.0f));
	sphere_points.push_back(glm::vec3(0.0f, -25.0f, 0.0f));
	sphere_points.push_back(glm::vec3(0.0f, 0.0f, 25.0f));
	sphere_points.push_back(glm::vec3(0.0f, 0.0f, -25.0f));
	sphere_points.push_back(glm::vec3(0.0f, 12.0f, 25.0f));
	sphere_points.push_back(glm::vec3(0.0f, -12.0f, -25.0f));*/
	//eye
	sphere_points.push_back(glm::vec3(-18.0f, 15.0f, -5.0f));
	sphere_points.push_back(glm::vec3(-8.0f, 14.0f, -4.0f));
	sphere_points.push_back(glm::vec3(-20.0f, 7.0f, -6.0f));
	sphere_points.push_back(glm::vec3(-7.0f, 5.0f, -5.0f));

	sphere_points.push_back(glm::vec3(18.0f, 15.0f, -5.0f));
	sphere_points.push_back(glm::vec3(8.0f, 14.0f, -4.0f));
	sphere_points.push_back(glm::vec3(20.0f, 7.0f, -6.0f));
	sphere_points.push_back(glm::vec3(7.0f, 5.0f, -5.0f));

	//nose
	sphere_points.push_back(glm::vec3(-8.0f, -5.0f, -2.0f));
	sphere_points.push_back(glm::vec3(0.0f, -7.0f, 1.0f));
	sphere_points.push_back(glm::vec3(8.0f, -5.0f, -2.0f));

	//mouth
	sphere_points.push_back(glm::vec3(-12.5, -10, -2.0f));
	sphere_points.push_back(glm::vec3(0.0f, -12.0f, -1.0f));
	sphere_points.push_back(glm::vec3(12.5f, -10.0f, -2.0f));
	sphere_points.push_back(glm::vec3(0.0f, -15.0f, -1.0f));

//	glm::mat4 projectionMatrix = glm::frustum(-50.0f, 50.0f, -50.0f, 50.0f, -30.0f, 30.0f);
	glm::mat4 projectionMatrix = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, -30.0f, 30.0f);
	glm::mat4 modelMatirx = glm::mat4(1.0f);
	glm::mat4 viewMatrix = glm::mat4(1.0f);

	double rotato_val = 0.0f;
	std::vector<std::vector< glm::vec3>> resualt_vals;

	for (auto & val : sphere_points)
	{
		std::vector<glm::vec3> temp_list;
		modelMatirx = glm::mat4(1.0f);
		modelMatirx = glm::scale(modelMatirx, glm::vec3(2.0f, 2.0f, 2.0f));
		//modelMatirx = glm::rotate(modelMatirx, glm::radians(1.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		for (int i = 0; i < 3; i++)
		{
			modelMatirx = glm::rotate(modelMatirx, glm::radians(30.0f), glm::vec3(0.0f, 1.0f, 0.0f));
			glm::vec3 temp = projectionMatrix*viewMatrix*modelMatirx*glm::vec4(val, 0.0f);
			temp_list.push_back(temp);
		}
		resualt_vals.push_back(temp_list);
	}
	outResualtDataToCSV("./OutPutFile/test_sphereVertices.csv", resualt_vals);

	/*std::vector<std::vector< glm::vec3>>().swap(resualt_vals);

	for (auto & val : sphere_points)
	{
		std::vector<glm::vec3> temp_list;
		modelMatirx = glm::mat4(1.0f);
		modelMatirx = glm::rotate(modelMatirx, glm::radians(60.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		for (int i = 0; i < 720.0f /2.0f; i++)
		{
			modelMatirx = glm::rotate(modelMatirx, glm::radians(-2.0f), glm::vec3(0.0f, 1.0f, 0.0f));
			glm::vec3 temp = projectionMatrix*modelMatirx*glm::vec4(val, 0.0f);
			temp_list.push_back(temp);
		}
		resualt_vals.push_back(temp_list);
	}
	outResualtDataToCSV("./OutPutFile/test_Resualt_sphereVertices.csv", resualt_vals);*/
	std::vector<std::vector< glm::vec3>>().swap(resualt_vals);
#endif // Running_CreateCSV



	std::vector<Mat_d> vecMat;

	ifstream  ifd;

	ifd.open("./zh/test.bin");
	Mat_d temp;
	if (ifd.is_open())
	{
		while (MatFile::operator>>(ifd, temp))
		{
			vecMat.push_back(temp.clone());
		}

	}


	int vecMatcount = static_cast<int>(vecMat.size());

	Mat_d iMat(2 * vecMatcount, temp.rows,0.0);
	int tCols = temp.cols;
	for (int i = 0; i < vecMatcount; i++)
	{
		Mat_d tempMat = vecMat[i].clone().t();
		for (int j = 0; j < tCols; j++)
		{
			tempMat.row(j).copyTo(iMat.row(i+j*vecMatcount));
		}
	}
	
	Fg_NRSFM::testFrameWorkInstance instance;

	instance.Load2DMat(iMat);

	instance.Running();

	system("pause");
	return 1;
}