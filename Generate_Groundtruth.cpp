#include <iostream>
#include <opencv2/opencv.hpp>

#include <numeric>

using namespace std;
using namespace cv;

bool load_images(const string dirname, vector< Mat >& img_lst, bool gray = 0)
{
	vector< string > files;
	cv::glob(dirname, files);
	for (size_t i = 0; i < files.size(); ++i)
	{
		Mat img = cv::imread(files[i]); // load the image
		if (img.empty())            // invalid image, skip it.
		{
			cout << files[i] << " is invalid!" << endl;
			continue;
		}
		if (gray)
		{
			cvtColor(img, img, COLOR_BGR2GRAY);
		}
		img_lst.push_back(img);
	}
	return false;
}

//输入格式是Mat类型，I1，I2代表是输入的两幅图像
double getPSNR(const Mat& I1, const Mat& I2)
{
	Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|AbsDiff函数是 OpenCV 中计算两个数组差的绝对值的函数
	s1.convertTo(s1, CV_32F);  // 这里我们使用的CV_32F来计算，因为8位无符号char是不能进行平方计算
	s1 = s1.mul(s1);           // |I1 - I2|^2

	Scalar s = sum(s1);         //对每一个通道进行加和

	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

	if (sse <= 1e-10) // 对于非常小的值我们将约等于0
		return 0;
	else
	{
		double  mse = sse / (double)(I1.channels() * I1.total());//计算MSE
		double psnr = 10.0*log10((255 * 255) / mse);
		return psnr;//返回PSNR
	}
}

void compare_classical_methods(Mat img1)
{
	//先定义PSNR
	double p1, p2, p3;

	img1 = imread("src.jpg");

	double time0 = static_cast<double>(getTickCount());
	Mat med;
	cv::medianBlur(img1, med, 5);
	time0 = ((double)getTickCount() - time0) / getTickFrequency();
	cout << time0 * 1000 << endl;

	time0 = static_cast<double>(getTickCount());
	Mat gau;
	cv::GaussianBlur(img1, gau, cv::Size(5, 5), 3, 3);
	time0 = ((double)getTickCount() - time0) / getTickFrequency();
	cout << time0 << endl;

	time0 = static_cast<double>(getTickCount());
	Mat bil;
	bilateralFilter(img1, bil, 25, 5, 5);
	time0 = ((double)getTickCount() - time0) / getTickFrequency();
	cout << time0 << endl;

	cout << "\n" << endl;

	imwrite("med.jpg", med);
	imwrite("gau.jpg", gau);
	imwrite("bil.jpg", bil);

	p1 = getPSNR(img1, med);
	p2 = getPSNR(img1, gau);
	p3 = getPSNR(img1, bil);

	cout << p1 << endl;
	cout << p2 << endl;
	cout << p3 << endl;

	cv::waitKey();
}

void get_clean_from_same_view(string path, string result_name)
{
	vector<string> files;
	cv::glob(path, files);

	vector< vector<int> > p0, p1, p2;
	for (int i = 0; i < files.size(); i++)
	{
		Mat img = cv::imread(files[i]);

		vector<int> pp0, pp1, pp2;

		for (int x = 0; x < img.cols; x++)
		{
			for (int y = 0; y < img.rows; y++)
			{
				pp0.push_back(img.at<Vec3b>(y, x)[0]);
				pp1.push_back(img.at<Vec3b>(y, x)[1]);
				pp2.push_back(img.at<Vec3b>(y, x)[2]);
			}
		}

		p0.push_back(pp0);
		p1.push_back(pp1);
		p2.push_back(pp2);
	}

	Mat img0 = cv::imread(files[0]);
	Mat new_img = Mat::zeros(img0.size(), img0.type());

	long long idx = 0;
	for (int x = 0; x < new_img.cols; x++)
	{
		for (int y = 0; y < new_img.rows; y++)
		{
			int sum0 = 0, sum1 = 0, sum2 = 0;;

			for (int i = 0; i < p0.size(); i++)
			{
				sum0 += p0[i][idx];
				sum1 += p1[i][idx];
				sum2 += p2[i][idx];
			}
			new_img.at<Vec3b>(y, x)[0] = sum0 / p0.size();
			new_img.at<Vec3b>(y, x)[1] = sum1 / p1.size();
			new_img.at<Vec3b>(y, x)[2] = sum2 / p2.size();

			idx++;
		}
	}

	Mat result = new_img.clone();
	result_name = result_name + ".png";
	imwrite(result_name, new_img);

	// double psnr = getPSNR(img0, new_img);
}



int main()
{
	//string view0_path = "E:\\database\\nuclear_n2n\\train\\";
	//string view_name = "22";
	//view0_path += view_name;
	//Mat view0;
	//double psnr0;
	//get_clean_from_same_view(view0_path, view_name + ".jpg", view0, psnr0);
	//cout << psnr0 << endl;
	//imshow("view0", view0);
	//cv::waitKey();

	for (int i = 37; i <= 37; i++)
	{
		string path = "E:/database/nuclear/nuclear_same_noise/accepted/pairs/MoreThan10/long-shot (" + to_string(i) + ")/*.png";
		get_clean_from_same_view(path, to_string(i));
	}


	//std::system("pause");
	return 0;
}