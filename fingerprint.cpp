#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string.h>
#include <vector>

using namespace std;
using namespace cv;

//variable for making bin file 
int N_minutiae;   //manutiae Number
int N_singular;   //singular point Number
int I_width;   //image width
int I_height;   //image height
float angle;
vector<unsigned char> m_angle;
vector<unsigned char> m_type;
vector<int> x_point;
vector<int> y_point;

vector<unsigned char> s_angle;
vector<unsigned char> s_type;
vector<int> s_x_point;
vector<int> s_y_point;


void Segmentation(Mat input, int blockSize
	, int blackT, int whiteT, Mat& meanMask, Mat& varMask) {
	Mat origin = input.clone();

	int sg_blockSize = blockSize;   //segment blocksize

	Mat sg_m, sg_v;  //Mean mask , Variance mask

				 //zero-padding
	copyMakeBorder(origin, sg_m, 0, sg_blockSize, 0, sg_blockSize, BORDER_CONSTANT, 0);
	copyMakeBorder(origin, sg_v, 0, sg_blockSize, 0, sg_blockSize, BORDER_CONSTANT, 0);

	Mat segmented_m, segmented_v;

	// Mask
	for (int img_x = 0; img_x < origin.cols; img_x += sg_blockSize) {
		for (int img_y = 0; img_y < origin.rows; img_y += sg_blockSize) {
			Mat sg_block = sg_m(Rect(img_x, img_y, sg_blockSize, sg_blockSize));

			//calculate mean
			int pixelCount = 0;
			int sg_mean = 0;
			for (int x = 0; (x < sg_blockSize) && (img_y + x < origin.rows); x++) {
				for (int y = 0; (y < sg_blockSize) && (img_x + y < origin.cols); y++) {
					sg_mean += (int)sg_m.at<uchar>(img_y + x, img_x + y);
					pixelCount++;
				}
			}
			sg_mean = sg_mean / pixelCount;

			//calculate variance
			int sg_variance = 0;
			int sg_dev = 0;
			for (int x = 0; (x < sg_blockSize) && (img_y + x < origin.rows); x++) {
				for (int y = 0; (y < sg_blockSize) && (img_x + y < origin.cols); y++) {
					sg_dev = (int)sg_m.at<uchar>(img_y + x, img_x + y);
					sg_dev -= sg_mean;
					sg_variance += sg_dev * sg_dev;
					pixelCount++;
				}
			}
			sg_variance = sg_variance / pixelCount;

			//Mean fill in Matrix
			//black (0)
			if (sg_mean > whiteT /*|| sg_mean < blackT*/)
			{
				for (int x = 0; (x < sg_blockSize) && (img_y + x < origin.rows); x++) {
					for (int y = 0; (y < sg_blockSize) && (img_x + y < origin.cols); y++) {
						sg_m.at<uchar>(img_y + x, img_x + y) = 0;
						//sg_v.at<uchar>(img_y + x, img_x + y) = 0;
					}
				}
			}
			//white (255)
			else {
				for (int x = 0; (x < sg_blockSize) && (img_y + x < origin.rows); x++) {
					for (int y = 0; (y < sg_blockSize) && (img_x + y < origin.cols); y++) {
						sg_m.at<uchar>(img_y + x, img_x + y) = 255;
						//sg_v.at<uchar>(img_y + x, img_x + y) = 255;
					}
				}
			}

			//variance fill in image
			//black (0)
			if (/*sg_variance > whiteT ||*/ sg_variance < blackT)
			{
				for (int x = 0; (x < sg_blockSize) && (img_y + x < origin.rows); x++) {
					for (int y = 0; (y < sg_blockSize) && (img_x + y < origin.cols); y++) {
						sg_v.at<uchar>(img_y + x, img_x + y) = 0;
					}
				}
			}
			//white (255)
			else {
				for (int x = 0; (x < sg_blockSize) && (img_y + x < origin.rows); x++) {
					for (int y = 0; (y < sg_blockSize) && (img_x + y < origin.cols); y++) {
						sg_v.at<uchar>(img_y + x, img_x + y) = 255;
					}
				}
			}

		}
		segmented_m = sg_m(Rect(0, 0, origin.cols, origin.rows)); // Mean mask
		segmented_v = sg_v(Rect(0, 0, origin.cols, origin.rows)); // Variance mask

		meanMask = segmented_m;
		varMask = segmented_v;
	}
}


void orientation(const Mat &inputImage, Mat &orientationMap, int blockSize)
{
	Mat fprintWithDirectionsSmoo = inputImage.clone();//copy
	Mat tmp(inputImage.size(), inputImage.type()); //make matrix equal size and equal type with Inputimage
	Mat coherence(inputImage.size(), inputImage.type());
	orientationMap = tmp.clone();

	//Gradiants x and y
	Mat grad_x, grad_y;

	Sobel(inputImage, grad_x, CV_32F, 1, 0);//grad_x
	Sobel(inputImage, grad_y, CV_32F, 0, 1);//grad_y

								  //Vector vield
	Mat Fx(inputImage.size(), inputImage.type()),
		Fy(inputImage.size(), inputImage.type()),
		Fx_gauss,
		Fy_gauss;
	Mat smoothed(inputImage.size(), inputImage.type());

	// Local orientation for each block
	int width = inputImage.cols;
	int height = inputImage.rows;
	int blockH;
	int blockW;

	//select block
	for (int i = 0; i < height; i += blockSize)
	{
		for (int j = 0; j < width; j += blockSize)
		{
			float Gsx = 0.0;
			float Gsy = 0.0;
			float Gxx = 0.0;
			float Gyy = 0.0;

			//for check bounds of img
			blockH = ((height - i) < blockSize) ? (height - i) : blockSize;
			blockW = ((width - j) < blockSize) ? (width - j) : blockSize;

			//average at block W¬çW
			for (int u = i; u < i + blockH; u++)
			{
				for (int v = j; v < j + blockW; v++)
				{
					Gsx += (grad_x.at<float>(u, v)*grad_x.at<float>(u, v)) - (grad_y.at<float>(u, v)*grad_y.at<float>(u, v));
					Gsy += 2 * grad_x.at<float>(u, v) * grad_y.at<float>(u, v);
					Gxx += grad_x.at<float>(u, v)*grad_x.at<float>(u, v);
					Gyy += grad_y.at<float>(u, v)*grad_y.at<float>(u, v);
				}
			}

			float coh = sqrt(pow(Gsx, 2) + pow(Gsy, 2)) / (Gxx + Gyy);
			//smoothed
			float fi = 0.5*fastAtan2(Gsy, Gsx)*CV_PI / 180;

			Fx.at<float>(i, j) = cos(2 * fi);
			Fy.at<float>(i, j) = sin(2 * fi);

			//fill blocks
			for (int u = i; u < i + blockH; u++)
			{
				for (int v = j; v < j + blockW; v++)
				{
					orientationMap.at<float>(u, v) = fi;
					Fx.at<float>(u, v) = Fx.at<float>(i, j);
					Fy.at<float>(u, v) = Fy.at<float>(i, j);
					coherence.at<float>(u, v) = (coh < 0.85) ? 1 : 0;
				}
			}

		}
	} ///for

	//GaussianBlur
	GaussianBlur(Fx, Fx_gauss, Size(5, 5), 0);
	GaussianBlur(Fy, Fy_gauss, Size(5, 5), 0);

	for (int m = 0; m < height; m++)
	{
		for (int n = 0; n < width; n++)
		{
			smoothed.at<float>(m, n) = 0.5*fastAtan2(Fy_gauss.at<float>(m, n), Fx_gauss.at<float>(m, n))*CV_PI / 180;
			if ((m%blockSize) == 0 && (n%blockSize) == 0) {
				int x = n;
				int y = m;
				int ln = sqrt(2 * pow(blockSize, 2)) / 2;
				float dx = ln * cos(smoothed.at<float>(m, n) - CV_PI / 2);
				float dy = ln * sin(smoothed.at<float>(m, n) - CV_PI / 2);
				arrowedLine(fprintWithDirectionsSmoo, Point(x, y + blockH), Point(x + dx, y + blockW + dy), Scalar::all(255), 1, LINE_AA, 0, 0.06*blockSize);
			}
		}
	}///for2
	orientationMap = smoothed.clone();
}

void Normalize(Mat & image)
{
	Scalar mean, dev;

	meanStdDev(image, mean, dev, noArray());   //calculate mean and StdDev

	double M = mean.val[0];
	double D = dev.val[0];

	//Normalize image
	for (int i(0); i < image.rows; i++)
	{
		for (int j(0); j < image.cols; j++)
		{
			if (image.at<float>(i, j) > M)
				image.at<float>(i, j) = 100.0 / 255 + sqrt(100.0 / 255 * pow(image.at<float>(i, j) - M, 2) / D);
			else
				image.at<float>(i, j) = 100.0 / 255 - sqrt(100.0 / 255 * pow(image.at<float>(i, j) - M, 2) / D);
		}
	}
}


void enhancement(const Mat &inputImage, Mat &orientationMap, int blockSize, Mat &dst)
{

	Mat fprintWithDirectionsSmoo = inputImage.clone();   //copy
	Mat tmp(inputImage.size(), inputImage.type());   //make matrix equal size and equal type with Inputimage
	Mat coherence(inputImage.size(), inputImage.type());
	orientationMap = tmp.clone();   //will be stored orietationMap

							//Gradiants x and y
	Mat grad_x, grad_y;


	Scharr(inputImage, grad_x, CV_32F, 1, 0);   //find grad_x
	Scharr(inputImage, grad_y, CV_32F, 0, 1);   //find grad_y

									 //Vector vield
	Mat Fx(inputImage.size(), inputImage.type()),
		Fy(inputImage.size(), inputImage.type()),
		Fx_gauss,
		Fy_gauss;
	Mat smoothed(inputImage.size(), inputImage.type());   //will be stored smoothed orientationMap

											 // Local orientation for each block
	int width = inputImage.cols;
	int height = inputImage.rows;
	int blockH;
	int blockW;

	//select block
	for (int i = 0; i < height; i += blockSize)
	{
		for (int j = 0; j < width; j += blockSize)
		{
			float Gsx = 0.0;
			float Gsy = 0.0;
			float Gxx = 0.0;
			float Gyy = 0.0;

			//for check bounds of img
			blockH = ((height - i) < blockSize) ? (height - i) : blockSize;
			blockW = ((width - j) < blockSize) ? (width - j) : blockSize;

			//average at block W¬çW
			for (int u = i; u < i + blockH; u++)
			{
				for (int v = j; v < j + blockW; v++)
				{
					Gsx += (grad_x.at<float>(u, v)*grad_x.at<float>(u, v)) - (grad_y.at<float>(u, v)*grad_y.at<float>(u, v));
					Gsy += 2 * grad_x.at<float>(u, v) * grad_y.at<float>(u, v);
					Gxx += grad_x.at<float>(u, v)*grad_x.at<float>(u, v);
					Gyy += grad_y.at<float>(u, v)*grad_y.at<float>(u, v);
				}
			}

			float coh = sqrt(pow(Gsx, 2) + pow(Gsy, 2)) / (Gxx + Gyy);
			//theta of pixel
			float fi = 0.5*fastAtan2(Gsy, Gsx)*CV_PI / 180;

			Fx.at<float>(i, j) = cos(2 * fi);
			Fy.at<float>(i, j) = sin(2 * fi);

			//fill blocks
			for (int u = i; u < i + blockH; u++)
			{
				for (int v = j; v < j + blockW; v++)
				{
					orientationMap.at<float>(u, v) = fi;
					Fx.at<float>(u, v) = Fx.at<float>(i, j);
					Fy.at<float>(u, v) = Fy.at<float>(i, j);
					coherence.at<float>(u, v) = (coh < 0.85) ? 1 : 0;
				}
			}

		}
	} ///for


	  //GAUSSIAN BLUR SMOOTHING Gaussianblur smoothing
	GaussianBlur(Fx, Fx_gauss, Size(5, 5), 0);
	GaussianBlur(Fy, Fy_gauss, Size(5, 5), 0);


	//smoothed Orientation
	for (int m = 0; m < height; m++)
	{
		for (int n = 0; n < width; n++)
		{
			smoothed.at<float>(m, n) = 0.5*fastAtan2(Fy_gauss.at<float>(m, n), Fx_gauss.at<float>(m, n))*CV_PI / 180;
			if ((m%blockSize) == 0 && (n%blockSize) == 0) {
				int x = n;
				int y = m;
				int ln = sqrt(2 * pow(blockSize, 2)) / 2;
				float dx = ln * cos(smoothed.at<float>(m, n) - CV_PI / 2);
				float dy = ln * sin(smoothed.at<float>(m, n) - CV_PI / 2);
				//draw line direction
				arrowedLine(fprintWithDirectionsSmoo, Point(x, y + blockH), Point(x + dx, y + blockW + dy), Scalar::all(255), 1, LINE_AA, 0, 0.06*blockSize);
			}
		}
	}///for2
	orientationMap = smoothed.clone();

	//   Matrix for Gabor filter   //
	Mat dst2(inputImage.size(), inputImage.type());
	Mat kernel3;
	Mat src3 = inputImage.clone();

	int kernel_size = 7; //gabor filter kernel size
	int special = (kernel_size - 1) / 2;

	//padding source image
	Mat temp = inputImage.clone();
	copyMakeBorder(temp, temp, special, special, special, special, BORDER_REFLECT);
	Mat forgab(temp.size(), temp.type());

	//padding Orientaion Imge
	Mat stemp = smoothed.clone();
	copyMakeBorder(stemp, stemp, special, special, special, special, BORDER_REFLECT);

	// gabor filter parameter 
	double sig = 9, lm = 7.2, gm = 0.02, ps = 0;
	double theta;
	float ffi;

	/////Gabor filtering
	for (int m = special; m < temp.rows - special; m++) {
		for (int n = special; n < temp.cols - special; n++) {
			theta = stemp.at<float>(m, n);
			kernel3 = getGaborKernel(Size(kernel_size, kernel_size), sig, theta, lm, gm, ps, CV_32F);
			ffi = 0;
			for (int k = 0; k < kernel_size; k++) {
				for (int l = 0; l < kernel_size; l++) {
					ffi += temp.at<float>(m - special + k, n - special + l)*kernel3.at<float>(kernel_size - 1 - k, kernel_size - 1 - l);
				}
			}
			forgab.at<float>(m, n) = ffi / (kernel_size * kernel_size);
		}
	}

	//Roi of original image size
	Rect roi(special, special, src3.cols, src3.rows);
	Mat forgab_roi = forgab(roi);
	src3 = forgab_roi;

	src3.convertTo(src3, CV_8U, 255, 0);   //0~255 scale
	dst = src3;
	imshow("Smoothed orientation field", smoothed);
	imshow("Orientation", fprintWithDirectionsSmoo);
}

void singularpoint(const Mat &inputImage, Mat &orien, Mat &core, Mat &delta)
{
	Mat cmasked = inputImage.clone();
	cmasked.convertTo(cmasked, CV_32F);//convert type for using orientation function 
	orientation(cmasked, orien, 9);//block size is 9

	Mat sp;
	sp = orien.clone();
	//Normalize orientation field 0~7 value 
	for (int i = 0; i < sp.rows; i++) {
		for (int j = 0; j < sp.cols; j++) {
			if (sp.at<float>(i, j) < (CV_PI*0.125)) sp.at<float>(i, j) = 0;
			else if (sp.at<float>(i, j) < (CV_PI*0.25)) sp.at<float>(i, j) = 1;
			else if (sp.at<float>(i, j) < (CV_PI*0.375)) sp.at<float>(i, j) = 2;
			else if (sp.at<float>(i, j) < (CV_PI*0.5)) sp.at<float>(i, j) = 3;
			else if (sp.at<float>(i, j) < (CV_PI*0.625)) sp.at<float>(i, j) = 4;
			else if (sp.at<float>(i, j) < (CV_PI*0.75)) sp.at<float>(i, j) = 5;
			else if (sp.at<float>(i, j) < (CV_PI*0.875)) sp.at<float>(i, j) = 6;
			else sp.at<float>(i, j) = 7;
		}
	}
	int a[9];
	int va;
	int sub;
	//core & delta Map
	core = Mat::zeros(sp.size(), sp.type());
	delta = Mat::zeros(sp.size(), sp.type());
	for (int i = 1; i < sp.rows - 1; i++) {
		for (int j = 1; j < sp.cols - 1; j++) {
			va = 0;
			a[0] = sp.at<float>(i - 1, j - 1);
			a[1] = sp.at<float>(i - 1, j);
			a[2] = sp.at<float>(i - 1, j + 1);
			a[3] = sp.at<float>(i, j + 1);
			a[4] = sp.at<float>(i + 1, j + 1);
			a[5] = sp.at<float>(i + 1, j);
			a[6] = sp.at<float>(i + 1, j - 1);
			a[7] = sp.at<float>(i, j - 1);
			a[8] = a[0];
			for (int k = 0; k < 7; k++) {
				sub = a[k] - a[k + 1];
				if (sub == -7) sub = 1;
				if (sub == 7) sub = -1;
				va += sub;
			}
			//calculate gradient gap
			if (va == -8) core.at<float>(i, j) = 1;
			if (va == 8) delta.at<float>(i, j) = 1;
		}
	}
}
void thinningIteration(cv::Mat& im, int iter)
{
	cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);   //zero matrix

											  //thinning
	for (int i = 1; i < im.rows - 1; i++)
	{
		for (int j = 1; j < im.cols - 1; j++)
		{
			uchar p2 = im.at<uchar>(i - 1, j);
			uchar p3 = im.at<uchar>(i - 1, j + 1);
			uchar p4 = im.at<uchar>(i, j + 1);
			uchar p5 = im.at<uchar>(i + 1, j + 1);
			uchar p6 = im.at<uchar>(i + 1, j);
			uchar p7 = im.at<uchar>(i + 1, j - 1);
			uchar p8 = im.at<uchar>(i, j - 1);
			uchar p9 = im.at<uchar>(i - 1, j - 1);

			int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
				(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
				(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
				(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
			int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
			int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
			int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
				marker.at<uchar>(i, j) = 1;
		}
	}

	im &= ~marker;
}


void thinning(cv::Mat& im)
{
	im /= 255;   //0~1 scale

	cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
	cv::Mat diff;

	//thinning iteration
	do {
		thinningIteration(im, 0);
		thinningIteration(im, 1);
		cv::absdiff(im, prev, diff);
		im.copyTo(prev);
	} while (cv::countNonZero(diff) > 0);

	im *= 255;   //0~255 scale
}



void detect(Mat & minutiae, Mat & mask, Mat & OrientMap)
{
	int end = 0, endcheck = 0;
	int bif = 0, bifcheck = 0;
	int found = 0;
	Mat color2;
	cvtColor(minutiae, color2, COLOR_GRAY2BGR);


	int a, b, c, d, e;
	for (int x = 1; x < minutiae.rows - 1; x++) {
		for (int y = 1; y < minutiae.cols - 1; y++) {
			a = mask.at<uchar>(x, y);
			b = mask.at<uchar>(x - 1, y);
			c = mask.at<uchar>(x, y - 1);
			d = mask.at<uchar>(x + 1, y);
			e = mask.at<uchar>(x, y + 1);

			if (a != 0 && b != 0 && c != 0 && d != 0 && e != 0) {   //for not detecting point cut by mask because it is not ending point
				found = 0;
				if (minutiae.at<uchar>(x, y) == 255) {

					if ((minutiae.at<uchar>(x - 1, y - 1)) != (minutiae.at<uchar>(x, y - 1))) found++;
					if ((minutiae.at<uchar>(x, y - 1)) != (minutiae.at<uchar>(x + 1, y - 1))) found++;
					if ((minutiae.at<uchar>(x + 1, y - 1)) != (minutiae.at<uchar>(x + 1, y))) found++;
					if ((minutiae.at<uchar>(x + 1, y)) != (minutiae.at<uchar>(x + 1, y + 1))) found++;
					if ((minutiae.at<uchar>(x + 1, y + 1)) != (minutiae.at<uchar>(x, y + 1))) found++;
					if ((minutiae.at<uchar>(x, y + 1)) != (minutiae.at<uchar>(x - 1, y + 1))) found++;
					if ((minutiae.at<uchar>(x - 1, y + 1)) != (minutiae.at<uchar>(x - 1, y))) found++;
					if ((minutiae.at<uchar>(x - 1, y)) != (minutiae.at<uchar>(x - 1, y - 1))) found++;
				}

				if (found == 2) end = 1;
				else if (found == 6) bif = 1;

				//for make bin file
				if (end || bif) {
					x_point.push_back(y);
					y_point.push_back(x);
					angle = (OrientMap.at<float>(x, y) * (-1) + (CV_PI / 2)) * 180 * 255 / CV_PI / 360;
					m_angle.push_back((unsigned char)angle);
					if (end) m_type.push_back(1);
					else m_type.push_back(3);
				}
				//

				if ((end)) {
					Point pt1 = Point(y, x);
					circle(color2, pt1, 3, Scalar(255, 0, 0), 1, 8, 0);   //blue circle
					end = 0;
					endcheck++;
				}
				if ((bif)) {
					Point pt2 = Point(y, x);
					circle(color2, pt2, 3, Scalar(0, 0, 255), 1, 8, 0);   //red circle
					bif = 0;
					bifcheck++;
				}
			}
		}
	}
	imshow("minutiae_color", color2);
	N_minutiae = endcheck + bifcheck;   //all minutiae number
	cout << "end point Num : " << endcheck << endl;
	cout << "bif point Num : " << bifcheck << endl;
}

void spdetect(Mat & minutiae, Mat & mask, Mat & OrientMap, Mat & coreMap, Mat & deltaMap)
{
	Mat color3;
	cvtColor(minutiae, color3, COLOR_GRAY2BGR);
	int core = 0, corecheck = 0;
	int delta = 0, deltacheck = 0;

	int a, b, c, d, e;
	int c1, c2, c3;
	int d1, d2, d3;
	for (int x = 1; x < minutiae.rows - 1; x++) {
		for (int y = 1; y < minutiae.cols - 1; y++) {
			a = mask.at<uchar>(x, y);
			b = mask.at<uchar>(x - 1, y);
			c = mask.at<uchar>(x, y - 1);
			d = mask.at<uchar>(x + 1, y);
			e = mask.at<uchar>(x, y + 1);

			c1 = coreMap.at<float>(x, y - 1);
			c2 = coreMap.at<float>(x - 1, y - 1);
			c3 = coreMap.at<float>(x - 1, y);

			d1 = deltaMap.at<float>(x, y - 1);
			d2 = deltaMap.at<float>(x - 1, y - 1);
			d3 = deltaMap.at<float>(x - 1, y);

			if (a != 0 && b != 0 && c != 0 && d != 0 && e != 0) {   //for not detecting point cut by mask because it is not ending point
				if (c1 == 1 || c2 == 1 || c3 == 1 || d1 == 1 || d2 == 1 || d3 == 1) continue; //avoid overlap various candidate
				if (coreMap.at<float>(x, y) == 1) core = 1;
				if (deltaMap.at<float>(x, y) == 1) delta = 1;

				//for make bin file
				if (core || delta) {
					s_x_point.push_back(y);
					s_y_point.push_back(x);
					angle = (OrientMap.at<float>(x, y) * (-1) + (CV_PI / 2)) * 180 * 255 / CV_PI / 360;
					s_angle.push_back((unsigned char)angle);
					if (core) s_type.push_back(10);
					else s_type.push_back(11);
				}
				//

				if ((core)) {
					Point pt1 = Point(y, x);
					circle(color3, pt1, 3, Scalar(0, 255, 0), 1, 8, 0);   //green circle
					core = 0;
					corecheck++;
				}
				if ((delta)) {
					Point pt2 = Point(y, x);
					circle(color3, pt2, 3, Scalar(0, 0, 255), 1, 8, 0);   //red circle
					delta = 0;
					deltacheck++;
				}
			}
		}
	}
	imshow("singular_color", color3);
	N_singular = corecheck + deltacheck;   //all singular point number
	cout << "core point Num : " << corecheck << endl;
	cout << "delta point Num : " << deltacheck << endl;
}




int main()
{
	Mat mat = imread("Team7/LSE/2019_7_3_R_I_1.bmp", IMREAD_GRAYSCALE); // READ INPUT IMAGE IN GRAY SCALE
	imshow("Original", mat);
	I_width = mat.cols;
	I_height = mat.rows;



	//      Segmentation      //
	Mat meanMask, varianceMask;
	Segmentation(mat, 5, 5, 230, meanMask, varianceMask);   //block size is 5
	Mat mask = meanMask & varianceMask;
	dilate(mask, mask, Mat(), Point(-1, -1), 3, 0, BORDER_CONSTANT);
	erode(mask, mask, Mat(), Point(-1, -1), 3, 0, BORDER_CONSTANT);
	imshow("mask", mask);

	//      Intensity Normalize      //
	equalizeHist(mat, mat);
	imshow("EqualizeHistogram", mat);
	mat.convertTo(mat, CV_32F, 1.0 / 255, 0);   //type : uchar -> float / scale 0~1
	Normalize(mat);
	imshow("Normalize", mat);

	//      Enhancement      //
	int blockSize = 15; // SPECIFY THE BLOCKSIZE;
	int height = mat.rows;
	int width = mat.cols;
	Mat orientationMap;
	Mat dst;

	enhancement(mat, orientationMap, blockSize, dst);
	imshow("Enhancement", dst);

	Mat orien, coreMap, deltaMap;
	singularpoint(dst, orien, coreMap, deltaMap);

	//         Binary         //
	adaptiveThreshold(dst, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, 0);
	imshow("Binary", dst);


	//      mask + Image      //
	Mat masked = dst.clone();
	masked |= ~mask;
	imshow("masked", masked);



	//      Thinning      //
	masked = ~masked;
	thinning(masked);
	imshow("thinning", masked);
	erode(mask, mask, Mat(), Point(-1, -1), 5, 0, BORDER_CONSTANT);

	//      Manutiae Detection      //
	detect(masked, mask, orientationMap);
	spdetect(masked, mask, orientationMap, coreMap, deltaMap);


	//      Final Image      //
	imshow("Destination", masked);



	//      store .bin file     //

	int lim = 0;

	//print value
	cout << "width: " << I_width << " height: " << I_height << endl;
	cout << "minutiae of Number: " << N_minutiae << endl;
	cout << "singular of Number: " << N_singular << endl;

	for (int i = 0; i < N_singular; i++) {
		if (i >= 3) {
			lim = 3;
			break;
		}
		printf("X[%d]: %d Y[%d]: %d O[%d]: %d T[%d]: %d\n", i, s_x_point[i], i, s_y_point[i], i, s_angle[i], i, s_type[i]);
		lim++;
	}

	for (int i = 0; i < N_minutiae; i++) {
		if (i + lim >= 50) break;
		printf("X[%d]: %d Y[%d]: %d O[%d]: %d T[%d]: %d\n", i, x_point[i], i, y_point[i], i, m_angle[i], i, m_type[i]);
	}


	//lim = 0;
	//ofstream output("2019_7_2_R_I_1.bin", ios::out | ios::binary);
	//output.write((char*)&I_width, sizeof(int));
	//output.write((char*)&I_height, sizeof(int));
	//output.write((char*)&N_minutiae, sizeof(int));
	//output.write((char*)&N_singular, sizeof(int));

	//for (int i = 0; i < N_singular; i++) {
	//   if (i >= 3) {
	//      lim = 3;
	//      break;
	//   }
	//   output.write((char*)&s_x_point[i], sizeof(int));
	//   output.write((char*)&s_y_point[i], sizeof(int));
	//   output.write((char*)&s_angle[i], sizeof(char));
	//   output.write((char*)&s_type[i], sizeof(char));
	//   lim++;
	//}

	//for (int i = 0; i < N_minutiae; i++) {
	//   if (i + lim >= 50) break;
	//   output.write((char*)&x_point[i], sizeof(int));
	//   output.write((char*)&y_point[i], sizeof(int));
	//   output.write((char*)&m_angle[i], sizeof(char));
	//   output.write((char*)&m_type[i], sizeof(char));
	//}
	//output.close();

	waitKey(0);
	return 0;
}