#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

Mat fruye(Mat I)
{
	/*Mat I = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);*/
	cvtColor(I, I, CV_BGR2GRAY);
	if (I.empty())
		return Mat();

	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(I.rows);
	int n = getOptimalDFTSize(I.cols); // on the border add zero values
	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	dft(complexI, complexI);            // this way the result may fit in the source matrix

	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude  
	Mat magI = planes[0];

	magI += Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);

	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center        
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant 
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a 
	// viewable image form (float between values 0 and 1).

	//imshow("Input Image", I);    // Show the result
	//imshow("spectrum magnitude", magI);
	//imshow(w_n, magI);
	return magI;
}

vector<Mat> cutImage(Mat src, int nn)
{
	vector<Mat> result;

	int sub_w = src.cols / nn;
	int sub_h = src.rows / nn;

	for (int i = 0; i < src.cols - sub_w; i += sub_w)
	{
		for (int j = 0; j < src.rows - sub_h; j += sub_h)
		{
			Mat sub_img = src(Rect(i, j, sub_w, sub_h));
			result.push_back(sub_img);
		}
	}

	return result;
}

string itos(int i)
{
	string result;
	stringstream s_o;
	s_o << i;
	s_o >> result;
	return result;
}


Mat mergeWholeImages(vector<Mat>& srcs, int nn)
{
	int sub_w = srcs[0].cols;
	int sub_h = srcs[0].rows;

	int total_w = sub_w * nn;
	int total_h = sub_h * nn;

	Mat total(total_h, total_w, srcs[0].type());

	for (int i = 0; i < nn; ++i)
	{
		for (int j = 0; j < nn; ++j)
		{
			Rect roi(i * sub_w,j * sub_h, sub_w, sub_h);
			srcs[i * nn + j].copyTo(total(roi));
		}
	}
	return total;
}

Mat wholeFu(Mat src, int nn)
{
	if (nn == 0)
	{
		return fruye(src);
	}

	int sub_w = src.cols / nn;
	int sub_h = src.rows / nn;
	int i, j;

	Mat total(src.rows, src.cols, CV_32F);

	for (i = 0; i < src.cols - sub_w;)
	{
		for (j = 0; j < src.rows - sub_h;)
		{
			Rect roi(i, j, sub_w, sub_h);
			Mat sub_img = src(roi);
			Mat fu = fruye(sub_img);
			cout << fu.rows << " " << fu.cols << endl;
			fu.copyTo(total(Rect(i, j, fu.cols, fu.rows)));
			j += fu.rows;
		}
		Rect roi(i, j, sub_w, src.rows - j);
		Mat sub_img = src(roi);
		Mat fu = fruye(sub_img);
		j = total.rows - fu.rows;
		fu.copyTo(total(Rect(i,  j, fu.cols, fu.rows)));
		i += fu.cols;
	}
	Rect roi(i, j, src.cols - i, src.rows - j);
	Mat sub_img = src(roi);
	Mat fu = fruye(sub_img);
	i = total.cols - fu.cols;
	fu.copyTo(total(Rect(i, j, fu.cols, fu.rows)));

	for (int k = 0; k < j;)
	{
		Rect roi(i, k, src.cols - i, sub_h);
		Mat sub_img = src(roi);
		Mat fu = fruye(sub_img);
		fu.copyTo(total(Rect(i, k, fu.cols, fu.rows)));
		k += fu.rows;
	}

	return total;
}

string video_path = "C:\\Users\\jek\\Desktop\\jinru.avi";

//int main(int argc, char ** argv)
//{
//	Mat frame;
//	VideoCapture capture;
//	capture.open(video_path.c_str());
//	if (!capture.isOpened())
//	{
//		return -1;
//	}
//
//	while (true)
//	{
//		if (!capture.read(frame))
//		{
//			break;
//		}
//
//		Mat fu = wholeFu(frame, 100);
//		Mat fu_1 = wholeFu(frame, 50);
//
//		imshow("fu", fu);
//		imshow("fu_1", fu_1);
//		//imshow("src", frame);
//
//		waitKey(5);
//	}
//
//	return 0;
//}

int main()
{
	string file_path = "C:\\Users\\jek\\Desktop\\4.bmp";

	Mat img = imread(file_path.c_str());

	for (int i = 0; i < 26; i += 2)
	{
		Mat fu = wholeFu(img, i);
		imshow("img" + itos(i), fu);
		waitKey(10);
	}
	//Mat fu = wholeFu(img, 24);
	//imshow("img" + itos(4), fu);

	waitKey(0);
	return 0;
}