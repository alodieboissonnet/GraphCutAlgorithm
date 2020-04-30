#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>

#include "maxflow/graph.h"

#include "image.h"

using namespace std;
using namespace cv;

// This section shows how to use the library to compute a minimum cut on the following graph:
//
//		        SOURCE
//		       /       \
//		     1/         \6
//		     /      4    \
//		   node0 -----> node1
//		     |   <-----   |
//		     |      3     |
//		     \            /
//		     5\          /1
//		       \        /
//		          SINK
//
///////////////////////////////////////////////////

void testGCuts()
{
	Graph<int,int,int> g(/*estimated # of nodes*/ 2, /*estimated # of edges*/ 1);
	g.add_node(2);
	g.add_tweights( 0,   /* capacities */  1, 5 );
	g.add_tweights( 1,   /* capacities */  6, 1 );
	g.add_edge( 0, 1,    /* capacities */  4, 3 );
	int flow = g.maxflow();
	cout << "Flow = " << flow << endl;
	for (int i=0;i<2;i++)
		if (g.what_segment(i) == Graph<int,int,int>::SOURCE)
			cout << i << " is in the SOURCE set" << endl;
		else
			cout << i << " is in the SINK set" << endl;
}


void gradient(Mat Ic, Mat& gradient) {
	Mat I;
	cvtColor(Ic, I, COLOR_BGR2GRAY);
	int m = I.rows, n = I.cols;
	gradient = Mat(m, n, CV_32F);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			float ix, iy;
			if (i == 0 || i == m - 1)
				iy = 0;
			else
				iy = (float(I.at<uchar>(i - 1, j)) - float(I.at<uchar>(i + 1, j))) / 2;
			if (j == 0 || j == n - 1)
				ix = 0;
			else
				ix = (float(I.at<uchar>(i, j + 1)) - float(I.at<uchar>(i, j - 1))) / 2;
			gradient.at<float>(i, j) = sqrt(ix * ix + iy * iy);
		}
	}

	//gradient.convertTo(gradient, -1, 0.001, 0);
}

float g(int i, int j, Mat& gradient, float alpha, float beta){
	return alpha / (1 + beta * pow(gradient.at<float>(i,j),2));
}

float ge(int i, int j, Mat img){
	return sqrt(pow(img.at<Vec3f>(i,j)[0] - 157,2) + pow(img.at<Vec3f>(i,j)[1] - 138,2) + pow(img.at<Vec3f>(i,j)[3] - 1,2));
}

float gi(int i, int j, Mat img){
	return sqrt(pow(img.at<Vec3f>(i,j)[0] - 247,2) + pow(img.at<Vec3f>(i,j)[1] - 255,2) + pow(img.at<Vec3f>(i,j)[3] - 202,2));
}

Graph<float,float,float> imageToGraph(Mat img){
	Graph<float,float,float> graph(img.rows * img.cols, 4 * img.rows * img.cols - (img.rows + img.cols));
	graph.add_node(img.rows*img.cols);

	Mat img_gray, gradient;
	Mat img_float = Mat(img.rows, img.cols, CV_32FC3);
	float alpha = 10000.0f;
	float beta = 1.0f;
	cvtColor(img, img_gray, COLOR_BGR2GRAY);
	img.convertTo(img_float, CV_32F);


	Laplacian(img_gray, gradient, CV_32F);
	gradient.convertTo(gradient, -1, 0.1, 0);
	imshow("gradient", gradient);

	for (int i=0; i<img.rows; i++){
		for (int j=0; j<img.cols; j++){
			graph.add_tweights(i*img.cols + j, gi(i,j,img_float), ge(i,j,img_float));
			if (j != img.cols-1)
				graph.add_edge(i*img.cols + j, i*img.cols + (j+1), 0.5 * (g(i,j,gradient, alpha, beta) + g(i,j+1,gradient,alpha, beta)), 0.5 * (g(i,j,gradient,alpha, beta) + g(i,j+1,gradient,alpha, beta)));
			if (i != img.rows - 1)
				graph.add_edge(i*img.cols + j, (i+1)*img.cols + j, 0.5 * (g(i,j,gradient,alpha, beta) + g(i+1,j,gradient,alpha, beta)), 0.5 * (g(i,j,gradient,alpha, beta) + g(i+1,j,gradient,alpha, beta)));
		}
	}

	int flow = graph.maxflow();
	cout << "Flow = " << flow << endl;
	Mat result = Mat(img.rows, img.cols, CV_32F);
	for (int i=0;i<img.rows;i++){
		for (int j=0; j<img.cols; j++){
			if (graph.what_segment(i*img.cols + j) == Graph<float, float, float>::SOURCE)
				result.at<float>(i,j) = 0;
			else
				result.at<float>(i,j) = 255;
		}
	}
	imshow("result", result);
	return result;
}




int main() {
	//testGCuts();

	Image<Vec3b> Icolor= Image<Vec3b>(imread("../fishes.jpg"));
	imageToGraph(Icolor);
	waitKey(0);

	return 0;
}
