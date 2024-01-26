// color.cpp
// Name: Mihir Chitre, Aditya Gurnani
// Date: 01/23/2024
// Purpose: Contains color picking functions

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace cv;
using namespace std;


int main()
{
    VideoCapture cap(0);
    Mat img;
    Mat imgHSV, mask, imgColor;
    int hmin = 0, smin = 0, vmin = 0; // Min HSV values
    int hmax = 179, smax = 255, vmax = 255; // Max HSV values

    namedWindow("Trackbars", (640, 200)); // Creating Window for trackbars
    createTrackbar("Hue Min", "Trackbars", &hmin, 179);
    createTrackbar("Hue Max", "Trackbars", &hmax, 179);
    createTrackbar("Sat Min", "Trackbars", &smin, 255);
    createTrackbar("Sat Max", "Trackbars", &smax, 255);
    createTrackbar("Val Min", "Trackbars", &vmin, 255);
    createTrackbar("Val Max", "Trackbars", &vmax, 255);

    while (true) {

        cap.read(img);
        cvtColor(img, imgHSV, COLOR_BGR2HSV);

        // Lower and Upper bounds for HSV filter
        Scalar lower(hmin, smin, vmin);
        Scalar upper(hmax, smax, vmax);

        inRange(imgHSV, lower, upper, mask);
        // hmin, smin, vmin, hmax, smax, vmax;
        cout << hmin << "," << smin << "," << vmin << "," << hmax << "," << smax << "," << vmax << endl;
        imshow("Image", img);
        imshow("Mask", mask);
        waitKey(1);
    }
}