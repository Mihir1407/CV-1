// filter.cpp
// Name: Mihir Chitre, Aditya Gurnani
// Date: 01/23/2024
// Purpose: Contains image manipulation functions

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Custom grayscale function with non-standard weights
int greyscale(cv::Mat &src, cv::Mat &dst)
{
    if (src.empty() || src.channels() != 3)
    {
        return -1; 
    }
    dst = cv::Mat(src.size(), src.type());
    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);

            // Custom weights for each channel
            double blueWeight = 0.3;  
            double greenWeight = 0.3; 
            double redWeight = 0.4;   

            uchar grayValue = static_cast<uchar>(pixel[0] * blueWeight +
                                                 pixel[1] * greenWeight +
                                                 pixel[2] * redWeight);

            dst.at<cv::Vec3b>(y, x) = cv::Vec3b(grayValue, grayValue, grayValue);
        }
    }

    return 0;
}

// Function to apply a sepia tone to an image
int sepiaTone(cv::Mat &src, cv::Mat &dst)
{
    if (src.empty() || src.channels() != 3)
    {
        return -1; 
    }
    dst = cv::Mat(src.size(), src.type());
    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);

            // Original colors
            float blue = pixel[0];
            float green = pixel[1];
            float red = pixel[2];

            // Calculate new color values as per the sepia formula
            float newRed = std::min(255.0f, 0.272f * red + 0.534f * green + 0.131f * blue);
            float newGreen = std::min(255.0f, 0.349f * red + 0.686f * green + 0.168f * blue);
            float newBlue = std::min(255.0f, 0.393f * red + 0.769f * green + 0.189f * blue);

            dst.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<uchar>(newBlue),
                                                static_cast<uchar>(newGreen),
                                                static_cast<uchar>(newRed));
        }
    }

    return 0; 
}

// Function to apply a 5x5 blur filter
int blur5x5_1(cv::Mat &src, cv::Mat &dst)
{
    if (src.empty() || src.channels() != 3)
    {
        return -1; 
    }
    dst = src.clone();

    // Gaussian kernel matrix
    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}};
    int kernelSum = 140; 
    for (int y = 2; y < src.rows - 2; y++)
    {
        for (int x = 2; x < src.cols - 2; x++)
        {
            int sumB = 0, sumG = 0, sumR = 0;

            for (int ky = -2; ky <= 2; ky++)
            {
                for (int kx = -2; kx <= 2; kx++)
                {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(y + ky, x + kx);
                    sumB += pixel[0] * kernel[ky + 2][kx + 2];
                    sumG += pixel[1] * kernel[ky + 2][kx + 2];
                    sumR += pixel[2] * kernel[ky + 2][kx + 2];
                }
            }

            dst.at<cv::Vec3b>(y, x) = cv::Vec3b(
                sumB / kernelSum,
                sumG / kernelSum,
                sumR / kernelSum);
        }
    }

    return 0;
}

// Function to apply a faster 5x5 blur filter using separable filters
int blur5x5_2(cv::Mat &src, cv::Mat &dst)
{
    if (src.empty() || src.channels() != 3)
    {
        return -1;
    }

    // Kernel
    int kernel[5] = {1, 2, 4, 2, 1};
    int kernelSum = 10; 
    dst = cv::Mat(src.size(), src.type());

    // Temporary image
    cv::Mat temp(src.size(), src.type());

    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 2; x < src.cols - 2; x++)
        {
            int sumB = 0, sumG = 0, sumR = 0;
            for (int k = -2; k <= 2; k++)
            {
                cv::Vec3b pixel = src.ptr<cv::Vec3b>(y)[x + k];
                sumB += pixel[0] * kernel[k + 2];
                sumG += pixel[1] * kernel[k + 2];
                sumR += pixel[2] * kernel[k + 2];
            }
            temp.ptr<cv::Vec3b>(y)[x] = cv::Vec3b(sumB / kernelSum, sumG / kernelSum, sumR / kernelSum);
        }
    }

    // Applying vertical blur
    for (int y = 2; y < src.rows - 2; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            int sumB = 0, sumG = 0, sumR = 0;
            for (int k = -2; k <= 2; k++)
            {
                cv::Vec3b pixel = temp.ptr<cv::Vec3b>(y + k)[x];
                sumB += pixel[0] * kernel[k + 2];
                sumG += pixel[1] * kernel[k + 2];
                sumR += pixel[2] * kernel[k + 2];
            }
            dst.ptr<cv::Vec3b>(y)[x] = cv::Vec3b(sumB / kernelSum, sumG / kernelSum, sumR / kernelSum);
        }
    }

    return 0;
}

// Function to apply a 3x3 Sobel X filter
int sobelX3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.channels() != 3) {
        return -1;
    }

    dst = cv::Mat(src.size(), CV_16SC3);

    // Sobel X kernel
    int kernel[3] = {-1, 0, 1};

    for (int y = 0; y < src.rows; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            cv::Vec3s sum(0, 0, 0);
            for (int k = -1; k <= 1; k++) {
                cv::Vec3b pixel = src.at<cv::Vec3b>(y, x + k);
                sum[0] += pixel[0] * kernel[k + 1];
                sum[1] += pixel[1] * kernel[k + 1];
                sum[2] += pixel[2] * kernel[k + 1];
            }
            dst.at<cv::Vec3s>(y, x) = sum;
        }
    }

    return 0;
}

// Function to apply a 3x3 Sobel Y filter
int sobelY3x3(cv::Mat &src, cv::Mat &dst) {
    if (src.empty() || src.channels() != 3) {
        return -1;
    }

    dst = cv::Mat(src.size(), CV_16SC3);

    // Sobel Y kernel
    int kernel[3] = {-1, 0, 1};

    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3s sum(0, 0, 0);
            for (int k = -1; k <= 1; k++) {
                cv::Vec3b pixel = src.at<cv::Vec3b>(y + k, x);
                sum[0] += pixel[0] * kernel[k + 1];
                sum[1] += pixel[1] * kernel[k + 1];
                sum[2] += pixel[2] * kernel[k + 1];
            }
            dst.at<cv::Vec3s>(y, x) = sum;
        }
    }

    return 0;
}

// Function to generate a gradient magnitude image from X and Y Sobel images
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    if (sx.empty() || sy.empty() || sx.channels() != 3 || sy.channels() != 3) {
        return -1;
    }
    dst = cv::Mat(sx.size(), CV_8UC3);

    // Gradient magnitude calculation
    for (int y = 0; y < sx.rows; y++) {
        for (int x = 0; x < sx.cols; x++) {
            cv::Vec3s sxPixel = sx.at<cv::Vec3s>(y, x);
            cv::Vec3s syPixel = sy.at<cv::Vec3s>(y, x);
            cv::Vec3b magnitudePixel;

            for (int c = 0; c < 3; c++) {
                float mag = sqrt(sxPixel[c] * sxPixel[c] + syPixel[c] * syPixel[c]);
                magnitudePixel[c] = static_cast<uchar>(std::min(mag, 255.0f)); // Scale and convert to uchar
            }

            dst.at<cv::Vec3b>(y, x) = magnitudePixel;
        }
    }

    return 0;
}

// Function to blur and quantize a color image
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    if (src.empty() || src.channels() != 3 || levels <= 0) {
        return -1;
    }
    cv::Mat blurredSrc;
    blur5x5_2(src, blurredSrc);  
    dst = cv::Mat(src.size(), src.type());
    int bucketSize = 255 / levels;

    // Image Quantization
    for (int y = 0; y < blurredSrc.rows; y++) {
        for (int x = 0; x < blurredSrc.cols; x++) {
            cv::Vec3b pixel = blurredSrc.at<cv::Vec3b>(y, x);
            cv::Vec3b quantizedPixel;

            for (int c = 0; c < 3; c++) {
                int quantizedValue = (pixel[c] / bucketSize) * bucketSize;
                quantizedPixel[c] = static_cast<uchar>(quantizedValue);
            }

            dst.at<cv::Vec3b>(y, x) = quantizedPixel;
        }
    }

    return 0;
}

//Function to apply embossing effect on a color image.
int embossEffect(cv::Mat &src, cv::Mat &dst) {
    cv::Mat grey; // converting to gray scale
    cvtColor(src, grey, cv::COLOR_BGR2GRAY);

    // Applying Sobel filters in X and Y directions
    cv::Mat sobelX, sobelY;
    Sobel(grey, sobelX, CV_32F, 1, 0);
    Sobel(grey, sobelY, CV_32F, 0, 1);

    // Combine Sobel X and Y
    cv::Mat emboss = 0.5 * sobelX + 0.5 * sobelY;

    // Normalize and convert to 8-bit image
    normalize(emboss, emboss, 0, 255, cv::NORM_MINMAX);
    convertScaleAbs(emboss, dst);

    return 0;
}

//Function to grayscale the entire image except for the face
int colorfulFaceGrayscaleBackground(cv::Mat &src, cv::Mat &dst, std::vector<cv::Rect> &faces) {
    if (src.empty()) {
        return -1; 
    }

    cv::Mat colorSrc = src.clone();
    cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
    cv::cvtColor(dst, dst, cv::COLOR_GRAY2BGR); // Converting grayscale to 3 channels

    for (const auto &face : faces) {
        colorSrc(face).copyTo(dst(face)); // Copying color face regions to grayscale image
    }

    return 0; 
}

//Function to apply a thermal vision effect to an image
int thermalVisionEffect(const cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        return -1;
    }
    cv::Mat grayscale;
    cv::cvtColor(src, grayscale, cv::COLOR_BGR2GRAY);
    cv::applyColorMap(grayscale, dst, cv::COLORMAP_JET);  // Thermal effect

    return 0;
}

//Function to find and return the point at the center of the largest contour in the image.
cv::Point getContours(cv::Mat image) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    // Find contours in the image
    findContours(image, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Rect> boundRect(contours.size());
    cv::Point myPoint(0, 0);

    for (int i = 0; i < contours.size(); i++) {
        int area = contourArea(contours[i]);
        if (area > 1000) {
            float peri = arcLength(contours[i], true);
            std::vector<cv::Point> conPoly;
            approxPolyDP(contours[i], conPoly, 0.02 * peri, true);
            boundRect[i] = boundingRect(conPoly);
            myPoint.x = boundRect[i].x + boundRect[i].width / 2;
            myPoint.y = boundRect[i].y;
        }
    }
    return myPoint;
}

//Function to convert the source image to HSV color space and return centroids for contours of each color range
std::vector<std::vector<int>> findColor(cv::Mat img, const std::vector<std::vector<int>>& myColors) {
    cv::Mat imgHSV;
    cvtColor(img, imgHSV, cv::COLOR_BGR2HSV); // Converting the image from BGR to HSV color space
    std::vector<std::vector<int>> newPoints;

    for (int i = 0; i < myColors.size(); i++) {
        cv::Scalar lower(myColors[i][0], myColors[i][1], myColors[i][2]);
        cv::Scalar upper(myColors[i][3], myColors[i][4], myColors[i][5]);
        cv::Mat mask;
        inRange(imgHSV, lower, upper, mask);
        cv::Point myPoint = getContours(mask); //Get centroid of the largest contour
        if (myPoint.x != 0) {
            newPoints.push_back({myPoint.x, myPoint.y, i});
        }
    }
    return newPoints;
}

//FUnction to at specified points with specified colors
void paintMode(cv::Mat &img, std::vector<std::vector<int>> &newPoints, const std::vector<cv::Scalar> &myColorValues) {
    for (int i = 0; i < newPoints.size(); i++) {
        cv::circle(img, cv::Point(newPoints[i][0], newPoints[i][1]), 10, myColorValues[newPoints[i][2]], cv::FILLED);
    }
}

//Function to add a rain effect filter to an image
void rainEffect(cv::Mat &frame, int intensity) {
    for (int i = 0; i < intensity * 10; ++i) {
        int x = std::rand() % frame.cols;
        int y = std::rand() % frame.rows;
        int length = std::rand() % 15 + 5; // Length for raindrop
        int thickness = std::rand() % 3 + 1; // Thickness for raindrop
        cv::line(frame, cv::Point(x, y), cv::Point(x + length, y + length), cv::Scalar(255, 255, 255), thickness);
    }
}

