// vidDisplay.cpp
// Name: Mihir Chitre, Aditya Gurnani
// Date: 01/23/2024
// Purpose: Display live video from a camera in multiple modes.

#include <opencv2/opencv.hpp>
#include <iostream>
#include <ctime>
#include <chrono>
#include "faceDetect.h" 

int greyscale(cv::Mat &src, cv::Mat &dst);
int sepiaTone(cv::Mat &src, cv::Mat &dst);
int blur5x5_1(cv::Mat &src, cv::Mat &dst);
int blur5x5_2(cv::Mat &src, cv::Mat &dst);
int sobelX3x3(cv::Mat &src, cv::Mat &dst);
int sobelY3x3(cv::Mat &src, cv::Mat &dst);
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);
int embossEffect(cv::Mat &src, cv::Mat &dst);
int colorfulFaceGrayscaleBackground(cv::Mat &src, cv::Mat &dst, std::vector<cv::Rect> &faces);

// Function to get current time in milliseconds using chrono library
double getTime()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto epoch = now_ms.time_since_epoch();
    auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);
    return value.count() / 1000.0;
}

int main(int argc, char *argv[])
{
    // Load an image and apply blur effect to check for processing time
    cv::Mat image = cv::imread("film.jpg");
    cv::Mat blurredImage;
    cv::Mat sobelX, sobelY;
    bool faceDetectionEnabled = false; 
    int levels;
    int brightnessAdjustment = 0;
    if (image.empty())
    {
        std::cerr << "Could not open or find the image 'film.jpg'" << std::endl;
    }
    else
    {
        const int Ntimes = 10; 

        auto startTime = std::chrono::high_resolution_clock::now();

        // Applying blur filter 10 times
        for (int i = 0; i < Ntimes; i++)
        {
            blur5x5_1(image, blurredImage);
            image = blurredImage.clone(); 
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        std::cout << "Total blur processing time for " << Ntimes << " iterations: " << elapsed.count() << " seconds" << std::endl;

        startTime = std::chrono::high_resolution_clock::now();

        // Applying efficient blur filter 10 times
        for (int i = 0; i < Ntimes; i++)
        {
            blur5x5_2(image, blurredImage);
            image = blurredImage.clone(); 
        }

        endTime = std::chrono::high_resolution_clock::now();
        elapsed = endTime - startTime;
        std::cout << "Total faster blur processing time for " << Ntimes << " iterations: " << elapsed.count() << " seconds" << std::endl;
    }

    // Starting live video stream
    cv::VideoCapture capdev(0); 
    if (!capdev.isOpened())
    {
        std::cerr << "Unable to open video device" << std::endl;
        return -1;
    }

    cv::Size refS((int)capdev.get(cv::CAP_PROP_FRAME_WIDTH),
                  (int)capdev.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::cout << "Expected size: " << refS.width << " " << refS.height << std::endl;

    cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
    cv::Mat frame, processedFrame;
    std::vector<cv::Rect> faces; 
    char mode = 'c';

    for (;;)
    {
        capdev >> frame;
        if (frame.empty())
        {
            std::cerr << "Frame is empty" << std::endl;
            break;
        }

        char key = cv::waitKey(10);
        if (key != -1)
        {
            // Perform face detection for colorful face effect and face detection mode
            if (key == 'f')
            {
                faceDetectionEnabled = !faceDetectionEnabled; 
                std::cout << "Face detection: " << (faceDetectionEnabled ? "ON" : "OFF") << std::endl;
            }
            if (key == 't' || faceDetectionEnabled)
            {
                cv::Mat grey;
                cv::cvtColor(frame, grey, cv::COLOR_BGR2GRAY);
                detectFaces(grey, faces);
            }

            // Perform checks for brightness adjustment feature
            if (key == '+')
            {
                brightnessAdjustment += 10; 
            }
            else if (key == '-')
            {
                brightnessAdjustment -= 10; 
            }

            // Perform checks for other keys
            if (key == 'q')
            {
                break;
            }
            else if (key == 's')
            {
                std::time_t now = std::time(nullptr);
                char buffer[40];
                std::strftime(buffer, 40, "saved_frame_%Y%m%d_%H%M%S.jpg", std::localtime(&now));
                std::string filename = buffer;
                cv::imwrite(filename, processedFrame);
                std::cout << "Saved frame to " << filename << std::endl;
            }
            else
            {
                mode = key;
            }
        }

        switch (mode)
        {
        case 'g': // Default grayscale filter
            cv::cvtColor(frame, processedFrame, cv::COLOR_BGR2GRAY);
            break;
        case 'h': // Custom grayscale filter
            if (greyscale(frame, processedFrame) != 0)
            {
                std::cerr << "Error in grayscale conversion" << std::endl;
            }
            break;
        case 'p': // Sepia filter
            if (sepiaTone(frame, processedFrame) != 0)
            {
                std::cerr << "Error applying sepia filter" << std::endl;
            }
            break;
        case 'b': // Blur filter
            if (blur5x5_2(frame, processedFrame) != 0)
            {
                std::cerr << "Error applying blur effect" << std::endl;
            }
            break;
        case 'x': // Sobel X filter
            sobelX3x3(frame, processedFrame);
            cv::convertScaleAbs(processedFrame, processedFrame); // Convert to viewable format
            break;
        case 'y': // Sobel Y filter
            sobelY3x3(frame, processedFrame);
            cv::convertScaleAbs(processedFrame, processedFrame); // Convert to viewable format
            break;
        case 'm': // Gradient magnitude
            sobelX3x3(frame, sobelX);
            sobelY3x3(frame, sobelY);
            magnitude(sobelX, sobelY, processedFrame);
            break;
        case 'l': // Blur and quantize with 10 levels
            levels = 10; 
            blurQuantize(frame, processedFrame, levels);
            break;
        case 'e': // Emboss effect
            embossEffect(frame, processedFrame);
            break;
        case 't': // Colorful face with grayscale background
            colorfulFaceGrayscaleBackground(frame, processedFrame, faces);
            break;
        default:
            processedFrame = frame.clone();
            break;
        }

        // Placing boxes around the detected faces when face detection is enabled
        if (faceDetectionEnabled)
        {
            drawBoxes(processedFrame, faces); // Draw boxes around detected faces
        }

        // Perform brightness adjustment
        if (brightnessAdjustment != 0)
        {
            processedFrame.convertTo(processedFrame, -1, 1, brightnessAdjustment);
        }
        cv::imshow("Video", processedFrame);
    }

    capdev.release();
    cv::destroyAllWindows();
    return 0;
}
