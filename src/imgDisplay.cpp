    #include <opencv2/opencv.hpp>
    #include <iostream>

    int main() {
        cv::Mat image = cv::imread("film.jpg");  
        if(image.empty()) {
            std::cout << "Could not open or find the image" << std::endl;
            return -1;
        }
        cv::namedWindow("Display window", cv::WINDOW_NORMAL);
        cv::imshow("Display window", image);

        while(true) {
            char key = cv::waitKey(0);
            if(key == 'q') {
                break;
            }
        }
        return 0;
    }