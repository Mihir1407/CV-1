cmake_minimum_required(VERSION 3.10)

project(FirstProject2)

set(OpenCV_DIR C:/opencv/build)

find_package(OpenCV REQUIRED)

# Added executable for imgDisplay.cpp
add_executable(ImgDisplay src/imgDisplay.cpp)
target_link_libraries(ImgDisplay ${OpenCV_LIBS})

# Added executable for VidDisplay.cpp
add_executable(VidDisplay src/vidDisplay.cpp src/filter.cpp src/faceDetect.cpp)
target_link_libraries(VidDisplay ${OpenCV_LIBS})
