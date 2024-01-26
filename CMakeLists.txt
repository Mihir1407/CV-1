cmake_minimum_required(VERSION 3.10)

project(CV-1)

# Set the directory where CMake can find OpenCV
set(OpenCV_DIR D:/opencv/build)

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Include the OpenCV include directories
include_directories(${OpenCV_INCLUDE_DIRS}) 

# Added executable for imgDisplay.cpp
add_executable(ImgDisplay src/imgDisplay.cpp)
target_link_libraries(ImgDisplay ${OpenCV_LIBS})

# Added executable for color.cpp
add_executable(color src/color.cpp)
target_link_libraries(color ${OpenCV_LIBS})

# Added executable for VidDisplay.cpp
add_executable(VidDisplay src/vidDisplay.cpp src/filter.cpp src/faceDetect.cpp)
target_link_libraries(VidDisplay ${OpenCV_LIBS})
