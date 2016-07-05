project(JFDA)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
include(mini-caffe/mini-caffe.cmake)

add_executable(jfda main.cpp)
target_link_libraries(jfda libcaffe ${OpenCV_LIBS})
