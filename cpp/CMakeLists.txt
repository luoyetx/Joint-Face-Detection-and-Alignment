project(JFDA)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV)

include(mini-caffe/mini-caffe.cmake)

add_executable(jfda main.cpp jfda.hpp jfda.cpp)
target_link_libraries(jfda caffe ${OpenCV_LIBS})

add_executable(video video.cpp jfda.hpp jfda.cpp)
target_link_libraries(video caffe ${OpenCV_LIBS})
