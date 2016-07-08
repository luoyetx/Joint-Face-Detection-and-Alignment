project(JFDA)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
include(mini-caffe/mini-caffe.cmake)

include_directories(src)
file(GLOB SRC src/*.cpp src/*.hpp)

add_executable(jfda ${SRC})
target_link_libraries(jfda libcaffe ${OpenCV_LIBS})
