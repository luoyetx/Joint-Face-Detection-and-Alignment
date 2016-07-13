project(JFDA)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)

if(WIN32)
include(mini-caffe/mini-caffe.cmake)
set(LIBCAFFE libcaffe)
else()
set(CAFFE_HOME $ENV{CAFFE_HOME})
include_directories(${CAFFE_HOME}/include)
include_directories(${CAFFE_HOME}/build/src)
include_directories(/usr/local/cuda/include)
link_directories(${CAFFE_HOME}/build/lib)
set(LIBCAFFE caffe)
endif()

include_directories(src)
file(GLOB SRC src/*.cpp src/*.hpp)

add_executable(jfda ${SRC})
target_link_libraries(jfda ${LIBCAFFE} ${OpenCV_LIBS})
