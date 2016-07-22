project(JFDA)
cmake_minimum_required(VERSION 2.8)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
  message("OPENMP FOUND")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

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

# libjfda
add_library(jfda STATIC src/jfda.hpp src/jfda.cpp)
target_link_libraries(jfda ${LIBCAFFE} ${OpenCV_LIBS})

# detect wider
add_executable(wider src/wider.cpp)
target_link_libraries(wider jfda)

# simple test
add_executable(main src/main.cpp)
target_link_libraries(main jfda)

# fddb test
add_executable(fddb src/fddb.cpp)
target_link_libraries(fddb jfda)
