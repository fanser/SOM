project(SOM)
cmake_minimum_required(VERSION 2.8)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -fopenmp -pthread -fPIC")

#You can set your opencv dependencies at here
include_directories(/data01/home/fanzhongyue/miniconda2/envs/caffe/include)
link_directories(/data01/home/fanzhongyue/install/caffe_dep/lib)

#For building the test 
AUX_SOURCE_DIRECTORY(./test SERVER_LIST)                  
include_directories(./src)
add_executable(test ${SERVER_LIST} ./src/base/circleTopo.cc ./src/som.cc)
target_link_libraries(test opencv_imgproc opencv_highgui opencv_core)

#For builiding SOM library
add_library(som ${SERVER_LIST} ./src/base/circleTopo.cc ./src/som.cc)
target_link_libraries(som opencv_imgproc opencv_highgui opencv_core)

#For building train bin
add_executable(train ./src/train_som.cc)
target_link_libraries(train som)

