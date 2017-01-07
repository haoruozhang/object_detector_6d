project(ObjectDetector)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeScripts")
set(HDF5_USE_STATIC_LIBRARIES 1)

find_package(Glog REQUIRED)
find_package(GFlags REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenCV 2.4.10 REQUIRED)
find_package(PCL REQUIRED)
find_package(VTK 5.10 REQUIRED)
find_package(CUDA REQUIRED)
find_package(LMDB REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Caffe REQUIRED)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

aux_source_directory(./PatchGen/src SRC_PATCHGEN)
aux_source_directory(./HoughForest/src SRC_HOUGHFOREST)

file(GLOB INCLUDE_LIST
"./PatchGen/include/*.h"
"./PatchGen/include/*.hpp"
"./PatchGen/include/cuda/*.h"
"./HoughForest/include/*.h"
"./HoughForest/include/*.hpp"
"./HoughForest/include/cuda/*.h")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=sm_50
)

include_directories(
    /usr/include
    /usr/local/include/eigen3    
    ./PatchGen/include
    ./HoughForest/include
    ./PatchGen/include/cuda    
    ${GLOG_INCLUDE_DIRS}
    ${gflags_INCLUDE_DIRS}    
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${VTK_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${LMDB_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRECTORIES}
    ${PROTOBUF_INCLUDE_DIR}
    ${Caffe_INCLUDE_DIR}
)

CUDA_ADD_LIBRARY(surface_normals_generator PatchGen/src/cuda/surface_normals.cu PatchGen/src/cuda/cuda_utils.cu SHARED)
CUDA_ADD_LIBRARY(patch_extractor PatchGen/src/cuda/patch_extractor.cu PatchGen/src/cuda/cuda_utils.cu SHARED)


link_directories(
    ${VTK_LIBRARY_DIRS}
    ${Boost_LIBRARY_DIRS}
)

add_executable(PatchGen ${SRC_PATCHGEN} ${INCLUDE_LIST})
add_executable(HoughForest ${SRC_HOUGHFOREST} ${INCLUDE_LIST})

target_link_libraries(PatchGen
    ${VTK_LIBRARIES}
    ${OpenCV_LIBS}
    ${BOOST_LIBRARIES}
    gflags
    glog
    lmdb
    ${PROTOBUF_LIBRARIES}
    cuda
    patch_extractor
    surface_normals_generator
    ${Caffe_LIBS}
)

target_link_libraries(HoughForest
    glog
    ${GFLAGS_LIBRARIES}
    pthread
    ${CUDA_LIBRARIES}    
    ${BOOST_LIBRARIES}
    surface_normals_generator
    patch_extractor
    ${OpenCV_LIBS}
    ${VTK_LIBRARIES}
    ${PCL_LIBRARIES}
    ${Caffe_LIBS}
    protobuf
)



