cmake_minimum_required(VERSION 3.10...3.18)
project(cuda_ransac LANGUAGES CXX CUDA)

add_definitions(-w)

# build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# install
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install)

add_definitions(-DCOMPILEDWITHC14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_COMPILER "nvcc")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC") 
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC") 

find_package(CUDA REQUIRED)

include_directories(
  ${CMAKE_SOURCE_DIR}/thirdparty/
  ${CMAKE_SOURCE_DIR}/thirdparty/eigen-3.4.0
  /usr/local/cuda/include/)

include_directories(src)
add_subdirectory(src)

