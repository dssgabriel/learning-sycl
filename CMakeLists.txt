# Support a range of CMake versions, starting form C++20 support
cmake_minimum_required(VERSION 3.12...3.24)

project(learn-sycl
  VERSION 0.1.0
  DESCRIPTION "Learning SYCL and comparing performance"
  LANGUAGES CXX)

# Require Intel DPC++ (Intel's SYCL implementation)
find_package(IntelDPCPP REQUIRED)

add_library(utils OBJECT src/utils.cpp include/utils.hpp)
target_include_directories(utils PUBLIC include)
target_compile_features(utils PUBLIC cxx_std_17)

add_library(drivers OBJECT include/drivers.hpp)
target_include_directories(drivers PUBLIC include)
target_compile_features(drivers PUBLIC cxx_std_17)

add_library(kernels OBJECT include/kernels.hpp)
target_include_directories(kernels PUBLIC include)
target_compile_features(kernels PUBLIC cxx_std_17)

add_executable(learn-sycl src/main.cpp)
target_link_libraries(learn-sycl PUBLIC utils kernels)
target_compile_features(learn-sycl PUBLIC cxx_std_17)
