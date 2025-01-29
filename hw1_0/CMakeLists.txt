cmake_minimum_required(VERSION 3.17)
project(games101_hw1)

find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
set(CMAKE_CXX_STANDARD 11)

add_executable(games101_hw1 main.cpp)