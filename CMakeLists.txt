cmake_minimum_required(VERSION 3.22)
project(hpc_task1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20 -fopenmp -static-libgcc -static-libstdc++")

add_executable(hpc_task1 main.cpp)
