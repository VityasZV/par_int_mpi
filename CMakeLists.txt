cmake_minimum_required(VERSION 3.6)
project(p_i)

find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")


SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_CXX_STANDARD 17)


add_executable(p_i parralel_integral_mpi.cpp parralel_integral.cpp mpi_info.cpp)

