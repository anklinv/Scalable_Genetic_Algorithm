# cmake_minimum_required(VERSION <specify CMake version here>)
project(Distributed_Genetic_Algorithm)

set(CMAKE_CXX_COMPILER mpicxx)

set(CMAKE_CXX_STANDARD 17)

add_executable(DGA main.cpp ../logging/logging.cpp travelling_salesman_problem.cpp travelling_salesman_problem.hpp island.cpp island.hpp)
