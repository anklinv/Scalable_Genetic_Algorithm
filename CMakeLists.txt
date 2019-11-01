# cmake_minimum_required(VERSION <specify CMake version here>)
project(Distributed_Genetic_Algorithm)

set(CMAKE_CXX_COMPILER mpicxx)

set(CMAKE_CXX_STANDARD 17)

add_executable(Distributed_Genetic_Algorithm main.cpp logging/logging.cpp sequential/travelling_salesman_problem.cpp island/island.cpp)
