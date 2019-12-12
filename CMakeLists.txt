# cmake_minimum_required(VERSION <specify CMake version here>)
project(Distributed_Genetic_Algorithm)

set(CMAKE_CXX_COMPILER mpicxx)

set(CMAKE_CXX_STANDARD 17)

# cmake_minimum_required(VERSION 2.8)

SET(CMAKE_CXX_FLAGS  "-mavx2 -O3 -funroll-loops")

add_definitions(-D Int=uint16_t)
#[[add -D microbenchmark_breed to time the stuff in breed]]


add_executable(Distributed_Genetic_Algorithm main.cpp logging/logging.cpp sequential/travelling_salesman_problem.cpp island/island.cpp)
