# cmake_minimum_required(VERSION <specify CMake version here>)
project(Distributed_Genetic_Algorithm)

set(CMAKE_CXX_COMPILER mpicxx)

set(CMAKE_CXX_STANDARD 17)

#[[ cmake_minimum_required(VERSION 3.15) ]]

SET(CMAKE_CXX_FLAGS  "-mavx2 -O3 -funroll-loops")
#[[ -mbmi2 for bit manipulation ]]
#[[ -mavx2 -O3 -funroll-loops for optimizations ]]

add_definitions(-D Int=int -D Real=float -D microbenchmark_breed)
#[[ currently only Real={float} and Int={uint16_t, uint32_t, int} are supported ]]
#[[ -D microbenchmark_breed  for microbenchmarks ]]

add_executable(Distributed_Genetic_Algorithm main.cpp logging/logging.cpp sequential/travelling_salesman_problem.cpp island/island.cpp)
