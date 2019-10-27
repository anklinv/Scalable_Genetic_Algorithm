# cmake_minimum_required(VERSION <specify CMake version here>)
project(Distributed_Genetic_Algorithm)

set(CMAKE_CXX_COMPILER mpicxx)

set(CMAKE_CXX_STANDARD 17)

# SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
# SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
# SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")

add_executable(Distributed_Genetic_Algorithm main.cpp logging/logging.cpp sequential/travelling_salesman_problem.cpp sequential/travelling_salesman_problem.hpp island/island.hpp island/island.cpp)
