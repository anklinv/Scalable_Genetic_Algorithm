# cmake_minimum_required(VERSION <specify CMake version here>)
project(Distributed_Genetic_Algorithm)

set(CMAKE_CXX_STANDARD 17)

add_executable(Distributed_Genetic_Algorithm main.cpp sequential/city.cpp sequential/city.hpp sequential/travelling_salesman_problem.cpp sequential/travelling_salesman_problem.hpp sequential/population.cpp sequential/population.hpp sequential/city_generator.cpp sequential/city_generator.hpp)