project(DGA)

set(CMAKE_CXX_COMPILER mpicxx)

set(CMAKE_CXX_STANDARD 17)

add_executable(DGA driver.cpp travelling_salesman_problem.cpp travelling_salesman_problem.hpp) 
