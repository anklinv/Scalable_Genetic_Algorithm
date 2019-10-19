# Scalable_Genetic_Algorithm
 Scalable implementation for the course project of Design of Parallel and High Performance Computing HS19

## How to run
You need mpicxx, mpiexec (any MPI implementation) and also cmake
(install this using `sudo apt install cmake g++`)

Then you first run cmake:
`cmake .`
Compile the program:
`make`
Run the program:
`mpiexec -np <number of processes> ./Distributed_Genetic_Algorithm`

 # Setup

 ### Justin's setup:
 #### MacBook Pro:
* Processor Name:    Intel Core i7
* Processor Speed:    2.3 GHz
* Number of Processors:    1
* Total Number of Cores:    4
* L2 Cache (per Core):    256 KB
* L3 Cache:    6 MB
* Hyper-Threading Technology:    Enabled
* Memory:    8 GB (2 banks, 1600MHz)
#### Compiler:
* gcc version: 4.2.1 (possibly too outdated)
* Apple clang version 11.0.0 (clang-1100.0.33.8)
* Target: x86_64-apple-darwin18.7.0
* Thread model: posix

 # Progress

 ## Miscelaneous

 * write python preprocessing scripts: create node edge incidence matrix: done
 * python postprocessing: visualization of results via plots etc.

 ## 1st experiment: sequential implementation of GA

 * Working implementation by Valentin in C++
 * Done, needs to be benchmarked
 * Maybe some optimization and cleaning can be done
 * reads inputs from csv file as opposed to raw txt file
 * Implementation in C started by Justin: so far can only read node incidence matrix from a .csv file

 ## 2nd experiment: naive parallel implementation with MPI

 ## 3rd experiment: island based implementation

 ## 4th experiment: asynchronous migration

 ## 5th experiment: RDMA migration
