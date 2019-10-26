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

## How to run on leonhard
#### Load C++ compiler
```module load gcc/8.2.0```
#### Load MPI
```module load openmpi/4.0.1```
#### Compile
```mpiCC main.cpp sequential/travelling_salesman_problem.cpp logging/logging.cpp```
#### Test if it works
Do not do this exessively as it runs on the login node. Just use for sanity check.
```mpirun -np 2 ./Distributed_Genetic_Algorithm```
#### Submit job
Use the following command to submit a job on Leonhard:

```bsub -n 4 -W 00:10 -o log_test -R "rusage[mem=1024]" mpirun ./Distributed_Genetic_Algorithm```

`-n` specifies the number of cores to use. Note: in the mpirun call you do not need to actually specify `-np`.

`-W 00:10` specifies that we want to run it for at most 10 minutes (after which the process gets killed). As we should not submit jobs that run longer than 30 minutes, keep this at most at `00:30`, but if the value is lower it might get scheduled faster.

`-o log_test` specifies the name of the logfile. If unspecified then the cluster will create a cryptic name.

`-R "rusage[mem=1024]"` specifies how much memory is used PER CORE.

#### Look at all jobs
`bjobs -a` lists all jobs for that user. You can also use `bbjobs -a` to get more information.
#### Peek at the output (possible as soon as the job is running)
`bpeek -f` shows the console output of the job (if only one is active you do not need to specify the job-id)
#### Read the logfile
`vi log_test`

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
