#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <chrono>
#include <array>
#include <cassert>
#include <math.h>

#include <thrust/sort.h>

#define POP_SIZE 1000
#define PROB_SIZE 48
#define EPOCHS 100
#define ISLANDS 4
#define POP(i,j) population[i * PROB_SIZE + j]
#define DIST(i,j) problem[i * PROB_SIZE + j]

//https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
//https://github.com/thrust/thrust/wiki/Quick-Start-Guide#fancy-iterators

using namespace std;

__global__ void rank_individuals(int *population, float *problem, float *fitness, float *sorted_fitness) {
	//blockDim: number of threads in block
	//gridDim: number of blocks in grid
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = index; i < POP_SIZE; i++) {
		for (int j = 0; j < PROB_SIZE - 1; j++) {
			fitness[i] += DIST(j,j+1);
		}
		fitness[i] += DIST(PROB_SIZE - 1, 0);
		//fitness[i] = 1/fitness[i];
	}
    __synchthreads();
    thrust::sort(fitness,fitness+sorted_fitness, [this] (int i, int j) {
        return fitness[i] < fitness[j];
    }
    __syncthreads();
    for (int i = index; i < POP_SIZE; i++) {
        sorted_fitness[i] = 1/sorted_fitness[i];
    }
}

__global__ void

int main (void) {

	//arrays needed for problem
	int *population;
	float *problem;
	float *fitness;

	//random device
	random_device rd;
	mt19937 gen = mt19937(rd());

	//Unified memory (accessible from CPU and GPU)
	cudaMallocManaged(&problem, PROB_SIZE * PROB_SIZE *sizeof(float));
	cudaMallocManaged(&population, POP_SIZE * PROB_SIZE *sizeof(int));
	cudaMallocManaged(&fitness, POP_SIZE *sizeof(float));

	//initialize sequence of matrix indices to be used for initializing the population
	vector<int> tmp_indices(PROB_SIZE);
    	for (int i = 0; i < PROB_SIZE; ++i) {
        	tmp_indices[i] = i;
    	}

	//initialize problem (random for now, from file later)
	for (int i = 0; i < PROB_SIZE * PROB_SIZE; i++) {
		problem[i] = rand() % 100;
	}

	//initialize population (random) and fitness (zero)
	for (int i = 0; i < POP_SIZE; ++i) {
        	shuffle(tmp_indices.begin(), tmp_indices.end(), gen);
        	for (int j = 0; j < PROB_SIZE; ++j) {
            		POP(i,j) = tmp_indices[j];
        	}
		fitness[i] = 0.0f;
    	}

	//run GA
	for (int i = 0; i < EPOCHS; i++){
		//begin epoch by ranking fitness of population
		rank_individuals<<<ISLANDS, POP_SIZE / ISLANDS>>>(population,problem,fitness);

		//wait till previous kernel call returns
		cudaDeviceSynchronize();
	}

	cudaFree(population);
	cudaFree(problem);
	cudaFree(fitness);
	
	return 0;
}
