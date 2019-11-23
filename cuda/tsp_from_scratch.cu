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
#include <thrust/scan.h>

#define POP_SIZE 1000
#define PROB_SIZE 48
#define EPOCHS 100
#define ISLANDS 4
#define POP(i,j) population[i * PROB_SIZE + j]
#define NPOP(i,j) new_population[i * PROB_SIZE + j]
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
	__syncthreads();
}

__global__ void crossover(int *population, float *fitness, int *prefix_sort_fintess, int *new_population) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int parent1_id;
	int parent2_id;
	float roulette;
	int existing;
	for (int i = index; i < POP_SIZE; i++) {

		//get indices of the two parents by a random variable that has the normalized fitness values as distributions (use the thrust lower bound on the prefix sorted fitness values) 
		roulette = rand() % prefix_sort_fitness[POP_SIZE-1];
		parent1_id = thrust::lower_bound(thrust::device, prefix_sort_fitness, prefix_sort_fitness + POP_SIZE, roulette) - prefix_sort - 1;
		roulette = rand() % prefix_sort_fitness[POP_SIZE-1];
		parent2_id = thrust::lower_bound(thrust::device, prefix_sort_fitness, prefix_sort_fitness + POP_SIZE, roulette) - prefix_sort - 1;
	
		//fill the first half of the child with the first parent
		for (int j = 0; j < int(PROB_SIZE/2); j++){
			NPOP(i,j) = POP(parent1_id,j);
		
		//fill the rest of the child with -1 entries such that it is no longer garbage
		for (int j = int(PROB_SIZE/2); j < PROB_SIZE; j++) {
			NPOP(i,j) = -1;
		
		//for the second half of the child we only choose entries from parent 2 that aren't already in the child
		for (int j = int(PROB_SIZE/2); j < PROB_SIZE; j++) {
			for (int k = 0; k < PROB_SIZE; k++) {
				existing = thrust::find(thrust::device, *NPOP(i,0), *NPOP(i,j), POP(parent2_id,k)) - *NPOP(i,0);
				if (existing != j){
					NPOP(i,j) = existing;
					break;
				}
			}
		}
	}
	__syncthreads();					
}

__global__ void

int main (void) {

	//arrays needed for problem
	int *population, *sorted_fitness, *new_population;
	float *problem, *fitness, *prefix_sort_fitness;

	//random device
	random_device rd;
	mt19937 gen = mt19937(rd());

	//Unified memory (accessible from CPU and GPU)
	cudaMallocManaged(&problem, PROB_SIZE * PROB_SIZE *sizeof(float));
	cudaMallocManaged(&population, POP_SIZE * PROB_SIZE *sizeof(int));
	cudaMallocManaged(&new_population, POP_SIZE * PROB_SIZE *sizeof(int));
	cudaMallocManaged(&fitness, POP_SIZE *sizeof(float));
	cudaMallocManaged(&sorted_fitness, POP_SIZE *sizeof(int));
	cudaMallocManaged(&prefix_sort_fitness, POP_SIZE *sizeof(float));

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
		sorted_fitness[i] = i;
    	}

	//run GA
	for (int i = 0; i < EPOCHS; i++){
		//begin epoch by ranking fitness of population
		rank_individuals<<<ISLANDS, POP_SIZE / ISLANDS>>>(population,problem,fitness);
		
		//perform a sort of the fitness values to implement the elitism strategy
		thrust::sort_by_key(sorted_fitness, sorted_fitness + POP_SIZE, fitness);

		//perform a prefix sum of the fitness in order to easily perform roulette wheel selection of a suitable parent
		thrust::inclusive_scan(fitness, fitness + POP_SIZE, prefix_sort_fitness);

		//perform crossover on the islands
		crossover<<<ISLANDS, POP_SIZE / ISLANDS>>>(population,fitness,prefix_sort_fitness,new_population);
	}

	cudaFree(population);
	cudaFree(problem);
	cudaFree(fitness);
	cudaFree(sorted_fitness);
	cudaFree(prefix_sort_fintess);
	cudaFree(new_population);
	
	return 0;
}
