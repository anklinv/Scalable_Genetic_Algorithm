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
#define MUTATION 0.15   //rate
#define MIGRATION 5    //integer determining how many members migrate from one island
#define POP(i,j) population[i * PROB_SIZE + j]
#define NPOP(i,j) new_population[i * PROB_SIZE + j]
#define DIST(i,j) problem[i * PROB_SIZE + j]
#define ISLAND_MEMBERS POP_SIZE / ISLANDS

//https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
//https://github.com/thrust/thrust/wiki/Quick-Start-Guide#fancy-iterators

using namespace std;

__global__ void rank_individuals(int *population, float *problem, float *fitness, float *sorted_fitness) {
	//blockDim: number of threads in block
	//gridDim: number of blocks in grid
	int index = threadIdx.x + blockIdx.x * blockDim.x;
    for (int j = 0; j < PROB_SIZE - 1; j++) {
        fitness[index] += DIST(j,j+1);
    }
    fitness[index] += DIST(PROB_SIZE - 1, 0);
    fitness[index] = 1/fitness[index];
	__syncthreads();
}

__global__ void crossover_and_migrate(int *population, float *fitness, int *prefix_sort_fintess, int *new_population) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
    int temp[PROB_SIZE * MIGRATION];
	int parent1_id;
	int parent2_id;
    int migrant_1, migrant_2, island_1, island_2;
	float roulette;
    int r_1, r_2;
	int existing;
    int migrant_index;

    //get indices of the two parents by a random variable that has the normalized fitness values as distributions (use the thrust lower bound on the prefix sorted fitness values)
    roulette = rand() % prefix_sort_fitness[POP_SIZE-1];
    parent1_id = thrust::lower_bound(thrust::device, prefix_sort_fitness, prefix_sort_fitness + POP_SIZE, roulette) - prefix_sort - 1;
    roulette = rand() % prefix_sort_fitness[POP_SIZE-1];
    parent2_id = thrust::lower_bound(thrust::device, prefix_sort_fitness, prefix_sort_fitness + POP_SIZE, roulette) - prefix_sort - 1;
	
    //fill the first half of the child with the first parent
    for (int j = 0; j < int(PROB_SIZE/2); j++){
        NPOP(index,j) = POP(parent1_id,j);
		
    //fill the rest of the child with -1 entries such that it is no longer garbage
    for (int j = int(PROB_SIZE/2); j < PROB_SIZE; j++) {
        NPOP(index,j) = -1;
		
    //for the second half of the child we only choose entries from parent 2 that aren't already in the child
    for (int j = int(PROB_SIZE/2); j < PROB_SIZE; j++) {
        for (int k = 0; k < PROB_SIZE; k++) {
            existing = thrust::find(thrust::device, *NPOP(index,0), *NPOP(index,j), POP(parent2_id,k)) - *NPOP(index,0);
            if (existing != j){
                NPOP(index,j) = existing;
                break;
            }
        }
    }
    //perform the migration
    if (threadIdx.x == 0) {
        island_1 = index % ISLANDS;
        island_2 = (island_1 + 1) % ISLANDS;
        r_1 = rand() % ISLAND_MEMBERS;
        r_2 = rand() % ISLAND_MEMBERS;
    }
	__syncthreads();
    //Each thread takes a single gene from one of five
    if (threadIdx.x != 0 && threadIdx.x < PROB_SIZE * MIGRATION) {
        migrant_index = threadIdx.x - 1;
        temp[migrant_index] = NPOP(island_1 * ISLAND_MEMBERS + r_1, remainder(migrant_index, PROB_SIZE));
        NPOP(island_1 * POP_SIZE / ISLAND + r_1, remainder(migrant_index, PROB_SIZE)) = NPOP(island_2 * ISLAND_MEMBERS + r_2, remainder(migrant_index, PROB_SIZE))
        NPOP(island_2 * POP_SIZE / ISLAND + r_2, remainder(migrant_index, PROB_SIZE)) = temp[migrant_index];
    }
}

__global__ void mutate(int * population){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int temp;
    default_random_engine generator;
    for (int j = 0; j < PROB_SIZE; j++){
        bernoulli_distribution ber(MUTATION);
        if (ber(generator)){
            for (int k = 0; k < PROB_SIZE; k++) {
                bernoulli_distribution ber(MUTATION);
                if (ber(generator)){
                    temp = POP(index,j);
                    POP(index,j) = POP(index,k);
                    POP(index,k) = temp;
                    break;
                }
            }
        }
    }
}

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
		rank_individuals<<<ISLANDS, ISLAND_MEMBERS>>>(population,problem,fitness);
		
		//perform a sort of the fitness values to implement the elitism strategy
        for (int j = 0; j < ISLANDS; j++) {
            thrust::sort_by_key(sorted_fitness + j * ISLAND_MEMBERS, sorted_fitness + (j + 1) * ISLAND_MEMBERS, fitness + j * ISLAND_MEMBERS);
        }

		//perform a prefix sum of the fitness in order to easily perform roulette wheel selection of a suitable parent
        for (int j = 0; j < ISLANDS; j++) {
            thrust::inclusive_scan(fitness + j * ISLAND_MEMBERS, fitness + (j + 1) * ISLAND_MEMBERS, prefix_sort_fitness + j * ISLAND_MEMBERS);
        }
        
		//perform crossover on the islands
		crossover_and_migrate<<<ISLANDS, ISLAND_MEMBERS>>>(population,fitness,prefix_sort_fitness,new_population);
  
        //mutate the population
        mutate<<<ISLANDS, ISLAND_MEMBERS>>>(population);
	}

	cudaFree(population);
	cudaFree(problem);
	cudaFree(fitness);
	cudaFree(sorted_fitness);
	cudaFree(prefix_sort_fintess);
	cudaFree(new_population);
	
	return 0;
}
