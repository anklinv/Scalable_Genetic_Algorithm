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

#include <curand_kernel.h>

#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/extrema.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define POP_SIZE 1000
#define PROB_SIZE 48
#define EPOCHS 1000
#define ISLANDS 1
#define MUTATION_RATE 0.001f
#define MIGRATION 5    //integer determining how many members migrate from one island
#define POP(i,j) population[i * PROB_SIZE + j]
#define NPOP(i,j) new_population[i * PROB_SIZE + j]
#define DIST(i,j) problem[i * PROB_SIZE + j]
#define ISLAND_POP_SIZE (POP_SIZE / ISLANDS)

typedef float problem_t[PROB_SIZE][PROB_SIZE];

typedef int individual_t[PROB_SIZE];
typedef individual_t population_t[POP_SIZE];

typedef float fitness_t;
typedef fitness_t population_fitness_t[POP_SIZE];

//https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
//https://github.com/thrust/thrust/wiki/Quick-Start-Guide#fancy-iterators

using namespace std;

__global__ void rank_individuals(
    population_t population,
    problem_t problem,
    population_fitness_t fitness
) {
	//blockDim: number of threads in block
	//gridDim: number of blocks in grid
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    individual_t &i = population[index];
    float distance = 0;
    for (int j = 0; j < PROB_SIZE - 1; j++) {
        distance += problem[i[j]][i[j + 1]];
    }
    distance += problem[i[PROB_SIZE - 1]][i[0]];
    fitness[index] = 1 / distance;
	__syncthreads();
}

__global__ void crossover_and_migrate(
    population_t population,
    population_fitness_t fitness_prefix_sum,
    population_t new_population
) {
    int island = blockIdx.x;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
    individual_t &new_individual = new_population[index];

    // Get indices of the two parents randomly, weighted by fitness value
    // (uses thrust::lower_bound search on the prefix-summed fitness values)
    curandState s;
    curand_init(index, 0, 0, &s);

    int parent_1_idx = thrust::lower_bound(
        thrust::device,
        fitness_prefix_sum + island * ISLAND_POP_SIZE,
        fitness_prefix_sum + (island + 1) * ISLAND_POP_SIZE,
        curand_uniform(&s) * fitness_prefix_sum[POP_SIZE-1]
    ) - fitness_prefix_sum;
    individual_t &parent_1 = population[parent_1_idx];

    int parent_2_idx = thrust::lower_bound(
        thrust::device,
        fitness_prefix_sum + island * ISLAND_POP_SIZE,
        fitness_prefix_sum + (island + 1) * ISLAND_POP_SIZE,
        curand_uniform(&s) * fitness_prefix_sum[POP_SIZE-1]
    ) - fitness_prefix_sum;
    individual_t &parent_2 = population[parent_2_idx];

    // Initialise a bitmask of visited cities for crossover
    const int PROB_MASK_NUM_WORDS = (PROB_SIZE + 31) / 32;
    uint32_t prob_mask[PROB_MASK_NUM_WORDS];
    for (int j = 0; j < PROB_MASK_NUM_WORDS; j++) {
        prob_mask[j] = 0;
    }

    // Fill the first half of the child with the first parent
    // Store visited cities in a bitmask
    int *curr_gene = new_individual;
    for (int j = 0; j < PROB_SIZE / 2; j++) {
        *curr_gene++ = parent_1[j];
        prob_mask[parent_1[j] / 32] |= 1 << (parent_1[j] % 32);
    }

    // For the second half of the child we only choose
    // entries from parent 2 that aren't already in the child
    for (int j = 0; j < PROB_SIZE; j++) {
        if ((prob_mask[parent_2[j] / 32] & (1 << (parent_2[j] % 32))) == 0) {
            *curr_gene++ = parent_2[j];
        }
    }

    // //perform the migration
    // int target_island = (island + 1) % ISLANDS;
    // __shared__ int r_1, r_2;
    // if (threadIdx.x == 0) {
    //     r_1 = rand() % ISLAND_POP_SIZE;
    //     r_2 = rand() % ISLAND_POP_SIZE;
    // }
	__syncthreads();
    // //Each thread takes a single gene from one of five
    // int temp[PROB_SIZE * MIGRATION];
    // if (threadIdx.x != 0 && threadIdx.x < PROB_SIZE * MIGRATION) {
    //     int migrant_index = threadIdx.x - 1;
    //     temp[migrant_index] = NPOP(island_1 * ISLAND_POP_SIZE + r_1, remainder(migrant_index, PROB_SIZE));
    //     NPOP(island_1 * POP_SIZE / ISLAND + r_1, remainder(migrant_index, PROB_SIZE)) = NPOP(island_2 * ISLAND_POP_SIZE + r_2, remainder(migrant_index, PROB_SIZE))
    //     NPOP(island_2 * POP_SIZE / ISLAND + r_2, remainder(migrant_index, PROB_SIZE)) = temp[migrant_index];
    // }

    // Overwrite existing population with the newly created one
    for (int j = 0; j < PROB_SIZE; j++) {
        population[index][j] = new_individual[j];
    }
}

__global__ void mutate(population_t population){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    individual_t &i = population[index];

    curandState s;
    curand_init(index, 0, 0, &s);

    // TODO it would be way faster to select random pairs
    for (int j = 0; j < PROB_SIZE; j++){
        for (int k = 0; k < PROB_SIZE; k++) {
            if (curand_uniform(&s) < MUTATION_RATE) {
                int temp = i[j];
                i[j] = i[k];
                i[k] = temp;
            }
        }
    }
}

int main (void) {

    //arrays needed for problem
    individual_t *population, *new_population; // population_t
    float (*problem)[PROB_SIZE]; // problem_t
	float /**fitness, */*fitness_prefix_sum; // population_fitness_t
    int *sorted_fitness_idxs;
    thrust::device_vector<float> fitness(POP_SIZE);

	//random device
	random_device rd;
	mt19937 gen = mt19937(rd());

	//Unified memory (accessible from CPU and GPU)
	cudaMallocManaged(&problem, sizeof(problem_t));
	cudaMallocManaged(&population, sizeof(population_t));
	cudaMallocManaged(&new_population, sizeof(population_t));
	// cudaMallocManaged(&fitness, sizeof(population_fitness_t));
	cudaMallocManaged(&fitness_prefix_sum, sizeof(population_fitness_t));
	cudaMallocManaged(&sorted_fitness_idxs, POP_SIZE * sizeof(int));

	//initialize sequence of matrix indices to be used for initializing the population
	vector<int> tmp_indices(PROB_SIZE);
    	for (int i = 0; i < PROB_SIZE; ++i) {
        	tmp_indices[i] = i;
    	}

    //initialize problem (random for now, from file later)
    float problem_x[PROB_SIZE], problem_y[PROB_SIZE];
    uniform_real_distribution<float> dis(0.0f, 100.0f);
	for (int i = 0; i < PROB_SIZE; i++) {
        problem_x[i] = dis(gen);
        problem_y[i] = dis(gen);
    }
    for (int i = 0; i < PROB_SIZE; i++) {
        for (int j = 0; j < PROB_SIZE; j++) {
            problem[i][j] = sqrt(
                (problem_x[j] - problem_x[i]) * (problem_x[j] - problem_x[i]) +
                (problem_y[j] - problem_y[i]) * (problem_y[j] - problem_y[i])
            );
        }
    }

	//initialize population (random)
	for (int i = 0; i < POP_SIZE; ++i) {
        shuffle(tmp_indices.begin(), tmp_indices.end(), gen);
        for (int j = 0; j < PROB_SIZE; ++j) {
                population[i][j] = tmp_indices[j];
        }
    }

	//run GA
	for (int epoch = 0; epoch < EPOCHS; epoch++){
        //begin epoch by ranking fitness of population
        rank_individuals<<<ISLANDS, ISLAND_POP_SIZE>>>(
            population,
            problem,
            thrust::raw_pointer_cast(&fitness[0])
        );

        // cout << "before epoch " << epoch << ": " << endl;
        // for (int ind = 0; ind < POP_SIZE; ind++) {
        //     cout << "    " << fitness[ind];
        //     for (int gene = 0; gene < PROB_SIZE; gene++) {
        //         cout << " " << population[ind][gene];
        //     }
        //     cout << endl;
        // }
        cout << *thrust::max_element(fitness.begin(), fitness.end()) << endl;
        
        // TODO this actually changes order of fitness, but not population
		// //perform a sort of the fitness values to implement the elitism strategy
        // for (int i = 0; i < POP_SIZE; ++i) {
        //     sorted_fitness_idxs[i] = i;
        // }
        // for (int j = 0; j < ISLANDS; j++) {
        //     thrust::sort_by_key(
        //         thrust::device,
        //         fitness + j * ISLAND_POP_SIZE,
        //         fitness + (j + 1) * ISLAND_POP_SIZE,
        //         sorted_fitness_idxs + j * ISLAND_POP_SIZE
        //     );
        // }

        //perform a prefix sum of the fitness in order to easily perform roulette wheel selection of a suitable parent
        for (int j = 0; j < ISLANDS; j++) {
            thrust::inclusive_scan(
                thrust::device,
                &fitness[j * ISLAND_POP_SIZE],
                &fitness[(j + 1) * ISLAND_POP_SIZE],
                fitness_prefix_sum + j * ISLAND_POP_SIZE
            );
        }
        
		//perform crossover on the islands
		crossover_and_migrate<<<ISLANDS, ISLAND_POP_SIZE>>>(population, fitness_prefix_sum, new_population);
  
        //mutate the population
        mutate<<<ISLANDS, ISLAND_POP_SIZE>>>(population);
	}

	cudaFree(problem);
	cudaFree(population);
	cudaFree(new_population);
	// cudaFree(fitness);
	cudaFree(fitness_prefix_sum);
	cudaFree(sorted_fitness_idxs);
	
	return 0;
}
