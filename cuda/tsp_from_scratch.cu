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

// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>

#define POP_SIZE 1000
#define PROB_SIZE 48
#define EPOCHS 100
#define ISLANDS 4
#define MUTATION_RATE 0.15
#define MIGRATION 5    //integer determining how many members migrate from one island
#define POP(i,j) population[i * PROB_SIZE + j]
#define NPOP(i,j) new_population[i * PROB_SIZE + j]
#define DIST(i,j) problem[i * PROB_SIZE + j]
#define ISLAND_POP_SIZE POP_SIZE / ISLANDS

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
    population_fitness_t fitness,
    population_fitness_t fitness_prefix_sum,
    population_t new_population
) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
    individual_t &new_individual = new_population[index];
    //get indices of the two parents by a random variable that has the normalized fitness values as distributions (use the thrust lower bound on the prefix sorted fitness values)
    curandState s;
    curand_init(index, 0, 0, &s);
    // TODO search within a specific single island
    int parent_1_idx = thrust::lower_bound(
        thrust::device,
        fitness_prefix_sum,
        // fitness_prefix_sum + ISLAND_POP_SIZE,
        fitness_prefix_sum + POP_SIZE,
        curand_uniform(&s) * fitness_prefix_sum[POP_SIZE-1]
    ) - fitness_prefix_sum;
    int parent_2_idx = parent_1_idx; // TODO when parent1 is correct, copy code to parent2
    // int parent2_id = thrust::lower_bound(
    //     thrust::device, fitness_prefix_sum, fitness_prefix_sum + ISLAND_POP_SIZE, curand_uniform(curand_state)
    // ) - fitness_prefix_sum - 1;
    individual_t &parent_1 = population[parent_1_idx];
    individual_t &parent_2 = population[parent_2_idx];
	
    //fill the first half of the child with the first parent
    for (int j = 0; j < PROB_SIZE / 2; j++) {
        new_individual[j] = parent_1[j];
    }
		
    //fill the rest of the child with -1 entries such that it is no longer garbage
    // TODO is that even necessary?
    for (int j = PROB_SIZE / 2; j < PROB_SIZE; j++) {
        new_individual[j] = -1;
    }
		
    //for the second half of the child we only choose entries from parent 2 that aren't already in the child
    // TODO implement with masking
    for (int j = PROB_SIZE / 2; j < PROB_SIZE; j++) {
        for (int k = 0; k < PROB_SIZE; k++) {
            int existing = thrust::find(
                thrust::device,
                new_individual,
                new_individual + j,
                parent_2[k]
            ) - new_individual;
            // if this element does not feature in new_individual, add it
            if (existing == j) {
                new_individual[j] = existing;
                break;
            }
        }
    }
    // //perform the migration
    // if (threadIdx.x == 0) {
    //     island_1 = index % ISLANDS;
    //     island_2 = (island_1 + 1) % ISLANDS;
    //     r_1 = rand() % ISLAND_POP_SIZE;
    //     r_2 = rand() % ISLAND_POP_SIZE;
    // }
	// __syncthreads();
    // //Each thread takes a single gene from one of five
    // int temp[PROB_SIZE * MIGRATION];
    // if (threadIdx.x != 0 && threadIdx.x < PROB_SIZE * MIGRATION) {
    //     int migrant_index = threadIdx.x - 1;
    //     // TODO where does island_1, r_1 etc come from? are they shared?
    //     temp[migrant_index] = NPOP(island_1 * ISLAND_POP_SIZE + r_1, remainder(migrant_index, PROB_SIZE));
    //     NPOP(island_1 * POP_SIZE / ISLAND + r_1, remainder(migrant_index, PROB_SIZE)) = NPOP(island_2 * ISLAND_POP_SIZE + r_2, remainder(migrant_index, PROB_SIZE))
    //     NPOP(island_2 * POP_SIZE / ISLAND + r_2, remainder(migrant_index, PROB_SIZE)) = temp[migrant_index];
    // }
}

__global__ void mutate(population_t population){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    individual_t &i = population[index];

    curandState s;
    curand_init(index, 0, 0, &s);

    for (int j = 0; j < PROB_SIZE; j++){
        for (int k = 0; k < PROB_SIZE; k++) {
            // TODO this will favour early elements
            if (curand_uniform(&s) < MUTATION_RATE) {
                int temp = i[j];
                i[j] = i[k];
                i[k] = temp;
                break;
            }
        }
    }
}

int main (void) {

    //arrays needed for problem
    individual_t *population, *new_population; // population_t
    float (*problem)[PROB_SIZE]; // problem_t
	float *fitness, *fitness_prefix_sum; // population_fitness_t
    // int *sorted_fitness;

	//random device
	random_device rd;
	mt19937 gen = mt19937(rd());

	//Unified memory (accessible from CPU and GPU)
	cudaMallocManaged(&problem, sizeof(problem_t));
	cudaMallocManaged(&population, sizeof(population_t));
	cudaMallocManaged(&new_population, sizeof(population_t));
	cudaMallocManaged(&fitness, sizeof(population_fitness_t));
	cudaMallocManaged(&fitness_prefix_sum, sizeof(population_fitness_t));
	// cudaMallocManaged(sorted_fitness, POP_SIZE *sizeof(int));

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

	//initialize population (random) and fitness (zero)
	for (int i = 0; i < POP_SIZE; ++i) {
        	shuffle(tmp_indices.begin(), tmp_indices.end(), gen);
        	for (int j = 0; j < PROB_SIZE; ++j) {
            		population[i][j] = tmp_indices[j];
        	}
		fitness[i] = 0.0f;
		// sorted_fitness[i] = i;
    }

	//run GA
	for (int i = 0; i < EPOCHS; i++){
		//begin epoch by ranking fitness of population
		rank_individuals<<<ISLANDS, ISLAND_POP_SIZE>>>(population, problem, fitness);
		
		//perform a sort of the fitness values to implement the elitism strategy
        // for (int j = 0; j < ISLANDS; j++) {
        //     thrust::sort_by_key(sorted_fitness + j * ISLAND_POP_SIZE, sorted_fitness + (j + 1) * ISLAND_POP_SIZE, fitness + j * ISLAND_POP_SIZE);
        // }

        //perform a prefix sum of the fitness in order to easily perform roulette wheel selection of a suitable parent
        // TODO calculate inclusive scan within islands
        // for (int j = 0; j < ISLANDS; j++) {
        //     thrust::inclusive_scan(fitness + j * ISLAND_POP_SIZE, fitness + (j + 1) * ISLAND_POP_SIZE, fitness_prefix_sum + j * ISLAND_POP_SIZE);
        // }
        thrust::inclusive_scan(fitness, fitness + POP_SIZE, fitness_prefix_sum);
        
		//perform crossover on the islands
		crossover_and_migrate<<<ISLANDS, ISLAND_POP_SIZE>>>(population, fitness, fitness_prefix_sum, new_population);
  
        //mutate the population
        mutate<<<ISLANDS, ISLAND_POP_SIZE>>>(population);

        // float best_fitness = 0.0f;
        cout << fitness[0] << endl;
	}

	cudaFree(problem);
	cudaFree(population);
	cudaFree(new_population);
	cudaFree(fitness);
	cudaFree(fitness_prefix_sum);
	// cudaFree(sorted_fitness);
	
	return 0;
}
