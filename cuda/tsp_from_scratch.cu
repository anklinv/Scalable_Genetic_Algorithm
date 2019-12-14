#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>

#include <curand_kernel.h>

#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/extrema.h>

#define POP_SIZE 512
#define PROB_SIZE 48
#define PROB_FILE_NAME "att48.tsp"
#define EPOCHS 100000
#define EPOCHS_REPORT_INTERVAL 1000
#define ISLANDS 1
#define MUTATION_RATE 0.05f
// #define MIGRATION 5	//integer determining how many members migrate from one island
#define ISLAND_POP_SIZE (POP_SIZE / ISLANDS)
#define ELITES (ISLAND_POP_SIZE / 2)

#define DEBUG

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
    // Iterate over edges of an individual
    for (int j = 0; j < PROB_SIZE - 1; j++) {
        int from = i[j];
        int to = i[j + 1];
        distance += problem[from][to];
    }
    int last_city = i[PROB_SIZE - 1];
    int first_city = i[0];
    distance += problem[last_city][first_city];
    fitness[index] = 1 / distance;
}

__global__ void crossover_and_migrate(
    population_t population,
    population_fitness_t fitness_prefix_sum,
    population_t new_population,
    int *fitness_to_population
) {
    int island = blockIdx.x;
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    individual_t &new_individual = new_population[index];

    curandState s;
    curand_init(index, 0, 0, &s);
    
    // Get indices of the two parents randomly, weighted by fitness value
    // (uses thrust::lower_bound search on the prefix-summed fitness values)
    int parent_1_fitness_idx = thrust::lower_bound(
        thrust::device,
        fitness_prefix_sum + island * ISLAND_POP_SIZE,
        fitness_prefix_sum + (island + 1) * ISLAND_POP_SIZE,
        curand_uniform(&s) * fitness_prefix_sum[POP_SIZE-1]
    ) - fitness_prefix_sum;
    // Because fitness is sorted, get the respective population member index
    int parent_1_population_idx = fitness_to_population[parent_1_fitness_idx];
    individual_t &parent_1 = population[parent_1_population_idx];

    int parent_2_fitness_idx = thrust::lower_bound(
        thrust::device,
        fitness_prefix_sum + island * ISLAND_POP_SIZE,
        fitness_prefix_sum + (island + 1) * ISLAND_POP_SIZE,
        curand_uniform(&s) * fitness_prefix_sum[POP_SIZE-1]
    ) - fitness_prefix_sum;
    int parent_2_population_idx = fitness_to_population[parent_2_fitness_idx];
    individual_t &parent_2 = population[parent_2_population_idx];

    // Initialise a bitmask of visited cities for crossover
    const int PROB_MASK_NUM_WORDS = (PROB_SIZE + 31) / 32;
    uint32_t prob_mask[PROB_MASK_NUM_WORDS];
    for (int j = 0; j < PROB_MASK_NUM_WORDS; j++) {
        prob_mask[j] = 0;
    }

    // Don't cross over when dealing with elites
    for (int i = ISLAND_POP_SIZE - ELITES; i < ISLAND_POP_SIZE; i++) {
        if (index == fitness_to_population[island * ISLAND_POP_SIZE + i])	{
            return;
        }
    }

    // Partially fill the child with a random chunk of the first parent
    // Store visited cities in the bitmask
    int *curr_gene = new_individual;
    int gene_a = ceilf(curand_uniform(&s) * PROB_SIZE) - 1;
    int gene_b = ceilf(curand_uniform(&s) * PROB_SIZE) - 1;
    int gene_min, gene_max;
    if (gene_a > gene_b)	{ gene_max = gene_a; gene_min = gene_b; }
    else			{ gene_min = gene_a; gene_max = gene_b; }
    for (int j = gene_min; j < gene_max; j++) {
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
    //	 r_1 = rand() % ISLAND_POP_SIZE;
    //	 r_2 = rand() % ISLAND_POP_SIZE;
    // }
    __syncthreads();
    // //Each thread takes a single gene from one of five
    // int temp[PROB_SIZE * MIGRATION];
    // if (threadIdx.x != 0 && threadIdx.x < PROB_SIZE * MIGRATION) {
    //	 int migrant_index = threadIdx.x - 1;
    //	 temp[migrant_index] = NPOP(island_1 * ISLAND_POP_SIZE + r_1, remainder(migrant_index, PROB_SIZE));
    //	 NPOP(island_1 * POP_SIZE / ISLAND + r_1, remainder(migrant_index, PROB_SIZE)) = NPOP(island_2 * ISLAND_POP_SIZE + r_2, remainder(migrant_index, PROB_SIZE))
    //	 NPOP(island_2 * POP_SIZE / ISLAND + r_2, remainder(migrant_index, PROB_SIZE)) = temp[migrant_index];
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

    if (curand_uniform(&s) <= MUTATION_RATE) {
        int gene_a_idx = ceilf(curand_uniform(&s) * PROB_SIZE) - 1;
        int gene_b_idx = ceilf(curand_uniform(&s) * PROB_SIZE) - 1;
        int gene_a = i[gene_a_idx];
        int gene_b = i[gene_b_idx];
        i[gene_a_idx] = gene_b;
        i[gene_b_idx] = gene_a;
    }
}

int main (void) {

    //arrays needed for problem
    individual_t *population, *new_population; // population_t
    float (*problem)[PROB_SIZE]; // problem_t
    float *fitness, *fitness_prefix_sum; // population_fitness_t
    int *fitness_to_population;

    // random device
    random_device rd;
    mt19937 gen = mt19937(rd());

    //Unified memory (accessible from CPU and GPU)
    cudaMallocManaged(&problem, sizeof(problem_t));
    cudaMallocManaged(&population, sizeof(population_t));
    cudaMallocManaged(&new_population, sizeof(population_t));
    cudaMallocManaged(&fitness, sizeof(population_fitness_t));
    cudaMallocManaged(&fitness_prefix_sum, sizeof(population_fitness_t));
    cudaMallocManaged(&fitness_to_population, POP_SIZE * sizeof(int));

    //initialize sequence of matrix indices to be used for initializing the population
    vector<int> tmp_indices(PROB_SIZE);
    for (int i = 0; i < PROB_SIZE; ++i) {
        tmp_indices[i] = i;
    }

    // Initialize problem
    float problem_x[PROB_SIZE], problem_y[PROB_SIZE];

    // Generate random data
    // uniform_real_distribution<float> dis(0.0f, 100.0f);
    // for (int i = 0; i < PROB_SIZE; i++) {
    //     problem_x[i] = dis(gen);
    //     problem_y[i] = dis(gen);
    // }

    // Read data from a file
    std::ifstream f(PROB_FILE_NAME);
    char buf[80];
    int city;
    do {
        f.getline(buf, sizeof(buf) / sizeof(*buf));
    } while (strcmp(buf, "NODE_COORD_SECTION") != 0);
    for (int i = 0; i < PROB_SIZE; i++) {
        f >> city >> problem_x[i] >> problem_y[i];
    }

    // Calculate distance matrix
    for (int i = 0; i < PROB_SIZE; i++) {
        for (int j = 0; j < PROB_SIZE; j++) {
            problem[i][j] = sqrt(
                (problem_x[j] - problem_x[i]) * (problem_x[j] - problem_x[i]) +
                (problem_y[j] - problem_y[i]) * (problem_y[j] - problem_y[i])
            );
        }
    }

    // Randomly initialize population
    for (int i = 0; i < POP_SIZE; ++i) {
        shuffle(tmp_indices.begin(), tmp_indices.end(), gen);
        for (int j = 0; j < PROB_SIZE; ++j) {
                population[i][j] = tmp_indices[j];
        }
    }

    #ifdef DEBUG
        cerr << "debug on" << endl;
    #endif

    // Run GA
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        // Verify integrity of individuals
        #ifdef DEBUG
        {
            const int PROB_MASK_NUM_WORDS = (PROB_SIZE + 31) / 32;
            uint32_t prob_mask[PROB_MASK_NUM_WORDS];
            for (int i = 0; i < POP_SIZE; i++) {
                individual_t &ind = population[i];
                for (int j = 0; j < PROB_MASK_NUM_WORDS; j++) {
                    prob_mask[j] = 0;
                }
                for (int j = 0; j < PROB_SIZE; j++) {
                    if ((prob_mask[ind[j] / 32] & (1 << (ind[j] & 31))) != 0) {
                        cerr << "Assertion error: individuals traverse cities twice" << endl;
                    }
                    prob_mask[ind[j] / 32] |= (1 << (ind[j] & 31));
                }
            }
        }
        #endif
        // Begin epoch by ranking fitness of population
        rank_individuals<<<ISLANDS, ISLAND_POP_SIZE>>>(
            population,
            problem,
            fitness
        );
        cudaDeviceSynchronize();

        // cout << "before epoch " << epoch << ": " << endl;
        // for (int ind = 0; ind < POP_SIZE; ind++) {
        // 	 cout << "	" << fitness[ind];
        // 	 for (int gene = 0; gene < PROB_SIZE; gene++) {
        // 		 cout << " " << population[ind][gene];
        // 	 }
        // 	 cout << endl;
        // }
        if (epoch % EPOCHS_REPORT_INTERVAL == 0) {
            cout << 1 / *thrust::max_element(fitness, fitness + POP_SIZE) << endl;
        }
        
        // Perform a sort of the fitness values to implement the elitism strategy
        for (int i = 0; i < POP_SIZE; ++i) {
             fitness_to_population[i] = i;
        }
        for (int j = 0; j < ISLANDS; j++) {
             thrust::sort_by_key(
                 thrust::device,
                 fitness + j * ISLAND_POP_SIZE,
                 fitness + (j + 1) * ISLAND_POP_SIZE,
                 fitness_to_population + j * ISLAND_POP_SIZE
             );
        }
        // cout << fitness_to_population[0] << fitness_to_population[POP_SIZE-1] << endl;

        // Perform a prefix sum of the fitness in order to easily perform roulette wheel selection of a suitable parent
        for (int j = 0; j < ISLANDS; j++) {
            thrust::inclusive_scan(
                thrust::device,
                fitness + j * ISLAND_POP_SIZE,
                fitness + (j + 1) * ISLAND_POP_SIZE,
                fitness_prefix_sum + j * ISLAND_POP_SIZE
            );
        }
        
        // Perform crossover on the islands
        crossover_and_migrate<<<ISLANDS, ISLAND_POP_SIZE>>>(
            population,
            fitness_prefix_sum,
            new_population,
            fitness_to_population
        );
        cudaDeviceSynchronize();
  
        // Mutate the population
        mutate<<<ISLANDS, ISLAND_POP_SIZE>>>(population);
        cudaDeviceSynchronize();
    }

    cudaFree(problem);
    cudaFree(population);
    cudaFree(new_population);
    cudaFree(fitness);
    cudaFree(fitness_prefix_sum);
    cudaFree(fitness_to_population);
    
    return 0;
}
