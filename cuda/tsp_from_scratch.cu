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

#define POP_SIZE 64
#define PROB_SIZE 1291
#define PROB_FILE_NAME "d1291.tsp"
#define EPOCHS 10000
#define EPOCHS_REPORT_INTERVAL 1000
#define ISLANDS 1
#define MUTATION_RATE 0.05f
// #define MIGRATION 5	//integer determining how many members migrate from one island
#define ISLAND_POP_SIZE (POP_SIZE / ISLANDS)
#define ELITES (ISLAND_POP_SIZE / 2)

#undef DEBUG

typedef float problem_t[PROB_SIZE][PROB_SIZE];

typedef int individual_t[PROB_SIZE];
typedef individual_t population_t[POP_SIZE];

typedef float fitness_t;
typedef fitness_t population_fitness_t[POP_SIZE];

//https://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
//https://github.com/thrust/thrust/wiki/Quick-Start-Guide#fancy-iterators

using namespace std;

__device__ fitness_t rank_individual(
    const individual_t &ind,
    const problem_t problem
) {
    float distance = 0;
    // Iterate over edges of an individual
    for (int j = 0; j < PROB_SIZE - 1; j++) {
        int from = ind[j];
        int to = ind[j + 1];
        distance += problem[from][to];
    }
    int last_city = ind[PROB_SIZE - 1];
    int first_city = ind[0];
    distance += problem[last_city][first_city];
    float fitness = 1 / distance;
    fitness = fitness * fitness;
    fitness = fitness * fitness;
    return fitness;
}

__global__ void rank_individuals(
    const population_t population,
    const problem_t problem,
    population_fitness_t fitness
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    fitness[index] = rank_individual(population[index], problem);
}

__global__ void crossover_individuals(
    const problem_t problem,
    population_t population,
    population_fitness_t fitness,
    const population_fitness_t fitness_prefix_sum,
    population_t new_population,
    const int * const fitness_to_population
) {
    const int island = blockIdx.x;
    const int island_index = threadIdx.x;
    const int index = island * ISLAND_POP_SIZE + island_index;

    individual_t &new_individual = new_population[index];

    curandState s;
    curand_init(index, 0, 0, &s);
    
    // Get indices of the two parents randomly, weighted by fitness value
    // (uses thrust::lower_bound search on the prefix-summed fitness values)
    const float fitness_sum_first = fitness_prefix_sum[island * ISLAND_POP_SIZE];
    const float fitness_sum_last = fitness_prefix_sum[(island + 1) * ISLAND_POP_SIZE - 1];
    const float fitness_sum_diff = fitness_sum_last - fitness_sum_first;

    const int parent_1_fitness_idx = thrust::lower_bound(
        thrust::device,
        fitness_prefix_sum + island * ISLAND_POP_SIZE,
        fitness_prefix_sum + (island + 1) * ISLAND_POP_SIZE,
        fitness_sum_first + curand_uniform(&s) * fitness_sum_diff
    ) - 1 - fitness_prefix_sum;
    // Because fitness is sorted, get the respective population member index
    const int parent_1_population_idx = fitness_to_population[parent_1_fitness_idx];
    const individual_t &parent_1 = population[parent_1_population_idx];

    const int parent_2_fitness_idx = thrust::lower_bound(
        thrust::device,
        fitness_prefix_sum + island * ISLAND_POP_SIZE,
        fitness_prefix_sum + (island + 1) * ISLAND_POP_SIZE,
        fitness_sum_first + curand_uniform(&s) * fitness_sum_diff
    ) - 1 - fitness_prefix_sum;
    const int parent_2_population_idx = fitness_to_population[parent_2_fitness_idx];
    const individual_t &parent_2 = population[parent_2_population_idx];

    // Initialise a bitmask of visited cities for crossover
    const int PROB_MASK_NUM_WORDS = (PROB_SIZE + 31) / 32;
    uint32_t prob_mask[PROB_MASK_NUM_WORDS];
    for (int j = 0; j < PROB_MASK_NUM_WORDS; j++) {
        prob_mask[j] = 0;
    }

    // Don't cross over when dealing with elites
    bool elite = false;
    for (int i = ISLAND_POP_SIZE - ELITES; i < ISLAND_POP_SIZE; i++) {
        if (index == fitness_to_population[island * ISLAND_POP_SIZE + i])	{
            elite = true;
        }
    }

    if (elite == false) {
        // Partially fill the child with a random chunk of the first parent
        // Store visited cities in the bitmask
        int *curr_gene = new_individual;
        const int gene_a = ceilf(curand_uniform(&s) * PROB_SIZE) - 1;
        const int gene_b = ceilf(curand_uniform(&s) * PROB_SIZE) - 1;
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
    }
    __syncthreads();

    // TODO migration

    individual_t &old_individual = population[index];

    if (elite == false) {
        // Overwrite existing population with the newly created one
        for (int j = 0; j < PROB_SIZE; j++) {
            old_individual[j] = new_individual[j];
        }
    }

    // Mutate
    if (curand_uniform(&s) <= MUTATION_RATE) {
        int gene_a_idx = ceilf(curand_uniform(&s) * PROB_SIZE) - 1;
        int gene_b_idx = ceilf(curand_uniform(&s) * PROB_SIZE) - 1;
        int gene_a = old_individual[gene_a_idx];
        int gene_b = old_individual[gene_b_idx];
        old_individual[gene_a_idx] = gene_b;
        old_individual[gene_b_idx] = gene_a;
    }

    fitness[index] = rank_individual(old_individual, problem);
}

void solve(
    population_t population,
    problem_t problem,
    population_fitness_t fitness,
    population_fitness_t fitness_prefix_sum,
    population_t new_population,
    int *fitness_to_population,
    bool cold_start
) {
    if (cold_start) {
        // Begin epoch by ranking fitness of population
        rank_individuals<<<ISLANDS, ISLAND_POP_SIZE>>>(
            population,
            problem,
            fitness
        );
        cudaDeviceSynchronize();
    }
    
    // Perform a sort of the fitness values to implement the elitism strategy
    for (int i = 0; i < POP_SIZE; ++i) {
         fitness_to_population[i] = i;
    }
    for (int j = 0; j < ISLANDS; j++) {
         thrust::sort_by_key(
             thrust::host,
             fitness + j * ISLAND_POP_SIZE,
             fitness + (j + 1) * ISLAND_POP_SIZE,
             fitness_to_population + j * ISLAND_POP_SIZE
         );
    }

    // Perform a prefix sum of the fitness in order to easily perform roulette wheel selection of a suitable parent
    for (int j = 0; j < ISLANDS; j++) {
        thrust::inclusive_scan(
            thrust::host,
            fitness + j * ISLAND_POP_SIZE,
            fitness + (j + 1) * ISLAND_POP_SIZE,
            fitness_prefix_sum + j * ISLAND_POP_SIZE
        );
    }
    
    // Perform crossover on the islands
    crossover_individuals<<<ISLANDS, ISLAND_POP_SIZE>>>(
        problem,
        population,
        fitness,
        fitness_prefix_sum,
        new_population,
        fitness_to_population
    );
    cudaDeviceSynchronize();
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
        cout << "debug on" << endl;
    #endif

    // Run GA
    bool cold_start = true;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        solve(
            population,
            problem,
            fitness,
            fitness_prefix_sum,
            new_population,
            fitness_to_population,
            cold_start
        );
        cold_start = false;

        // cout << "before epoch " << epoch << ": " << endl;
        // for (int ind = 0; ind < POP_SIZE; ind++) {
        // 	 cout << "	" << fitness[ind];
        // 	 for (int gene = 0; gene < PROB_SIZE; gene++) {
        // 		 cout << " " << population[ind][gene];
        // 	 }
        // 	 cout << endl;
        // }
        if (epoch % EPOCHS_REPORT_INTERVAL == 0) {
            cudaDeviceSynchronize();
            cout << sqrt(sqrt(1 / *thrust::max_element(fitness, fitness + POP_SIZE))) << endl;
        }

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
    }

    cudaFree(problem);
    cudaFree(population);
    cudaFree(new_population);
    cudaFree(fitness);
    cudaFree(fitness_prefix_sum);
    cudaFree(fitness_to_population);
    
    return 0;
}
