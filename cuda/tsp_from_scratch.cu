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
    const problem_t problem,
    const individual_t &ind
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
    const problem_t problem,
    const population_t population,
    population_fitness_t fitness
) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    fitness[index] = rank_individual(problem, population[index]);
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

    fitness[index] = rank_individual(problem, old_individual);
}

// TODO use struct for arguments
void solve(
    problem_t problem_d,
    population_t population_d,
    population_t new_population_d,
    population_fitness_t fitness_h,
    population_fitness_t fitness_d,
    population_fitness_t fitness_prefix_sum_h,
    population_fitness_t fitness_prefix_sum_d,
    int *fitness_to_population_h,
    int *fitness_to_population_d,
    bool cold_start
) {
    if (cold_start) {
        // Begin epoch by ranking fitness of population
        rank_individuals<<<ISLANDS, ISLAND_POP_SIZE>>>(
            problem_d,
            population_d,
            fitness_d
        );
        cudaDeviceSynchronize();
    }
    
    cudaMemcpy(fitness_h, fitness_d, sizeof(population_fitness_t), cudaMemcpyDeviceToHost);
    // Perform a sort of the fitness values to implement the elitism strategy
    for (int i = 0; i < POP_SIZE; ++i) {
         fitness_to_population_h[i] = i;
    }
    for (int j = 0; j < ISLANDS; j++) {
         thrust::sort_by_key(
             thrust::host,
             fitness_h + j * ISLAND_POP_SIZE,
             fitness_h + (j + 1) * ISLAND_POP_SIZE,
             fitness_to_population_h + j * ISLAND_POP_SIZE
         );
    }
    
    // Perform a prefix sum of the fitness in order to easily perform roulette wheel selection of a suitable parent
    for (int j = 0; j < ISLANDS; j++) {
        thrust::inclusive_scan(
            thrust::host,
            fitness_h + j * ISLAND_POP_SIZE,
            fitness_h + (j + 1) * ISLAND_POP_SIZE,
            fitness_prefix_sum_h + j * ISLAND_POP_SIZE
        );
    }
    cudaMemcpy(fitness_d, fitness_h, sizeof(population_fitness_t), cudaMemcpyHostToDevice);
    cudaMemcpy(fitness_prefix_sum_d, fitness_prefix_sum_h, sizeof(population_fitness_t), cudaMemcpyHostToDevice);
    cudaMemcpy(fitness_to_population_d, fitness_to_population_h, sizeof(population_fitness_t), cudaMemcpyHostToDevice);
    
    // Perform crossover on the islands
    crossover_individuals<<<ISLANDS, ISLAND_POP_SIZE>>>(
        problem_d,
        population_d,
        fitness_d,
        fitness_prefix_sum_d,
        new_population_d,
        fitness_to_population_d
    );
    cudaDeviceSynchronize();
}

int main (void) {

    //arrays needed for problem
    individual_t *population_d, *new_population; // population_t
    float (*problem_d)[PROB_SIZE]; // problem_t
    float *fitness_d; // population_fitness_t
    float *fitness_prefix_sum_d; // population_fitness_t
    int *fitness_to_population_d;

    // random device
    random_device rd;
    mt19937 gen = mt19937(rd());

    float (* const problem_h)[PROB_SIZE] = (float (*)[PROB_SIZE])malloc(sizeof(problem_t));
    cudaMalloc(&problem_d, sizeof(problem_t));

    individual_t * const population_h = (individual_t *)malloc(sizeof(population_t));
    cudaMalloc(&population_d, sizeof(population_t));
    cudaMalloc(&new_population, sizeof(population_t));

    float * const fitness_h = (float *)malloc(sizeof(population_fitness_t));
    float * const fitness_prefix_sum_h = (float *)malloc(sizeof(population_fitness_t));
    int * const fitness_to_population_h = (int *)malloc(POP_SIZE * sizeof(*fitness_to_population_h));
    cudaMalloc(&fitness_d, sizeof(population_fitness_t));
    cudaMalloc(&fitness_prefix_sum_d, sizeof(population_fitness_t));
    cudaMalloc(&fitness_to_population_d, POP_SIZE * sizeof(*fitness_to_population_d));

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
            problem_h[i][j] = sqrt(
                (problem_x[j] - problem_x[i]) * (problem_x[j] - problem_x[i]) +
                (problem_y[j] - problem_y[i]) * (problem_y[j] - problem_y[i])
            );
        }
    }
    cudaMemcpy(problem_d, problem_h, sizeof(problem_t), cudaMemcpyHostToDevice);

    // Randomly initialize population
    for (int i = 0; i < POP_SIZE; ++i) {
        shuffle(tmp_indices.begin(), tmp_indices.end(), gen);
        for (int j = 0; j < PROB_SIZE; ++j) {
                population_h[i][j] = tmp_indices[j];
        }
    }
    cudaMemcpy(population_d, population_h, sizeof(population_t), cudaMemcpyHostToDevice);

    #ifdef DEBUG
        cout << "debug on" << endl;
    #endif

    // Run GA
    bool cold_start = true;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        solve(
            problem_d,
            population_d,
            new_population,
            fitness_h,
            fitness_d,
            fitness_prefix_sum_h,
            fitness_prefix_sum_d,
            fitness_to_population_h,
            fitness_to_population_d,
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
            cudaMemcpy(fitness_h, fitness_d, sizeof(population_fitness_t), cudaMemcpyDeviceToHost);
            cout << sqrt(sqrt(1 / *thrust::max_element(fitness_h, fitness_h + POP_SIZE))) << endl;
        }

        // Verify integrity of individuals
        #ifdef DEBUG
        {
            cudaMemcpy(population_h, population_d, sizeof(population_t), cudaMemcpyDeviceToHost)
            const int PROB_MASK_NUM_WORDS = (PROB_SIZE + 31) / 32;
            uint32_t prob_mask[PROB_MASK_NUM_WORDS];
            for (int i = 0; i < POP_SIZE; i++) {
                individual_t &ind = population_h[i];
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

    cudaFree(problem_d);
    cudaFree(population_d);
    cudaFree(new_population);
    cudaFree(fitness_d);
    cudaFree(fitness_prefix_sum_d);
    cudaFree(fitness_to_population_d);

    free(problem_h);
    free(population_h);
    free(fitness_h);
    free(fitness_prefix_sum_h);
    free(fitness_to_population_h);

    return 0;
}
