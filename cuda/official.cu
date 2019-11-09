#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <chrono>

#include "cudnn.h"
#include "util.h"
#include "official.h"

#define PROBLEM_SIZE 48
#define POPULATION_SIZE (PROBLEM_SIZE * 20)
#define EPOCHS 100

using namespace std;

#define cudaCheckError() {																\
	cudaError_t e=cudaGetLastError();													\
	if(e!=cudaSuccess) {																\
		printf("Cuda failure %s:%d:'%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));	\
		exit(EXIT_FAILURE);																\
	}																					\
}

void evaluate_fitness(const int *population, double *fitness, double *nei, int xthr, int ythr) {
    for(int i = 0; i < POPULATION_SIZE/(PROBLEM_SIZE * PROBLEM_SIZE), i++){
        double route_distance = 0.0;
        for(int j = 0; j < PROBLEM_SIZE - 1; j++){
            route_distance += nei[population[j + xthr * ythr * PROBLEM_SIZE * i] + PROBLE_SIZE * individual[j + 1 + xthr * ythr * PROBLEM_SIZE * i]];   //matrix lookup for a distance between two cities
        }
        route_distance += nei[population[PROBLEM_SIZE - 1 + xthr * ythr * PROBLEM_SIZE * i] + PROBLE_SIZE * individual[0 + xthr * ythr * PROBLEM_SIZE * i]];
}

/*
    NOT DONE YET
*/
void crossover(int *population, double *fitness, int xthr, int ythr){
    int temp_population[PROBLEM_SIZE * 2];
    int exclusion_list

   // Keep the best individuals
   for (int i = 0; i < this->elite_size; ++i) {
       for (int j = 0; j < this->problem_size; ++j) {
           temp_population[i][j] = this->population[this->ranks[i] + this->population_count * j]; //this works
       }
   }

   vector<double> correct_fitness;
   correct_fitness.reserve(this->population_count);
   for (auto f : this->fitness) {
       correct_fitness.push_back(1 / pow(f / this->fitness_sum, 4));
   }

   auto dist = std::discrete_distribution<>(correct_fitness.begin(), correct_fitness.end());

   // Breed any random individuals
   for (int i = this->elite_size; i < this->population_count; ++i) {
   int rand1 = dist(gen);
   int rand2 = dist(gen);
   int* parent1 = new int[this->problem_size];
   int* parent2 = new int[this->problem_size];
   while (rand1 == rand2) {
       rand2 = dist(gen);
   }
   parent1 = this->getGene(rand1);
   parent2 = this->getGene(rand2);
       this->breed(
               parent1,
               parent2,
               temp_population[i]);

   }

   for (int i = 0; i < this->population_count; ++i) {
       for (int j = 0; j < this->problem_size; ++j) {
           this->population[i + this->population_count * j] = temp_population[i][j]; //this doesnt work
       // cout << this->population[i + this->population_count * j] << endl;
       }
   }





    int geneA  = rand_range(0, PROBLEM_SIZE - 1);
    int geneB = rand_range(0, PROBLEM_SIZE - 1);
    int startGene = min(geneA, geneB);
    int endGene = max(geneA, geneB);

    set<int> selected;
    for (int i = startGene; i <= endGene; ++i) {
        child[i] = parent1[i];
        selected.insert(parent1[i]);
    }

    int index = 0;
    for (int i = 0; i < PROBLEM_SIZE; ++i) {
        // If not already chosen that city
        if (selected.find(parent2[i]) == selected.end()) {
            if (index == startGene) {
                index = endGene + 1;
            }
            child[index] = parent2[i];
            index++;
        }
    }
}

__global__ void kernel_tsp_ga(float *pInputs, float *pOutputs) {
	int Inx = blockIdx.x; //island
    int Iny = blockIdx.y; //island
    int Inz = blockIdx.z; //island
	int x_thread = threadIdx.y; //gene (column) of population
	int y_thread = threadIdx.x; //member (row) of group
    int z_thread = threadIdx.z; //group of consecutive members (rows) of population
	extern __shared__ float input[];
    extern __shared__ int population[];
    extern __shared__ float fitness[];
 
    random_device rd;
    gen = mt19937(rd());
 
    //copy lookup matrix into shared memory (to be seen by all blocks)
    input[x_thread + y_thread * PROBLEM_SIZE] = pInputs[x_thread + y_thread * PROBLEM_SIZE];
    
    //instantiate population (no syncthreads needed) POPULATION_SIZE must be divisible by PROBLEM_SIZE
    vector<int> tmp_indices(problem_size);
    tmp_indices[x_thread] = x_thread;
    __syncthreads();
    for(int i = 0; i < POPULATION_SIZE/(PROBLEM_SIZE * PROBLEM_SIZE), i++){
        shuffle(tmp_indices.begin(), tmp_indices.end(), gen);
        for (int z = 0; z < PROBLEM_SIZE; z++){
            population[z + x_thread * y_thread * PROBLEM_SIZE * i] = tmp_indices[x_thread];
        }
    }
    __syncthreads();
    
    //start GA
    for (i = 0; i < EPOCHS){
        //fitness evaluation
        evaluate_fitness(population, fitness, input, x_thread, y_thread);
        __syncthreads();
        
        //crossover
        
    }
}

int traveling_salesman_problem() {

	float *input_ = get_parameter(inputName, PROBLEM_SIZE * PROBLEM_SIZE);
    
	float *input, *population;
	uint64_t nT1 = 0, nT2 = 0;
	cudaError_t s;

	/////////////////////////////////

	// My Kernel

	/////////////////////////////////


	/*  1. Data preparation  */
	float *t_input, *ip;
	
	float *dummyBTdB, *dummyOP, *dummyATIA, *dummyIN;

	int nInput = PROBLEM_SIZE * PROBLEM_SIZE;
	int nPopulation = PROBLEM_SIZE * POPULATION_SIZE;

	cudaMalloc((void **) &input, nInput*sizeof(float));
	cudaMalloc((void **) &population, nPopulation*sizeof(float));
	cudaMemset((void *) input, 0, nInput * sizeof(float));
	cudaMemset((void *) output, 0, nOutput*sizeof(float));
	cudaMemcpy(input, input_, nInput*sizeof(float), cudaMemcpyHostToDevice);
	
	float output[nPopulation];

	

	/*  2. Computing  */
	nT1 = getTimeMicroseconds64();
	//cudaProfilerStart();

	//###B'dB the first matrix multiplication for data transform###
	kernel_tsp <<<dim3(BLOCK_NUM,BLOCK_NUM, (CHANNEL_NUMBER/128)), dim3(128, WINO_KERNEL_SIZE),(WINO_KERNEL_SIZE*WINO_KERNEL_SIZE*128)*sizeof(float) >>> (input, t_input);
	cudaDeviceSynchronize();

	cudaError_t errorBTDB = cudaGetLastError();
	if (errorBTDB != cudaSuccess) {
  		fprintf(stderr, "ERROR BTDB: %s \n", cudaGetErrorString(errorBTDB));
	}

	cudaFuncSetCacheConfig (kernel_tsp_ga, cudaFuncCachePreferShared); //maybe not necessary

	//cudaProfilerStop();
	nT2 = getTimeMicroseconds64();

	printf("TotalTime = %lu us\n", nT2-nT1); 


	/*  3. Copy back and free  */
	s = cudaMemcpy(tmp_winograd, output, nOutput*sizeof(float), cudaMemcpyDeviceToHost);
	printf("%s\n", cudaGetErrorName(s));
	//cudaCheckError();

	cudaFree(t_input);
	cudaFree(output);

	output_checker(tmp_winograd, tmp_cudnn, FEAT_SIZE, CHANNEL_NUMBER, 1);

	return ((nT2-nT1) << 16) | (nT2_cudnn-nT1_cudnn);
}
