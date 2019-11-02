__global__ void breedPopulationGPU(float* population, float* fitness, int numCities, int popSize, int parent) {
	int island = blockIdx.x;
	int couple = blockIdx.y
	int gene = threadIdx;
	int temp_pop[2][numCities];
	
	
}

int main()
{
	int numIslands;
	int popSize;
	int numCities;
	int 
	breedPopulationGPU<<<numIslands, popSize>>>(population, fitness, numCities, popSize);
}
