
#include "island.hpp"


using namespace std;


typedef struct Individual {
    
    int idx;
    double fitness;
    
    bool operator<(const Individual& other) const {
        this->fitness < other.fitness;
    }
    
} Individual;


void swapArrays(int* arrA, int* arrB, int length) {
    
    for(int idx = 0; idx < length; idx++) {
        
        int tmp = arrA[idx];
        arrA[idx] = arrB[idx];
        arrB[idx] = tmp;
    }
    
}


Island(TravellingSalesmanProblem* tsp, int migrationPeriod, int migrationAmount,
       int numPeriods) {
    
    this->tsp = tsp;
    
    this->migrationPeriod = migrationPeriod;
    this->migrationAmount = migrationAmount;
    this->numPeriods = numPeriods;
}


double Island::solve() {
    
    // this is needed for send and receive
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // additional local variables for convenience
    int migrationPeriod = this->migrationPeriod;
    int migrationAmount = this->migrationAmount;
    int numPeriods = this->numPeriods;
    TravellingSalesmanProblem* tsp = this->tsp;
    int numIndividualsIsland = tsp.population_count;
    int numNodes = tsp.problem_size;
    
    double bestLocalFitness;
    
    
    for(int it = 0; it < numPeriods; it++) {
        
        // do computation and get the migrationAmount individuals with
        // the highest fitness
        double res;
        res = tsp->solve(migrationPeriod);
        
        
        // set up buffers for sending data
        double* fitnessBuffer[migrationAmount];
        int* geneBuffer[migrationAmount * (tsp->problem_size)];
        
        vector<int> ranks = tsp->getRanks();
        for(int k = 0; k < migrationAmount; k++) {
            
            fitnessBuffer[k] = tsp->getFitness(ranks[k]);
            
            int* genes = getGenes(ranks[k]);
            for(int j = 0; j < (tsp -> problem_size); j++) {
                
                geneBuffer[k * (tsp -> problem_size) + j] = genes[j];
            }
        }
        
        // set up buffers for receiving data
        int sizeGeneReceive = migrationAmount * (tsp->problem_size) * size;
        int sizeFitnessReceive = migrationAmount * size;
        
        int geneReceive[sizeGeneReceive];
        double fitnessReceive[sizeFitnessReceive];
        
        
        // "Gathers data from all tasks and distribute the combined data to all tasks"
        // I suppose this is synchronous
        
        MPI_Allgather(geneBuffer, migrationAmount * (tsp->problem_size), MPI_INT,
                      geneReceive, migrationAmount * (tsp-problem_size) * size, MPI_INT, MPI_COMM_WORLD);
        
        MPI_Allgather(fitnessBuffer, migrationAmount, MPI_DOUBLE,
                      fitnessReceive, migrationAmount * size, MPI_INT, MPI_COMM_WORLD);
        
        
        // assemble result
        // - this maybe is computationally expensive
        // - there is some redundancy as the process is receiving its own data -> fix this
        // - this is an attempt to implement this more or less efficiently
        
        // use (idx, fitness) pairs
        Individual incomingIndividuals[migrationAmount * size];
        
        for(int idx = 0; idx < migrationAmount * size; idx++) {
            
            incomingIndividuals.idx = idx;
            incomingIndividuals.fitness = fitnessReceive[idx];
        }
        
        // sort the incoming data in ascending order
        // (the data at the current island is already sorted)
        sort(incomingIndividuals, incomingIndividuals + (migrationAmount * size));
        
        
        // use (idx, fitness) pairs
        Individual worstIslandIndividuals[migrationAmount * size];
        
        for(int idx = (numIndividualsIsland - 1) - (migrationAmount * size);
            idx <= numIndividualsIsland - 1; idx++) {
            
            worstIslandIndividuals.idx = ranks[idx];
            worstIslandIndividuals.fitness = getFitness(ranks[idx]);
        }
        
        // determine how many individuals are going to be replaced
        // once this is determined, just throw out the numReplaced weakest individuals of the island
        int numReplaced = 0;
        
        int idxIncoming = 0;
        int idxIsland = 0;
        
        while(idxIncoming < migrationAmount * size && idxIsland < migrationAmount * size) {
            
            currFitnessIncoming = incomingIndividuals[idx].fitness;
            currFitnessIsland = worstIslandIndividuals[idx].fitness;
            
            if(currFitnessIncoming < currFitnessIsland) { // incoming individual is better
                
                numReplaced++;
                
                idxIncoming++;
                idxIsland++;
                
            } else { // individual at island is better
                
                idxIsland++;
            }
        }
        
        // the "throwing out" is done by just replacing the genes of the corresponding individuals
        // - the fitness is updated too
        // - the ranks stored in the TSP object are no longer correct after this step
        for(int idx = 0; idx < numReplaced; idx++) {
            
            int* currGenes = getGenes(ranks[(numIndividualsIsland - 1) - idx]);
            int* newGenes = geneReceive[incomingIndividuals[idx].idx * numNodes]; // not sure if this is correct
            
            swapArrays(currGenes, newGenes, numNodes);
            
            // not sure if this is correct
            tsp.setFitness(ranks[(numIndividualsIsland - 1) - idx], incomingIndividuals[idx].fitness);
        }
        
    } // end numPeriods
    
    fitness = tsp.getFitness();
    bestLocalFitness = max_element(begin(fitness), end(fitness));
    
    return bestLocalFitness;
}


void Island::send() {}


void Island::receiveAll() {}

