
#include "island.hpp"


using namespace std;


double Island::solve() {
    
    // For MPI_Allgather and to compute how much data is received
    int rank, numProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    
    // Local helper variables for convenience
    int numNodes = (this->tsp).problem_size;
    int numIntsGene = numNodes;
    
    
    int migrationPeriod = this->migrationPeriod;
    int migrationAmount = this->migrationAmount;
    int numPeriods = this->numPeriods;
    TravellingSalesmanProblem* tsp = this->tsp;
    int numIndividualsIsland = tsp.population_count;
    
    
    for(int currPeriod = 0; currPeriod < numPeriods; currPeriod++) {
        
        // Run the GA for migrationPeriod iterations
        (this->tsp).solve(migrationPeriod);
        
        
        // Set up buffers for sending data
        double* sendBufferFitness[migrationAmount];
        int* sendBufferGenes[migrationAmount * numIntsGene];
        
        
        int* ranks = (this->tsp).getRanks();
        
        for(int indivIdx = 0; indivIdx < migrationAmount; indivIdx++) {
            
            sendBufferFitness[indivIdx] = (this->tsp).getFitness(ranks[indivIdx]);
            
            
            int* gene = getGene(ranks[indivIdx]);
            
            for(int nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
                sendBufferGenes[(indivIdx * numNodes) + nodeIdx] = gene[nodeIdx];
            }
            
        }
        
        
        // Set up buffers for receiving data
        int sizeGeneReceive = migrationAmount * (tsp->problem_size) * numProcesses;
        int sizeFitnessReceive = migrationAmount * numProcesses;
        
        int geneReceive[sizeGeneReceive];
        double fitnessReceive[sizeFitnessReceive];
        
        
        // "Gathers data from all tasks and distribute the combined data to all tasks"
        // I suppose this is synchronous
        
        MPI_Allgather(geneBuffer, migrationAmount * (tsp->problem_size), MPI_INT,
                      geneReceive, migrationAmount * (tsp-problem_size) * numProcesses, MPI_INT, MPI_COMM_WORLD);
        
        MPI_Allgather(fitnessBuffer, migrationAmount, MPI_DOUBLE,
                      fitnessReceive, migrationAmount * numProcesses, MPI_INT, MPI_COMM_WORLD);
        
        
        // assemble result
        // - this maybe is computationally expensive
        // - there is some redundancy as the process is receiving its own data -> fix this
        // - this is an attempt to implement this more or less efficiently
        
        // use (idx, fitness) pairs
        Individual incomingIndividuals[migrationAmount * numProcesses];
        
        for(int idx = 0; idx < migrationAmount * numProcesses; idx++) {
            
            incomingIndividuals.idx = idx;
            incomingIndividuals.fitness = fitnessReceive[idx];
        }
        
        // sort the incoming data in ascending order
        // (the data at the current island is already sorted)
        sort(incomingIndividuals, incomingIndividuals + (migrationAmount * numProcesses));
        
        
        // use (idx, fitness) pairs
        Individual worstIslandIndividuals[migrationAmount * numProcesses];
        
        for(int idx = (numIndividualsIsland - 1) - (migrationAmount * numProcesses);
            idx <= numIndividualsIsland - 1; idx++) {
            
            worstIslandIndividuals.idx = ranks[idx];
            worstIslandIndividuals.fitness = getFitness(ranks[idx]);
        }
        
        // determine how many individuals are going to be replaced
        // once this is determined, just throw out the numReplaced weakest individuals of the island
        int numReplaced = 0;
        
        int idxIncoming = 0;
        int idxIsland = 0;
        
        while(idxIncoming < migrationAmount * numProcesses && idxIsland < migrationAmount * numProcesses) {
            
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
    double bestLocalFitness = max_element(begin(fitness), end(fitness));
    
    return bestLocalFitness;
}

