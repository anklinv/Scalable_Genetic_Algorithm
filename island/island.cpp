
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
    int migrationAmount = this->migrationAmount;
    int numIndivsIsland = (this->tsp).population_count;
    
    
    for(int currPeriod = 0; currPeriod < this->numPeriods; currPeriod++) {
        
        // Run the GA for migrationPeriod iterations
        (this->tsp).solve(this->migrationPeriod);
        
        
        // Set up buffers for sending data
        double sendBufferFitness[migrationAmount];
        int sendBufferGenes[migrationAmount * numIntsGene];
        
        
        int* ranks = (this->tsp).getRanks();
        
        for(int indivIdx = 0; indivIdx < migrationAmount; indivIdx++) {
            
            sendBufferFitness[indivIdx] = (this->tsp).getFitness(ranks[indivIdx]);
            
            
            int* gene = (this->tsp).getGene(ranks[indivIdx]);
            
            for(int nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
                sendBufferGenes[(indivIdx * numNodes) + nodeIdx] = gene[nodeIdx];
            }
            
        }
        
        
        // Set up buffers for receiving data
        double receiveBufferFitness[migrationAmount * numProcesses];
        int receiveBufferGenes[migrationAmount * numNodes * numProcesses];
        
        
        // "Gathers data from all tasks and distribute the combined data to all tasks"
        // I suppose this is synchronized
        MPI_Allgather(sendBufferFitness, migrationAmount, MPI_DOUBLE,
                      receiveBufferFitness, migrationAmount * numProcesses, MPI_DOUBLE, MPI_COMM_WORLD);
        
        MPI_Allgather(sendBufferGenes, migrationAmount * numNodes, MPI_INT,
                      receiveBufferGenes, migrationAmount * numNodes * numProcesses, MPI_INT, MPI_COMM_WORLD);
        
        
        // Assemble result
        // - there is some redundancy as the process is receiving its own data
        // - this is an attempt to solve this more or less efficiently
        
        // Use (idx, fitness) pairs to facilitate sorting
        Individual incomingIndividuals[migrationAmount * numProcesses];
        
        for(int indivIdx = 0; indivIdx < migrationAmount * numProcesses; indivIdx++) {
            
            (incomingIndividuals[indivIdx]).idx = indivIdx;
            (incomingIndividuals[indivIdx]).fitness = receiveBufferFitness[indivIdx];
        }
        
        // Sort the incoming data in ascending order
        // (the data at the current island is already sorted)
        sort(incomingIndividuals, incomingIndividuals + (migrationAmount * numProcesses));
        
        
        // Use (idx, fitness) pairs
        Individual worstIslandIndividuals[migrationAmount * numProcesses];
        int helperIdx = 0;
        
        for(int indivIdx = numIndivsIsland - (migrationAmount * numProcesses);
            indivIdx < numIndivsIsland; indivIdx++) {
            
            (worstIslandIndividuals[helperIdx]).idx = ranks[indivIdx];
            (worstIslandIndividuals[helperIdx]).fitness = (this->tsp).getFitness(ranks[indivIdx]);
            
            helperIdx++;
        }
        
        
        // Determine how many individuals are going to be replaced
        int numReplaced = 0;
        
        int idxIncoming = 0;
        int idxIsland = 0;
        
        while(idxIncoming < migrationAmount * numProcesses && idxIsland < migrationAmount * numProcesses) {
            
            double currFitnessIncoming = (incomingIndividuals[idxIncoming]).fitness;
            double currFitnessIsland = (worstIslandIndividuals[idxIsland]).fitness;
            
            if(currFitnessIncoming < currFitnessIsland) { // incoming individual is better
                
                numReplaced++;
                
                idxIncoming++;
                idxIsland++;
                
            } else { // island individual is better
                
                idxIsland++;
            }
        }
        
        // Just throw out the numReplaced weakest individuals of the island
        // - the fitness is updated too
        // - the ranks stored in the TSP object are no longer correct after this step
        for(int indivIdx = 0; indivIdx < numReplaced; indivIdx++) {
            
            int* currGene = (this->tsp).getGene(ranks[(numIndivsIsland - 1) - indivIdx]);
            int* newGene = &receiveBufferGenes[incomingIndividuals[indivIdx].idx * numNodes];
            
            copyArray(newGene, currGene, numNodes);
            
            (this->tsp).setFitness(ranks[(numIndivsIsland - 1) - indivIdx],
                                   incomingIndividuals[indivIdx].fitness);
        }
        
    } // end numPeriods
    
    
    double bestLocalFitness;
    bestLocalFitness = (this->tsp).getMaxFitness();
    
    return bestLocalFitness;
}

